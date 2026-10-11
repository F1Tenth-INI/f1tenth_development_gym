"""Episode memory shared by the HWM planner (client) and learner (server).

Rows are ``[state_t, action_t]``: the raw 10-d car state at control step ``t`` and the
action in network output units (``physical / action_denorm``). Not clipped to
``[-1, 1]``. Nothing else is stored.

All rows live in one ring buffer of ``Settings.HWM_MEMORY_MAX_ROWS`` slots. Row ``g``
(its global insertion index) sits in slot ``g % capacity``; once the buffer is full,
every new row overwrites the oldest one. Rows arrive in time order from a single
actor, so one episode occupies consecutive indices.

Every stored superstate (``S = HWM_SUPER_STATE_SIZE`` consecutive rows of one episode)
also has an embedding from the LLD ``attention_norm_projector``, kept in the slot of its
last row. New rows are embedded on arrival and ``refresh_embeddings`` re-embeds all of
them after the projector weights change. Embeddings are computed without gradients.

With ``Settings.HWM_MEMORY_BANK`` the rows are never evicted: the row arrays grow, row ``g`` sits in
slot ``g`` (``g % capacity`` as before, ``capacity`` only grows). The LLD's keys (superstates) are
then a separate, limited array of ``HWM_MEMORY_BANK_SIZE`` indices into the rows, each ending a window of
``S`` rows plus the step it predicts (one episode; windows may overlap). Every row that completes a window
is offered as a key. When the bank is full an offered key replaces the key with the highest n_soft (its
kernel mass among the other keys) only if its own n_soft is lower, so dense regions are thinned first.
Embeddings and every n_soft are recomputed by ``refresh_embeddings``.

The client appends one row per control step and streams row slices to the learner;
the learner appends the received slices. Both sides therefore hold the same rows
(modulo dropped frames).
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch
from torch import nn

from TrainingLite.hwm.shared.observation import _normalize_rows, lld_rows_for_super, lld_super_flat_dim
from utilities.Settings import Settings

# Slots scanned per distance block; temporaries are (B, _SEARCH_CHUNK) floats.
_SEARCH_CHUNK = 65536
_EMBED_CHUNK = 65536


class MemoryManager:
    def __init__(
        self,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        device: str | torch.device = "cpu",
        capacity: Optional[int] = None,
        superstate_size: Optional[int] = None,
        overlap: bool = False,
        bank: Optional[bool] = None,
    ):
        # True: every row that ends a full window is a key (keys may share rows). False: tiled per episode.
        self.overlap = bool(overlap)
        self.state_dim = int(Settings.HWM_STATE_DIM if state_dim is None else state_dim)
        self.action_dim = int(Settings.HWM_ACTION_DIM if action_dim is None else action_dim)
        self.row_dim = self.state_dim + self.action_dim
        self.capacity = int(Settings.HWM_MEMORY_MAX_ROWS if capacity is None else capacity)
        self.superstate_size = int(
            Settings.HWM_SUPER_STATE_SIZE if superstate_size is None else superstate_size
        )
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.superstate_size < 1:
            raise ValueError(f"superstate_size must be at least 1, got {self.superstate_size}")
        self.device = torch.device(device)

        # Ingest (event loop) and training (worker thread) touch the buffer concurrently on the learner.
        self._lock = threading.RLock()
        self._rows = torch.zeros((self.capacity, self.row_dim), dtype=torch.float32, device=self.device)
        # Global insertion index held by each slot, -1 while empty.
        self._steps = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        # Number of consecutive rows of the same episode ending at each row (1 at the episode start).
        self._run_len = torch.zeros((self.capacity,), dtype=torch.long, device=self.device)
        # True for the last HWM_MEMORY_DROP_FINAL_STEPS rows of a finished episode. Such rows
        # are never a neighbor-bank transition and never inside a training batch.
        self._final = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)
        # Kernel density of each stored key among the other keys, for HWM_LLD_DENSITY_WEIGHTING.
        # NaN until computed; recomputed with the embeddings, filled in on demand for new rows.
        self._density = torch.full((self.capacity,), float("nan"), dtype=torch.float32, device=self.device)
        self._projector: Optional[nn.Module] = None
        self._query_projector: Optional[nn.Module] = None
        self._emb: Optional[torch.Tensor] = None
        self._clock = 0
        self._size = 0
        self._open_episode: Optional[int] = None
        self._open_start = 0
        self._episodes_closed = 0
        # HWM_MEMORY_BANK: rows are never evicted, the keys are a limited array of row indices.
        self.bank = bool(getattr(Settings, "HWM_MEMORY_BANK", False)) if bank is None else bool(bank)
        self.bank_size = int(getattr(Settings, "HWM_MEMORY_BANK_SIZE", 1000))
        self.bank_exclude_overlap = bool(getattr(Settings, "HWM_MEMORY_BANK_EXCLUDE_OVERLAP", True))
        if self.bank:
            if self.bank_size <= 0:
                raise ValueError(f"HWM_MEMORY_BANK_SIZE must be positive, got {self.bank_size}")
            # Global index of the last row of each key (-1 empty), its embedding and its n_soft.
            self._bank_rows = torch.full((self.bank_size,), -1, dtype=torch.long, device=self.device)
            self._bank_emb: Optional[torch.Tensor] = None
            self._bank_density = torch.zeros((self.bank_size,), dtype=torch.float32, device=self.device)
            self._bank_count = 0
            # Row slot -> position in the bank, -1 if the row ends no key.
            self._bank_index = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def attach_projector(self, projector: nn.Module, query_projector: Optional[nn.Module] = None) -> None:
        """Use ``projector`` for stored keys and ``query_projector`` for live queries.

        ``query_projector`` defaults to ``projector``, which is the shared-encoder case.
        Stored rows are embedded with the key projector.
        """
        with self._lock:
            self._projector = projector
            self._query_projector = projector if query_projector is None else query_projector
            probe = self._embed(torch.zeros((1, lld_super_flat_dim()), device=self.device))
            if self.bank:
                self._emb = None
                self._bank_emb = torch.zeros(
                    (self.bank_size, int(probe.shape[-1])), dtype=torch.float32, device=self.device
                )
                self._bank_rebuild()
                return
            self._emb = torch.zeros(
                (self.capacity, int(probe.shape[-1])), dtype=torch.float32, device=self.device
            )
            self.refresh_embeddings()

    def refresh_embeddings(self) -> None:
        """Re-embed every stored superstate with the current projector weights."""
        with self._lock:
            if self._projector is None:
                return
            if self.bank:
                count = self._bank_count
                for chunk in torch.arange(count, device=self.device).split(_EMBED_CHUNK):
                    self._bank_emb[chunk] = self._embed(self._key_rows(self._bank_rows[chunk]))
                self._bank_recompute_density()
                return
            slots = torch.nonzero(self._keyed_mask()).squeeze(1)
            for chunk in slots.split(_EMBED_CHUNK):
                self._emb[chunk] = self._embed(self._key_rows(self._steps[chunk]))
            self._density.fill_(float("nan"))
            if bool(getattr(Settings, "HWM_LLD_DENSITY_WEIGHTING", False)):
                self._fill_density(torch.nonzero(self._match_mask()).squeeze(1))

    @torch.no_grad()
    def _fill_density(self, slots: torch.Tensor) -> None:
        """Compute the missing densities of the match ``slots``.

        The density of a key is ``sum_k exp(-||e_j - e_k|| / T)`` over every match key ``k``,
        itself included, from the cached key embeddings and the LLD's attention temperature:
        ``n_soft`` of the key. Keys that already have a value keep it.
        """
        if slots.numel() == 0:
            return
        missing = slots[torch.isnan(self._density[slots])]
        if missing.numel() == 0:
            return
        temperature = float(Settings.HWM_LLD_ATTENTION_TEMPERATURE)
        reference = self._emb[slots]
        for chunk in missing.split(1024):
            distance = torch.cdist(self._emb[chunk], reference)
            self._density[chunk] = torch.exp(-distance / temperature).sum(dim=1)

    def embed(self, superstates: torch.Tensor) -> torch.Tensor:
        """Key-encoder embedding ``(B, emb)`` of normalized flat superstates ``(B, S * row_dim)``."""
        return self._embed_locked(self._projector, superstates)

    def embed_query(self, superstates: torch.Tensor) -> torch.Tensor:
        """Query-encoder embedding. The key encoder when no separate query projector was attached."""
        projector = getattr(self, "_query_projector", None) or self._projector
        return self._embed_locked(projector, superstates)

    def _embed_locked(self, projector: nn.Module, superstates: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            # The lock only guards swapping the projector, which compiled code cannot trace.
            return self._run_projector(projector, superstates.to(self.device))
        with self._lock:
            if projector is None:
                raise RuntimeError("embed needs a projector; call attach_projector first")
            return self._run_projector(projector, superstates.to(self.device))

    @torch.no_grad()
    def _embed(self, superstates: torch.Tensor) -> torch.Tensor:
        return self._run_projector(self._projector, superstates)

    @torch.no_grad()
    def _embed_query(self, superstates: torch.Tensor) -> torch.Tensor:
        projector = getattr(self, "_query_projector", None) or self._projector
        return self._run_projector(projector, superstates)

    def _run_projector(self, projector: nn.Module, superstates: torch.Tensor) -> torch.Tensor:
        param = next(projector.parameters())
        out = projector(superstates.to(device=param.device, dtype=param.dtype))
        return out.to(device=self.device, dtype=torch.float32)

    def _key_rows(self, last_steps: torch.Tensor) -> torch.Tensor:
        """Normalized, flattened superstates ending at ``last_steps``, oldest row first."""
        offset = torch.arange(self.superstate_size, device=self.device) - (self.superstate_size - 1)
        rows = self._rows[(last_steps[:, None] + offset) % self.capacity]
        return lld_rows_for_super(_normalize_rows(rows, self.state_dim), self.state_dim).flatten(start_dim=1)

    def _keyed_mask(self) -> torch.Tensor:
        """Slots whose superstate lies in one episode and is fully stored."""
        first = self._steps - (self.superstate_size - 1)
        return (first >= self._oldest) & (self._run_len >= self.superstate_size)

    def _match_mask(self) -> torch.Tensor:
        """Slots that end a superstate whose next row is stored in the same episode.

        Windows are ``S + 1`` rows (the superstate plus the step it predicts) and are
        tiled per episode, so no two share a row. The first window is the first
        ``S + 1`` rows; the next starts immediately after it.
        """
        if self.bank:
            return self._bank_index >= 0
        g = self._steps
        nxt = (g + 1) % self.capacity
        width = self.superstate_size + 1
        run = self._run_len[nxt]
        tiled = torch.ones_like(run, dtype=torch.bool) if self.overlap else (run % width == 0)
        return (
            (g - (self.superstate_size - 1) >= self._oldest)
            & (g + 1 < self._clock)
            & (run >= width)
            & tiled
            & ~self._final
            & ~self._final[nxt]
        )

    def _key_emb(self, slots: torch.Tensor) -> torch.Tensor:
        """Embeddings of the keys ending at row ``slots``."""
        if self.bank:
            return self._bank_emb[self._bank_index[slots]]
        return self._emb[slots]

    def _key_density(self, slots: torch.Tensor) -> torch.Tensor:
        """n_soft of the keys ending at row ``slots``."""
        if self.bank:
            return self._bank_density[self._bank_index[slots]].clone()
        self._fill_density(slots)
        return self._density[slots].clone()

    # ------------------------------------------------------------------
    # Key bank (HWM_MEMORY_BANK)
    # ------------------------------------------------------------------
    def _bank_kernel(self, emb: torch.Tensor, step: int, count: int) -> torch.Tensor:
        """Kernel ``exp(-d / T)`` of an embedding to the first ``count`` keys, ``(count,)``.

        Keys that share a row with the window ending at ``step`` (``|Δg| <= S``, hence the same episode)
        count as zero when ``HWM_MEMORY_BANK_EXCLUDE_OVERLAP`` is on: they are near copies of it.
        """
        temperature = float(Settings.HWM_LLD_ATTENTION_TEMPERATURE)
        kernel = torch.exp(-torch.linalg.norm(self._bank_emb[:count] - emb[None], dim=-1) / temperature)
        if self.bank_exclude_overlap:
            kernel = kernel.masked_fill((self._bank_rows[:count] - int(step)).abs() <= self.superstate_size, 0.0)
        return kernel

    def _bank_recompute_density(self) -> None:
        """n_soft of every key from the current embeddings: ``1 + sum of the kernel to the other keys``."""
        count = self._bank_count
        if count == 0:
            return
        temperature = float(Settings.HWM_LLD_ATTENTION_TEMPERATURE)
        emb = self._bank_emb[:count]
        steps = self._bank_rows[:count]
        for chunk in torch.arange(count, device=self.device).split(1024):
            kernel = torch.exp(-torch.cdist(emb[chunk], emb) / temperature)
            if self.bank_exclude_overlap:
                kernel = kernel.masked_fill((steps[chunk][:, None] - steps[None, :]).abs() <= self.superstate_size, 0.0)
                self._bank_density[chunk] = 1.0 + kernel.sum(dim=1)
            else:
                self._bank_density[chunk] = kernel.sum(dim=1)  # includes the key itself

    def _bank_offer(self, step: int, emb: torch.Tensor) -> bool:
        """Offer the key ending at row ``step`` with embedding ``emb``. True if it entered the bank."""
        count = self._bank_count
        kernel = self._bank_kernel(emb, step, count)
        n_in = 1.0 + float(kernel.sum())
        if count < self.bank_size:
            slot = count
            self._bank_count = count + 1
        else:
            slot = int(torch.argmax(self._bank_density[:count]))
            if not n_in < float(self._bank_density[slot]):
                return False
            # Remove the densest key: the others lose its kernel, the new key does not see it.
            gone = self._bank_kernel(self._bank_emb[slot], int(self._bank_rows[slot]), count)
            gone[slot] = 0.0
            self._bank_density[:count] -= gone
            self._bank_index[int(self._bank_rows[slot]) % self.capacity] = -1
            kernel[slot] = 0.0
            n_in = 1.0 + float(kernel.sum())
        self._bank_density[:count] += kernel
        self._bank_rows[slot] = int(step)
        self._bank_emb[slot] = emb
        self._bank_density[slot] = n_in
        self._bank_index[int(step) % self.capacity] = slot
        return True

    def _bank_offer_steps(self, steps: torch.Tensor) -> None:
        """Offer the keys ending at the global rows ``steps`` (ascending), in order."""
        for chunk in steps.split(_EMBED_CHUNK):
            emb = self._embed(self._key_rows(chunk))
            for step, vector in zip(chunk.tolist(), emb):
                self._bank_offer(step, vector)

    def _bank_candidates(self, first: int, last: int) -> torch.Tensor:
        """Rows ``g`` in ``[first, last]`` that end a complete window: ``S`` rows plus the next, one episode."""
        start = max(int(first), self._oldest)
        if int(last) < start:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        g = torch.arange(start, int(last) + 1, device=self.device)
        nxt = (g + 1) % self.capacity
        ok = (
            (self._run_len[nxt] >= self.superstate_size + 1)
            & (self._steps[nxt] == g + 1)
            & ~self._final[g % self.capacity]
            & ~self._final[nxt]
        )
        return g[ok]

    def _bank_rebuild(self) -> None:
        """Empty the bank and offer every stored window in order (a projector was just attached)."""
        self._bank_rows.fill_(-1)
        self._bank_density.zero_()
        self._bank_index.fill_(-1)
        self._bank_count = 0
        self._bank_offer_steps(self._bank_candidates(self._oldest, self._clock - 2))

    def _bank_purge_final(self) -> None:
        """Drop keys whose last row or next row is a dropped final row of a finished episode."""
        count = self._bank_count
        if count == 0:
            return
        rows = self._bank_rows[:count]
        bad = self._final[rows % self.capacity] | self._final[(rows + 1) % self.capacity]
        if not bool(bad.any()):
            return
        keep_rows = rows[~bad].clone()
        keep_emb = self._bank_emb[:count][~bad].clone()
        kept = int(keep_rows.numel())
        self._bank_index[rows % self.capacity] = -1
        self._bank_rows[:count] = -1
        self._bank_rows[:kept] = keep_rows
        self._bank_emb[:kept] = keep_emb
        self._bank_count = kept
        self._bank_index[keep_rows % self.capacity] = torch.arange(kept, device=self.device)
        self._bank_recompute_density()

    def _bank_ingest(self, before: int) -> None:
        """Offer the windows completed by the rows appended since global row ``before``."""
        self._bank_offer_steps(self._bank_candidates(before - 1, self._clock - 2))

    def _ensure_capacity(self, needed: int) -> None:
        """Bank mode: grow the row arrays so ``needed`` rows fit; nothing is ever overwritten."""
        if needed <= self.capacity:
            return
        new = max(int(needed), 2 * self.capacity)
        live = torch.nonzero(self._steps >= 0).squeeze(1)
        steps = self._steps[live]
        slots = steps % new

        def moved(old: torch.Tensor, fill: float) -> torch.Tensor:
            out = torch.full((new, *old.shape[1:]), fill, dtype=old.dtype, device=self.device)
            out[slots] = old[live]
            return out

        self._rows = moved(self._rows, 0.0)
        self._steps = moved(self._steps, -1)
        self._run_len = moved(self._run_len, 0)
        self._final = moved(self._final, False)
        self._density = moved(self._density, float("nan"))
        self.capacity = new
        self._bank_index = torch.full((new,), -1, dtype=torch.long, device=self.device)
        count = self._bank_count
        if count:
            self._bank_index[self._bank_rows[:count] % new] = torch.arange(count, device=self.device)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def add_step(self, state: np.ndarray, action: np.ndarray, episode_id: int) -> None:
        """Append one ``[state, action]`` row to the (open) episode."""
        self.add_batch(
            np.asarray(state, dtype=np.float32).reshape(1, self.state_dim),
            np.asarray(action, dtype=np.float32).reshape(1, self.action_dim),
            episode_id,
        )

    def add_batch(
        self, states: np.ndarray, actions: np.ndarray, episode_id: int, episode_end: bool = False
    ) -> int:
        """Append a block of rows and optionally close the episode.

        Rows for an episode other than the open one start a new episode.
        """
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.state_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}")
        rows = torch.as_tensor(np.concatenate((states, actions), axis=1), device=self.device)
        with self._lock:
            if self.bank:
                self._ensure_capacity(self._size + int(rows.shape[0]))
            # A single write must not hit the same slot twice.
            for chunk in rows.split(self.capacity):
                self._append(chunk, int(episode_id))
            if episode_end:
                self.end_episode(episode_id)
        return int(rows.shape[0])

    def _append(self, rows: torch.Tensor, episode_id: int) -> None:
        n = int(rows.shape[0])
        if n == 0:
            return
        if self._open_episode != episode_id:
            if self._open_episode is not None:
                self._episodes_closed += 1
            self._open_episode = episode_id
            self._open_start = self._clock
            prev_run = 0
        else:
            prev_run = int(self._run_len[(self._clock - 1) % self.capacity])
        steps = torch.arange(self._clock, self._clock + n, device=self.device)
        slots = steps % self.capacity
        self._rows[slots] = rows
        self._steps[slots] = steps
        self._density[slots] = float("nan")
        self._final[slots] = False
        self._run_len[slots] = prev_run + torch.arange(1, n + 1, device=self.device)
        self._clock += n
        self._size = min(self.capacity, self._size + n)
        if self._projector is not None:
            if self.bank:
                self._bank_ingest(self._clock - n)
            else:
                keyed = slots[self._run_len[slots] >= self.superstate_size]
                if keyed.numel():
                    self._emb[keyed] = self._embed(self._key_rows(self._steps[keyed]))

    def end_episode(self, episode_id: int) -> None:
        """Close the open episode; the next row starts a new one."""
        with self._lock:
            if self._open_episode == int(episode_id):
                drop = int(getattr(Settings, "HWM_MEMORY_DROP_FINAL_STEPS", 0) or 0)
                if drop > 0:
                    first = max(self._clock - drop, self._open_start, self._oldest)
                    if first < self._clock:
                        steps = torch.arange(first, self._clock, device=self.device)
                        self._final[steps % self.capacity] = True
                        if self.bank:
                            self._bank_purge_final()
                self._open_episode = None
                self._episodes_closed += 1

    def clear(self) -> None:
        with self._lock:
            self._steps.fill_(-1)
            self._final.zero_()
            self._run_len.zero_()
            self._size = 0
            self._open_episode = None
            self._episodes_closed = 0
            if self.bank:
                self._bank_rows.fill_(-1)
                self._bank_density.zero_()
                self._bank_index.fill_(-1)
                self._bank_count = 0

    def export_rows(self) -> Optional[dict[str, torch.Tensor]]:
        """Stored rows oldest-first, for writing the neighbor bank to disk. None if empty."""
        with self._lock:
            if self._size == 0:
                return None
            order = (torch.arange(self._size, device=self.device) + self._oldest) % self.capacity
            return {
                "rows": self._rows[order].detach().cpu().clone(),
                "run_len": self._run_len[order].detach().cpu().clone(),
            }

    def import_rows(self, payload: dict[str, torch.Tensor], episode_id_base: int = 1_000_000) -> int:
        """Load a bank from ``export_rows`` as closed episodes. No-op if this memory already has rows."""
        if payload is None or self.total_steps > 0:
            return 0
        rows = payload["rows"]
        run = payload["run_len"].reshape(-1)
        starts = [0]
        for index in range(1, int(run.shape[0])):
            if int(run[index]) == 1:
                starts.append(index)
        starts.append(int(run.shape[0]))
        imported = 0
        for episode, (begin, end) in enumerate(zip(starts, starts[1:])):
            block = rows[begin:end]
            self.add_batch(
                block[:, : self.state_dim].numpy(),
                block[:, self.state_dim :].numpy(),
                int(episode_id_base) + episode,
                episode_end=True,
            )
            imported += end - begin
        return imported

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    @property
    def _oldest(self) -> int:
        return self._clock - self._size

    def open_length(self, episode_id: int) -> int:
        if self._open_episode != int(episode_id):
            return 0
        return self._clock - self._open_start

    def _open_rows(self, episode_id: int, start: int, end: int) -> torch.Tensor:
        n = self.open_length(episode_id)
        end = min(int(end), n)
        start = min(int(start), end)
        first = self._open_start + start
        if first < self._oldest:
            raise RuntimeError(
                f"rows {start}..{end} of episode {episode_id} were overwritten; "
                f"raise Settings.HWM_MEMORY_MAX_ROWS (now {self.capacity})"
            )
        idx = torch.arange(first, self._open_start + end, device=self.device) % self.capacity
        return self._rows[idx]

    def slice_open(self, episode_id: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Rows ``[start:end)`` of the open episode as ``(states, actions)`` arrays (client streaming)."""
        with self._lock:
            block = self._open_rows(episode_id, start, end).cpu().numpy()
        return block[:, : self.state_dim], block[:, self.state_dim :]

    def recent(self, episode_id: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Last ``length`` rows of the open episode, left-padded with zeros.

        Returns ``(rows (length, row_dim), mask (length,))`` where ``mask`` is True
        for real rows. ``rows[:, :state_dim]`` are states, ``rows[:, state_dim:]`` actions.
        """
        length = int(length)
        rows = torch.zeros((length, self.row_dim), dtype=torch.float32, device=self.device)
        mask = torch.zeros((length,), dtype=torch.bool, device=self.device)
        with self._lock:
            total = self.open_length(episode_id)
            n = min(length, total, self._size)
            if n > 0:
                rows[-n:] = self._open_rows(episode_id, total - n, total)
                mask[-n:] = True
        return rows, mask

    def sample_transitions(
        self, count: Optional[int] = None, skip_start_rows: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``count`` independent random transitions ``(row t, row t+1)`` from anywhere in memory.

        Every stored transition inside one episode is equally likely: the next row must be
        stored in the same episode, neither row may be a dropped final row, and the step
        must not be among the first ``skip_start_rows`` rows of its episode. Drawn without
        replacement when enough exist. Returns ``(current (count, row_dim),
        next (count, row_dim), steps (count,))``, ``steps`` the global insertion indices of
        the current rows. Raises if there is no such transition.
        """
        count = int(Settings.HWM_LLD_BATCH_SIZE if count is None else count)
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        with self._lock:
            nxt = (self._steps + 1) % self.capacity
            ok = (
                (self._steps >= self._oldest)
                & (self._steps + 1 < self._clock)
                & (self._run_len[nxt] == self._run_len + 1)
                & (self._run_len - 1 >= int(skip_start_rows))
                & ~self._final
                & ~self._final[nxt]
            )
            slots = torch.nonzero(ok).squeeze(1)
            if slots.numel() == 0:
                raise RuntimeError("no stored transition is available for an LLD batch")
            if slots.numel() >= count:
                pick = slots[torch.randperm(int(slots.numel()), device=self.device)[:count]]
            else:
                pick = slots[torch.randint(int(slots.numel()), (count,), device=self.device)]
            return self._rows[pick].clone(), self._rows[(pick + 1) % self.capacity].clone(), self._steps[pick].clone()

    def preceding(
        self, steps: np.ndarray | torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``length`` rows before each of ``steps`` in the same episode, oldest first.

        ``steps`` are ``(B,)`` global insertion indices of stored rows. Returns
        ``(rows (B, length, row_dim), mask (B, length))``. Positions before the episode
        start (or before the oldest stored row) are zero with ``mask`` False, so a step
        sees exactly the history ``recent`` gives inference at that point.
        """
        length = int(length)
        with self._lock:
            steps = torch.as_tensor(steps).to(device=self.device, dtype=torch.long).reshape(-1)
            back = torch.arange(length, 0, -1, device=self.device)
            prev = steps[:, None] - back
            run = self._run_len[steps % self.capacity]
            mask = (back[None, :] < run[:, None]) & (prev >= self._oldest)
            rows = self._rows[prev % self.capacity] * mask[..., None]
        return rows, mask

    def episode_positions(self, steps: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Index of each stored row within its episode, 0 at the episode start. ``steps`` ``(B,)``."""
        with self._lock:
            steps = torch.as_tensor(steps).to(device=self.device, dtype=torch.long).reshape(-1)
            return self._run_len[steps % self.capacity] - 1

    @property
    def num_episodes(self) -> int:
        """Episodes closed since the last ``clear``."""
        return self._episodes_closed

    @property
    def next_step(self) -> int:
        """Global insertion index the next stored row will get."""
        return self._clock

    @property
    def total_steps(self) -> int:
        """Rows currently stored."""
        return self._size

    def get_similars(
        self,
        query: np.ndarray | torch.Tensor,
        num_similars: int,
        before: Optional[np.ndarray | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest stored superstates in projector space, each with its next row.

        ``query`` is ``(S * row_dim,)`` or ``(B, S * row_dim)``: normalized superstates,
        flattened oldest row first, as built in ``build_LLD_observation_inference``.
        They are embedded with the attached projector and compared with the stored
        embeddings by Euclidean distance. A match ending at row ``g`` returns rows
        ``g - S + 1 .. g + 1``: the matched superstate, then the next step, all from one
        episode.

        ``before`` is ``(B,)`` global insertion indices (global row indices). If given,
        query ``b`` only sees windows whose every row, including the next step, was
        stored strictly before ``before[b]``.

        Windows returned for one query never share a row: their matches are at least
        ``S + 1`` steps apart. Each pick is the nearest window that does not overlap the
        ones already picked. Only the embeddings are scanned, block by block; rows are
        gathered for the picked windows alone.

        Returns ``(windows (B, num_similars, S + 1, row_dim), distances (B, num_similars))``.
        Windows hold the raw stored rows, oldest first. Missing matches are zero rows
        with infinite distance.
        """
        width = self.superstate_size + 1
        count = int(num_similars)
        query = torch.as_tensor(query).detach().to(device=self.device, dtype=torch.float32)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        expected = lld_super_flat_dim()
        if query.shape[-1] != expected:
            raise ValueError(f"query last dim must be {expected}, got {query.shape[-1]}")
        batch = int(query.shape[0])
        windows = torch.zeros(
            (batch, max(count, 0), width, self.row_dim), dtype=torch.float32, device=self.device
        )
        distances = torch.full((batch, max(count, 0)), float("inf"), device=self.device)
        if count <= 0:
            return windows, distances

        cutoff = None
        if before is not None:
            cutoff = torch.as_tensor(before).to(device=self.device, dtype=torch.long).reshape(-1)
            if int(cutoff.shape[0]) != batch:
                raise ValueError(f"before must have {batch} entries, got {int(cutoff.shape[0])}")

        with self._lock:
            if self._projector is None:
                raise RuntimeError("get_similars needs a projector; call attach_projector first")
            slots = torch.nonzero(self._match_mask()).squeeze(1)
            if slots.numel() == 0:
                return windows, distances
            q = self._embed_query(query)

            # Each pick removes at most 2 * width - 1 candidates, so this many always yield `count` picks.
            gap = 1 if self.overlap else width  # picks closer than this many steps are the same or overlapping
            pool = count * (2 * gap - 1)
            cand_d: list[torch.Tensor] = []
            cand_slots: list[torch.Tensor] = []
            for chunk in slots.split(_SEARCH_CHUNK):
                d = torch.cdist(q, self._key_emb(chunk))
                if cutoff is not None:
                    d = d.masked_fill(self._steps[chunk][None, :] + 1 >= cutoff[:, None], float("inf"))
                top_d, top_i = d.topk(min(pool, int(chunk.numel())), dim=1, largest=False)
                cand_d.append(top_d)
                cand_slots.append(chunk[top_i])
            cand_dist = torch.cat(cand_d, dim=1)
            cand_slot = torch.cat(cand_slots, dim=1)
            cand_dist, order = cand_dist.topk(min(pool, int(cand_dist.shape[1])), dim=1, largest=False)
            cand_slot = cand_slot.gather(1, order)
            cand_step = self._steps[cand_slot]

            alive = torch.isfinite(cand_dist)
            picked = torch.zeros((batch, count), dtype=torch.long, device=self.device)
            found = torch.zeros((batch, count), dtype=torch.bool, device=self.device)
            for k in range(count):
                has = alive.any(dim=1)
                if not bool(has.any()):
                    break
                # Candidates are sorted, so the first alive one is the nearest.
                first = alive.float().argmax(dim=1, keepdim=True)
                picked[:, k] = cand_slot.gather(1, first).squeeze(1)
                found[:, k] = has
                alive &= (cand_step - cand_step.gather(1, first)).abs() >= gap

            offset = torch.arange(width, device=self.device) - (self.superstate_size - 1)
            match_steps = self._steps[picked]
            gathered = self._rows[(match_steps[..., None] + offset) % self.capacity]
            windows = gathered * found[..., None, None]
            exact = torch.linalg.norm(self._key_emb(picked) - q[:, None, :], dim=-1)
            distances = torch.where(found, exact, distances)
        return windows, distances

    def get_all_matches(
        self,
        query: np.ndarray | torch.Tensor,
        before: Optional[np.ndarray | torch.Tensor] = None,
        after: Optional[np.ndarray | torch.Tensor] = None,
        return_density: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Every non-overlapping stored superstate that has a next row, for full-bank attention.

        Returns ``(windows (N, S + 1, row_dim), valid (B, N), distances (B, N))``. The
        windows are one bank shared by every query, not copied per query; ``N`` is the
        number of match slots in the memory. ``before`` marks a window invalid for a
        query unless its next row is strictly older than that query, preserving the
        learner's no-future-data rule. ``after`` ``(B,)``, if given, additionally lets a
        window through when its first row is at or after ``after[b]``: rows further in the
        future than that. Windows between ``before`` and ``after`` stay invalid.
        Invalid windows have infinite distance. With ``return_density`` a fourth result
        ``(N,)`` holds the stored kernel density of each window (see ``_fill_density``).

        No nearest-neighbor decision is made here. The LLD receives every returned
        superstate and computes its differentiable attention weights from the live
        query, so changing an action can smoothly move weight between straight and
        turning examples.
        """
        width = self.superstate_size + 1
        query = torch.as_tensor(query).detach().to(device=self.device, dtype=torch.float32)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        expected = lld_super_flat_dim()
        if query.shape[-1] != expected:
            raise ValueError(f"query last dim must be {expected}, got {query.shape[-1]}")
        batch = int(query.shape[0])
        cutoff = None
        if before is not None:
            cutoff = torch.as_tensor(before).to(device=self.device, dtype=torch.long).reshape(-1)
            if int(cutoff.shape[0]) != batch:
                raise ValueError(f"before must have {batch} entries, got {int(cutoff.shape[0])}")

        with self._lock:
            if self._projector is None:
                raise RuntimeError("get_all_matches needs a projector; call attach_projector first")
            slots = torch.nonzero(self._match_mask()).squeeze(1)
            count = int(slots.numel())
            if count == 0:
                empty = (
                    torch.zeros((0, width, self.row_dim), dtype=torch.float32, device=self.device),
                    torch.zeros((batch, 0), dtype=torch.bool, device=self.device),
                    torch.full((batch, 0), float("inf"), device=self.device),
                )
                return empty + (torch.ones(0, device=self.device),) if return_density else empty
            offset = torch.arange(width, device=self.device) - (self.superstate_size - 1)
            windows = self._rows[(self._steps[slots, None] + offset) % self.capacity]
            valid = torch.ones((batch, count), dtype=torch.bool, device=self.device)
            if cutoff is not None:
                older = self._steps[slots][None, :] + 1 < cutoff[:, None]
                if after is not None:
                    restart = torch.as_tensor(after).to(device=self.device, dtype=torch.long).reshape(-1)
                    if int(restart.shape[0]) != batch:
                        raise ValueError(f"after must have {batch} entries, got {int(restart.shape[0])}")
                    older = older | (self._steps[slots][None, :] - (self.superstate_size - 1) >= restart[:, None])
                valid &= older
            q = self._embed_query(query)
            distances = torch.cdist(q, self._key_emb(slots))
            distances = distances.masked_fill(~valid, float("inf"))
            if return_density:
                return windows, valid, distances, self._key_density(slots)
        return windows, valid, distances
