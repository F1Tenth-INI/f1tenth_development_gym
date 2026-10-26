"""Episode memory shared by the HWM planner (client) and learner (server).

Rows are ``[state_t, action_t]``: the raw 10-d car state at control step ``t`` and the
action in network output units (``physical / action_denorm``). Not clipped to
``[-1, 1]``. Nothing else is stored.

Row ``g`` (its global insertion index) sits in slot ``g % capacity``; ``capacity`` starts at
``Settings.HWM_MEMORY_MAX_ROWS`` and the row arrays double when more rows arrive. Rows arrive in
time order from a single actor, so one episode occupies consecutive indices. A row is kept only
while a key in the bank still reads it (the key's ``S`` rows and the next row). Rows whose key was
rejected or replaced are dropped once the episode closes, and they are not written to disk.

The LLD's keys (superstates) are a separate, limited array of ``HWM_MEMORY_BANK_SIZE`` indices into
the rows, each ending a window of ``S = HWM_SUPER_STATE_SIZE`` rows plus the step it predicts (one
episode; windows may overlap). Every row that completes a window is offered as a key. When the bank
is full an offered key replaces the key with the highest n_soft (its kernel mass among the other
keys) only if its own n_soft is lower, so dense regions are thinned first. Embeddings come from the
LLD ``attention_norm_projector``. They are computed without gradients when a key arrives, and
``refresh_embeddings`` re-embeds all keys and recomputes every n_soft after the projector weights change.

The client appends one row per control step and streams row slices to the learner;
the learner appends the received slices. Both sides therefore hold the same rows
(modulo dropped frames), and each keeps its own key bank.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch
from torch import nn

from TrainingLite.hwm.shared.observation import _normalize_rows, lld_delta, lld_rows_for_super, lld_super_flat_dim
from utilities.Settings import Settings

# Keys embedded per projector call.
_EMBED_CHUNK = 65536


class MemoryManager:
    def __init__(
        self,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        device: str | torch.device = "cpu",
        capacity: Optional[int] = None,
        superstate_size: Optional[int] = None,
        keep_all_rows: bool = False,
    ):
        self.state_dim = int(Settings.HWM_STATE_DIM if state_dim is None else state_dim)
        self.action_dim = int(Settings.HWM_ACTION_DIM if action_dim is None else action_dim)
        self.row_dim = self.state_dim + self.action_dim
        # Initial row allocation. Closed-episode rows that no kept key uses are dropped.
        self.capacity = int(Settings.HWM_MEMORY_MAX_ROWS if capacity is None else capacity)
        self.superstate_size = int(
            Settings.HWM_SUPER_STATE_SIZE if superstate_size is None else superstate_size
        )
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.superstate_size < 1:
            raise ValueError(f"superstate_size must be at least 1, got {self.superstate_size}")
        self.device = torch.device(device)
        # True: no row is ever dropped for lack of a key (the learner trains on every incoming transition).
        # Only the key bank, and the saved file's bank_ends, decide which windows are retrievable neighbors.
        self.keep_all_rows = bool(keep_all_rows)

        # Ingest (event loop) and training (worker thread) touch the buffer concurrently on the learner.
        self._lock = threading.RLock()
        self._rows = torch.zeros((self.capacity, self.row_dim), dtype=torch.float32, device=self.device)
        # Global insertion index held by each slot, -1 while empty.
        self._steps = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        # Number of consecutive rows of the same episode ending at each row (1 at the episode start).
        self._run_len = torch.zeros((self.capacity,), dtype=torch.long, device=self.device)
        # True for the last HWM_MEMORY_DROP_FINAL_STEPS rows of a finished episode. Such rows
        # are never a key's next row and never inside a training batch.
        self._final = torch.zeros((self.capacity,), dtype=torch.bool, device=self.device)
        self._projector: Optional[nn.Module] = None
        self._clock = 0
        self._size = 0
        self._open_episode: Optional[int] = None
        self._open_start = 0
        self._episodes_closed = 0

        # The key bank: rows are never evicted, the keys are a limited array of row indices.
        self.bank_size = int(Settings.HWM_MEMORY_BANK_SIZE)
        self.bank_exclude_overlap = bool(Settings.HWM_MEMORY_BANK_EXCLUDE_OVERLAP)
        if self.bank_size <= 0:
            raise ValueError(f"HWM_MEMORY_BANK_SIZE must be positive, got {self.bank_size}")
        # Global index of the last row of each key (-1 empty), its embedding and its n_soft.
        self._bank_rows = torch.full((self.bank_size,), -1, dtype=torch.long, device=self.device)
        self._bank_emb: Optional[torch.Tensor] = None
        self._bank_density = torch.zeros((self.bank_size,), dtype=torch.float32, device=self.device)
        # Pose-free normalized superstate and the delta leaving it, written when the key enters.
        # A rollout reads [:count], which is a view: the rows are not copied again.
        self._bank_super: Optional[torch.Tensor] = None
        self._bank_delta: Optional[torch.Tensor] = None
        self._bank_count = 0
        # Row slot -> position in the bank, -1 if the row ends no key.
        self._bank_index = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def attach_projector(self, projector: nn.Module) -> None:
        """Use ``projector`` (the LLD's ``attention_norm_projector``) to embed keys and queries."""
        with self._lock:
            self._projector = projector
            probe = self._embed(torch.zeros((1, lld_super_flat_dim()), device=self.device))
            self._bank_emb = torch.zeros(
                (self.bank_size, int(probe.shape[-1])), dtype=torch.float32, device=self.device
            )
            self._bank_rebuild()

    def refresh_embeddings(self) -> None:
        """Re-embed every key with the current projector weights and recompute every n_soft."""
        with self._lock:
            if self._projector is None:
                return
            count = self._bank_count
            for chunk in torch.arange(count, device=self.device).split(_EMBED_CHUNK):
                self._bank_emb[chunk] = self._embed(self._key_rows(self._bank_rows[chunk]))
            self._bank_recompute_density()

    def embed(self, superstates: torch.Tensor) -> torch.Tensor:
        """Embedding ``(B, emb)`` of normalized flat superstates ``(B, S * row_dim)``."""
        with self._lock:
            if self._projector is None:
                raise RuntimeError("embed needs a projector; call attach_projector first")
            return self._run_projector(self._projector, superstates.to(self.device))

    @torch.no_grad()
    def _embed(self, superstates: torch.Tensor) -> torch.Tensor:
        return self._run_projector(self._projector, superstates)

    def _run_projector(self, projector: nn.Module, superstates: torch.Tensor) -> torch.Tensor:
        param = next(projector.parameters())
        out = projector(superstates.to(device=param.device, dtype=param.dtype))
        return out.to(device=self.device, dtype=torch.float32)

    def _key_rows(self, last_steps: torch.Tensor) -> torch.Tensor:
        """Normalized, flattened superstates ending at ``last_steps``, oldest row first."""
        return self._key_inputs(last_steps)[0]

    def _key_inputs(self, last_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(super (K, flat), delta (K, d))`` of the windows ending at ``last_steps``.

        ``super`` is the pose-free normalized window, oldest row first, without the next row.
        ``delta`` is the LLD target from that window's last row to the next stored row.
        """
        width = self.superstate_size + 1
        offset = torch.arange(width, device=self.device) - (self.superstate_size - 1)
        windows = self._rows[(last_steps[:, None] + offset) % self.capacity]
        state_dim = self.state_dim
        delta = lld_delta(windows[..., -2, :state_dim], windows[..., -1, :state_dim])
        packed = lld_rows_for_super(_normalize_rows(windows[..., :-1, :], state_dim), state_dim)
        return packed.flatten(start_dim=1), delta

    def _ensure_bank_inputs(self, flat: int, delta_dim: int) -> None:
        if self._bank_super is not None:
            return
        self._bank_super = torch.zeros((self.bank_size, int(flat)), dtype=torch.float32, device=self.device)
        self._bank_delta = torch.zeros((self.bank_size, int(delta_dim)), dtype=torch.float32, device=self.device)

    def lld_bank(self) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Views of the key bank: superstates, deltas, n_soft, ending rows. None if it is empty.

        The four tensors are slices of the bank storage, so a rollout can hold them for every
        step without copying the memory rows.
        """
        count = self._bank_count
        if count == 0 or self._bank_super is None:
            return None
        return (
            self._bank_super[:count],
            self._bank_delta[:count],
            self._bank_density[:count],
            self._bank_rows[:count],
        )

    # ------------------------------------------------------------------
    # Key bank
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

    def _bank_offer(
        self, step: int, emb: torch.Tensor, super_row: torch.Tensor, delta_row: torch.Tensor
    ) -> bool:
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
        self._bank_super[slot] = super_row
        self._bank_delta[slot] = delta_row
        self._bank_density[slot] = n_in
        self._bank_index[int(step) % self.capacity] = slot
        return True

    def _bank_offer_steps(self, steps: torch.Tensor) -> None:
        """Offer the keys ending at the global rows ``steps`` (ascending), in order."""
        for chunk in steps.split(_EMBED_CHUNK):
            supers, deltas = self._key_inputs(chunk)
            self._ensure_bank_inputs(int(supers.shape[-1]), int(deltas.shape[-1]))
            emb = self._embed(supers)
            for step, vector, super_row, delta_row in zip(chunk.tolist(), emb, supers, deltas):
                self._bank_offer(step, vector, super_row, delta_row)

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
        keep_super = self._bank_super[:count][~bad].clone()
        keep_delta = self._bank_delta[:count][~bad].clone()
        kept = int(keep_rows.numel())
        self._bank_index[rows % self.capacity] = -1
        self._bank_rows[:count] = -1
        self._bank_rows[:kept] = keep_rows
        self._bank_emb[:kept] = keep_emb
        self._bank_super[:kept] = keep_super
        self._bank_delta[:kept] = keep_delta
        self._bank_count = kept
        self._bank_index[keep_rows % self.capacity] = torch.arange(kept, device=self.device)
        self._bank_recompute_density()

    def _bank_ingest(self, before: int) -> None:
        """Offer the windows completed by the rows appended since global row ``before``."""
        self._bank_offer_steps(self._bank_candidates(before - 1, self._clock - 2))

    def _ensure_capacity(self, needed: int) -> None:
        """Grow the row arrays so ``needed`` rows fit; nothing is ever overwritten."""
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
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_id: int,
        episode_end: bool = False,
        offer: bool = True,
    ) -> int:
        """Append a block of rows and optionally close the episode.

        Rows for an episode other than the open one start a new episode.
        ``offer`` is false while restoring a saved bank, so closing the episode does not
        drop the rows before those keys are installed.
        """
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.state_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}")
        rows = torch.as_tensor(np.concatenate((states, actions), axis=1), device=self.device)
        with self._lock:
            self._ensure_capacity(self._size + int(rows.shape[0]))
            # A single write must not hit the same slot twice.
            for chunk in rows.split(self.capacity):
                self._append(chunk, int(episode_id), offer=offer)
            if episode_end:
                self.end_episode(episode_id, compact=offer)
        return int(rows.shape[0])

    def _append(self, rows: torch.Tensor, episode_id: int, offer: bool = True) -> None:
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
        self._final[slots] = False
        self._run_len[slots] = prev_run + torch.arange(1, n + 1, device=self.device)
        self._clock += n
        self._size = min(self.capacity, self._size + n)
        if offer and self._projector is not None:
            self._bank_ingest(self._clock - n)

    def end_episode(self, episode_id: int, compact: bool = True) -> None:
        """Close the open episode; the next row starts a new one.

        With ``compact``, rows that no remaining key reads are dropped.
        """
        with self._lock:
            if self._open_episode == int(episode_id):
                drop = int(getattr(Settings, "HWM_MEMORY_DROP_FINAL_STEPS", 0) or 0)
                if drop > 0:
                    first = max(self._clock - drop, self._open_start, self._oldest)
                    if first < self._clock:
                        steps = torch.arange(first, self._clock, device=self.device)
                        self._final[steps % self.capacity] = True
                        self._bank_purge_final()
                self._open_episode = None
                self._episodes_closed += 1
                if compact:
                    self._compact_kept_rows()

    def clear(self) -> None:
        with self._lock:
            self._steps.fill_(-1)
            self._final.zero_()
            self._run_len.zero_()
            self._size = 0
            self._open_episode = None
            self._episodes_closed = 0
            self._bank_rows.fill_(-1)
            self._bank_density.zero_()
            self._bank_index.fill_(-1)
            self._bank_count = 0

    def _bank_covered_steps(self) -> torch.Tensor:
        """Global steps a current key still reads: its ``S`` rows and the next row."""
        count = self._bank_count
        if count == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        offset = torch.arange(1 - self.superstate_size, 2, device=self.device)
        covered = (self._bank_rows[:count][:, None] + offset).reshape(-1)
        known = (covered >= self._oldest) & (covered < self._clock)
        covered = covered[known]
        if covered.numel() == 0:
            return covered
        stored = self._steps[covered % self.capacity] == covered
        return torch.unique(covered[stored], sorted=True)

    def _kept_steps(self, include_open: bool) -> torch.Tensor:
        """Steps to keep. The open episode stays whole so streaming offsets still match."""
        parts = [self._bank_covered_steps()]
        if include_open and self._open_episode is not None and self._clock > self._open_start:
            parts.append(torch.arange(self._open_start, self._clock, device=self.device))
        parts = [part for part in parts if part.numel()]
        if not parts:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.unique(torch.cat(parts), sorted=True)

    def _pack_steps(self, kept: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rows, run_len)`` of ``kept`` global steps, one run per contiguous same-episode span."""
        source = kept % self.capacity
        rows = self._rows[source]
        old_run = self._run_len[source]
        count = int(kept.shape[0])
        breaks = torch.ones(count, dtype=torch.bool, device=self.device)
        if count > 1:
            breaks[1:] = (kept[1:] != kept[:-1] + 1) | (old_run[1:] != old_run[:-1] + 1)
        index = torch.arange(count, device=self.device)
        start = torch.cummax(torch.where(breaks, index, torch.zeros(count, dtype=torch.long, device=self.device)), 0).values
        return rows, index - start + 1

    def _compact_kept_rows(self) -> None:
        """Drop closed-episode rows that no key reads, and point the bank at what remains.

        Caller holds ``self._lock``. Key embeddings and n_soft stay on their slots: only the
        row index of each key changes, and a kept window stays consecutive.
        """
        if self.keep_all_rows:
            return
        kept = self._kept_steps(include_open=True)
        if (
            kept.numel() == self._size
            and self._size > 0
            and int(kept[0]) == self._oldest
            and int(kept[-1]) == self._clock - 1
        ):
            return
        count = self._bank_count
        open_episode = self._open_episode
        open_start = self._open_start
        if kept.numel() == 0:
            capacity = max(int(Settings.HWM_MEMORY_MAX_ROWS), 1)
            self._rows = torch.zeros((capacity, self.row_dim), dtype=torch.float32, device=self.device)
            self._steps = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
            self._run_len = torch.zeros((capacity,), dtype=torch.long, device=self.device)
            self._final = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
            self.capacity = capacity
            self._clock = 0
            self._size = 0
            self._open_episode = None
            self._open_start = 0
            self._episodes_closed = 0
            self._bank_rows.fill_(-1)
            self._bank_density.zero_()
            self._bank_count = 0
            self._bank_index = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
            return
        packed, run_len = self._pack_steps(kept)
        stored = int(kept.shape[0])
        capacity = max(stored, int(Settings.HWM_MEMORY_MAX_ROWS))
        self._rows = torch.zeros((capacity, self.row_dim), dtype=torch.float32, device=self.device)
        self._steps = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        self._run_len = torch.zeros((capacity,), dtype=torch.long, device=self.device)
        self._final = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        self._rows[:stored] = packed
        self._steps[:stored] = torch.arange(stored, device=self.device)
        self._run_len[:stored] = run_len
        self.capacity = capacity
        self._clock = stored
        self._size = stored
        if count:
            self._bank_rows[:count] = torch.searchsorted(kept, self._bank_rows[:count].clone())
            self._bank_rows[count:] = -1
        if open_episode is not None and open_start < int(kept[-1]) + 1:
            self._open_episode = open_episode
            self._open_start = int(torch.searchsorted(kept, torch.tensor(open_start, device=self.device)))
        else:
            self._open_episode = None
            self._open_start = stored
        closed = self._open_start if self._open_episode is not None else stored
        breaks = torch.ones(stored, dtype=torch.bool, device=self.device)
        if stored > 1:
            breaks[1:] = self._run_len[1:stored] == 1
        self._episodes_closed = int(breaks[:closed].sum()) if closed else 0
        self._bank_index = torch.full((capacity,), -1, dtype=torch.long, device=self.device)
        if count:
            self._bank_index[self._bank_rows[:count]] = torch.arange(count, device=self.device)

    def export_rows(self) -> Optional[dict[str, torch.Tensor]]:
        """Rows a kept key still reads, oldest first (every stored row except the dropped final ones with
        ``keep_all_rows``). None if the bank is empty.

        ``bank_ends`` are the positions of those keys in ``rows`` (the last row of each
        superstate). A reload offers those keys only, so a discarded window that sits
        inside a neighbour's rows does not come back.
        """
        with self._lock:
            if self.keep_all_rows:
                steps = torch.arange(self._oldest, self._clock, device=self.device)
                slots = steps % self.capacity
                kept = steps[(self._steps[slots] == steps) & ~self._final[slots]]
            else:
                kept = self._bank_covered_steps()
            if kept.numel() == 0 or self._bank_count == 0:
                return None
            rows, run_len = self._pack_steps(kept)
            return {
                "rows": rows.detach().cpu().clone(),
                "run_len": run_len.detach().cpu().clone(),
                "bank_ends": torch.searchsorted(kept.cpu(), self._bank_rows[: self._bank_count].detach().cpu()),
            }

    def import_rows(self, payload: dict[str, torch.Tensor], episode_id_base: int = 1_000_000) -> int:
        """Load ``export_rows``. A file without ``bank_ends`` is admitted, then discarded rows are dropped.

        No-op if this memory already has rows. Returns the number of rows still stored.
        """
        if payload is None or self.total_steps > 0:
            return 0
        rows = payload["rows"]
        run = payload["run_len"].reshape(-1)
        bank_ends = payload.get("bank_ends")
        starts = [0]
        for index in range(1, int(run.shape[0])):
            if int(run[index]) == 1:
                starts.append(index)
        starts.append(int(run.shape[0]))
        restore = bank_ends is not None
        for episode, (begin, end) in enumerate(zip(starts, starts[1:])):
            block = rows[begin:end]
            self.add_batch(
                block[:, : self.state_dim].numpy(),
                block[:, self.state_dim :].numpy(),
                int(episode_id_base) + episode,
                episode_end=True,
                offer=not restore,
            )
        if restore:
            ends = torch.as_tensor(bank_ends, dtype=torch.long, device=self.device).reshape(-1)
            with self._lock:
                if int(ends.numel()):
                    self._bank_offer_steps(ends)
                self._compact_kept_rows()
        return self._size

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

    def sample_batch(self, batch_size: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One run of ``batch_size`` consecutive ``[state, action]`` rows from one episode.

        Returns ``(rows (batch_size, row_dim), steps (batch_size,))``. ``steps`` are the
        global insertion indices, for ``get_all_matches(before=...)``. Every run that lies
        inside one stored episode is equally likely. Raises if there is none.
        """
        length = int(Settings.HWM_LLD_BATCH_SIZE if batch_size is None else batch_size)
        if length <= 0:
            raise ValueError(f"batch_size must be positive, got {length}")
        with self._lock:
            ok = (self._steps - (length - 1) >= self._oldest) & (self._run_len >= length)
            ends = torch.nonzero(ok).squeeze(1)
            if ends.numel() == 0:
                raise RuntimeError(
                    f"no episode has {length} consecutive steps; cannot sample a full LLD batch"
                )
            end = int(self._steps[ends[int(torch.randint(int(ends.numel()), ()))]])
            steps = torch.arange(end - length + 1, end + 1, device=self.device)
            return self._rows[steps % self.capacity].clone(), steps

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

    def get_all_matches(
        self,
        query: np.ndarray | torch.Tensor,
        before: Optional[np.ndarray | torch.Tensor] = None,
        after: Optional[np.ndarray | torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every key in the bank with its next row, for full-bank attention.

        Returns ``(windows (N, S + 1, row_dim), valid (B, N), distances (B, N), density (N,))``,
        keys in the order their rows were stored. The windows are one bank shared by every query, not copied per query; ``N`` is the
        number of keys in the bank. ``before`` marks a window invalid for a query unless
        its next row is strictly older than that query, preserving the learner's
        no-future-data rule. ``after`` ``(B,)``, if given, additionally lets a window
        through when its first row is at or after ``after[b]``: rows further in the
        future than that. Windows between ``before`` and ``after`` stay invalid. Invalid
        windows have infinite distance. ``density`` is the stored n_soft of each key.

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
            count = self._bank_count
            if count == 0:
                return (
                    torch.zeros((0, width, self.row_dim), dtype=torch.float32, device=self.device),
                    torch.zeros((batch, 0), dtype=torch.bool, device=self.device),
                    torch.full((batch, 0), float("inf"), device=self.device),
                    torch.ones(0, device=self.device),
                )
            # Keys in the order their rows were stored, whatever the bank positions.
            order = torch.argsort(self._bank_rows[:count])
            last = self._bank_rows[:count][order]
            offset = torch.arange(width, device=self.device) - (self.superstate_size - 1)
            windows = self._rows[(last[:, None] + offset) % self.capacity]
            valid = torch.ones((batch, count), dtype=torch.bool, device=self.device)
            if cutoff is not None:
                older = last[None, :] + 1 < cutoff[:, None]
                if after is not None:
                    restart = torch.as_tensor(after).to(device=self.device, dtype=torch.long).reshape(-1)
                    if int(restart.shape[0]) != batch:
                        raise ValueError(f"after must have {batch} entries, got {int(restart.shape[0])}")
                    older = older | (last[None, :] - (self.superstate_size - 1) >= restart[:, None])
                valid &= older
            q = self._run_projector(self._projector, query)
            distances = torch.cdist(q, self._bank_emb[:count][order])
            distances = distances.masked_fill(~valid, float("inf"))
            return windows, valid, distances, self._bank_density[:count][order].clone()
