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

from TrainingLite.hwm.shared.observation import _normalize_rows
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
    ):
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
        self._projector: Optional[nn.Module] = None
        self._emb: Optional[torch.Tensor] = None
        self._clock = 0
        self._size = 0
        self._open_episode: Optional[int] = None
        self._open_start = 0
        self._episodes_closed = 0

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def attach_projector(self, projector: nn.Module) -> None:
        """Use ``projector`` (normalized flat superstate -> embedding) and embed all stored rows."""
        with self._lock:
            self._projector = projector
            probe = self._embed(
                torch.zeros((1, self.superstate_size * self.row_dim), device=self.device)
            )
            self._emb = torch.zeros(
                (self.capacity, int(probe.shape[-1])), dtype=torch.float32, device=self.device
            )
            self.refresh_embeddings()

    def refresh_embeddings(self) -> None:
        """Re-embed every stored superstate with the current projector weights."""
        with self._lock:
            if self._projector is None:
                return
            slots = torch.nonzero(self._keyed_mask()).squeeze(1)
            for chunk in slots.split(_EMBED_CHUNK):
                self._emb[chunk] = self._embed(self._key_rows(self._steps[chunk]))

    @torch.no_grad()
    def _embed(self, superstates: torch.Tensor) -> torch.Tensor:
        param = next(self._projector.parameters())
        out = self._projector(superstates.to(device=param.device, dtype=param.dtype))
        return out.to(device=self.device, dtype=torch.float32)

    def _key_rows(self, last_steps: torch.Tensor) -> torch.Tensor:
        """Normalized, flattened superstates ending at ``last_steps``, oldest row first."""
        offset = torch.arange(self.superstate_size, device=self.device) - (self.superstate_size - 1)
        rows = self._rows[(last_steps[:, None] + offset) % self.capacity]
        return _normalize_rows(rows, self.state_dim).flatten(start_dim=1)

    def _keyed_mask(self) -> torch.Tensor:
        """Slots whose superstate lies in one episode and is fully stored."""
        first = self._steps - (self.superstate_size - 1)
        return (first >= self._oldest) & (self._run_len >= self.superstate_size)

    def _match_mask(self) -> torch.Tensor:
        """Slots that end a superstate whose next row is stored in the same episode."""
        g = self._steps
        nxt = (g + 1) % self.capacity
        return (
            (g - (self.superstate_size - 1) >= self._oldest)
            & (g + 1 < self._clock)
            & (self._run_len[nxt] >= self.superstate_size + 1)
        )

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
        self._run_len[slots] = prev_run + torch.arange(1, n + 1, device=self.device)
        self._clock += n
        self._size = min(self.capacity, self._size + n)
        if self._projector is not None:
            keyed = slots[self._run_len[slots] >= self.superstate_size]
            if keyed.numel():
                self._emb[keyed] = self._embed(self._key_rows(self._steps[keyed]))

    def end_episode(self, episode_id: int) -> None:
        """Close the open episode; the next row starts a new one."""
        with self._lock:
            if self._open_episode == int(episode_id):
                self._open_episode = None
                self._episodes_closed += 1

    def clear(self) -> None:
        with self._lock:
            self._steps.fill_(-1)
            self._run_len.zero_()
            self._size = 0
            self._open_episode = None
            self._episodes_closed = 0

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

    def sample_batch(self, batch_size: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """One run of ``batch_size`` consecutive ``[state, action]`` rows from one episode.

        Returns ``(rows (batch_size, row_dim), steps (batch_size,))``. ``steps`` are the
        global insertion indices, for ``get_similars(before=...)``. Every run that lies
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

        ``before`` is ``(B,)`` global insertion indices (see ``sample_batch``). If given,
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
        if query.shape[-1] != self.superstate_size * self.row_dim:
            raise ValueError(
                f"query last dim must be {self.superstate_size * self.row_dim}, got {query.shape[-1]}"
            )
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
            q = self._embed(query)

            # Each pick removes at most 2 * width - 1 candidates, so this many always yield `count` picks.
            pool = count * (2 * width - 1)
            cand_d: list[torch.Tensor] = []
            cand_slots: list[torch.Tensor] = []
            for chunk in slots.split(_SEARCH_CHUNK):
                d = torch.cdist(q, self._emb[chunk])
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
                alive &= (cand_step - cand_step.gather(1, first)).abs() >= width

            offset = torch.arange(width, device=self.device) - (self.superstate_size - 1)
            match_steps = self._steps[picked]
            gathered = self._rows[(match_steps[..., None] + offset) % self.capacity]
            windows = gathered * found[..., None, None]
            exact = torch.linalg.norm(self._emb[picked] - q[:, None, :], dim=-1)
            distances = torch.where(found, exact, distances)
        return windows, distances
