"""Episode memory shared by the HWM planner (client) and learner (server).

Each episode is one tensor of shape ``(T, state_dim + action_dim)`` whose rows are
``[state_t, action_t]``: the raw 10-d car state at control step ``t`` and the
action in network output units (``physical / action_denorm``). Not clipped to
``[-1, 1]``. Nothing else is stored.

The client appends one row per control step and streams row slices to the
learner; the learner appends the received slices. Both sides therefore hold the
same episode tensors (modulo dropped frames).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from utilities.Settings import Settings


class MemoryManager:
    def __init__(self, state_dim: Optional[int] = None, action_dim: Optional[int] = None, device: str = "cpu"):
        self.state_dim = int(Settings.HWM_STATE_DIM if state_dim is None else state_dim)
        self.action_dim = int(Settings.HWM_ACTION_DIM if action_dim is None else action_dim)
        self.row_dim = self.state_dim + self.action_dim
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        # Closed episodes as tensors; open episodes as growing lists of rows.
        self.episodes: dict[int, torch.Tensor] = {}
        self._open: dict[int, list[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def add_step(self, state: np.ndarray, action: np.ndarray, episode_id: int) -> None:
        """Append one ``[state, action]`` row to the (open) episode."""
        row = np.concatenate(
            (
                np.asarray(state, dtype=np.float32).reshape(self.state_dim),
                np.asarray(action, dtype=np.float32).reshape(self.action_dim),
            )
        )
        self._open.setdefault(int(episode_id), []).append(row)

    def add_batch(
        self, states: np.ndarray, actions: np.ndarray, episode_id: int, episode_end: bool = False
    ) -> int:
        """Append a block of rows (server side) and optionally close the episode."""
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.state_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f"states/actions length mismatch: {states.shape[0]} vs {actions.shape[0]}")
        self._open.setdefault(int(episode_id), []).extend(np.concatenate((states, actions), axis=1))
        if episode_end:
            self.end_episode(episode_id)
        return int(states.shape[0])

    def end_episode(self, episode_id: int) -> Optional[torch.Tensor]:
        """Freeze the open episode into a ``(T, row_dim)`` tensor. Returns None if empty."""
        rows = self._open.pop(int(episode_id), None)
        if not rows:
            return None
        tensor = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=self.device)
        self.episodes[int(episode_id)] = tensor
        return tensor

    def clear(self) -> None:
        self.episodes.clear()
        self._open.clear()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def open_length(self, episode_id: int) -> int:
        return len(self._open.get(int(episode_id), ()))

    def slice_open(self, episode_id: int, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Rows ``[start:end)`` of the open episode as ``(states, actions)`` arrays (client streaming)."""
        rows = self._open.get(int(episode_id), [])[start:end]
        if not rows:
            empty = np.zeros((0, self.row_dim), dtype=np.float32)
            return empty[:, : self.state_dim], empty[:, self.state_dim :]
        block = np.stack(rows, axis=0)
        return block[:, : self.state_dim], block[:, self.state_dim :]

    def episode_tensor(self, episode_id: int) -> torch.Tensor:
        """Full ``(T, row_dim)`` tensor of a closed or open episode (empty if unknown)."""
        episode_id = int(episode_id)
        if episode_id in self.episodes:
            return self.episodes[episode_id]
        rows = self._open.get(episode_id)
        if not rows:
            return torch.zeros((0, self.row_dim), dtype=torch.float32, device=self.device)
        return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=self.device)

    def recent(self, episode_id: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Last ``length`` rows of an episode, left-padded with zeros.

        Returns ``(rows (length, row_dim), mask (length,))`` where ``mask`` is True
        for real rows. ``rows[:, :state_dim]`` are states, ``rows[:, state_dim:]`` actions.
        """
        full = self.episode_tensor(episode_id)
        n = min(int(length), int(full.shape[0]))
        rows = torch.zeros((int(length), self.row_dim), dtype=torch.float32, device=self.device)
        mask = torch.zeros((int(length),), dtype=torch.bool, device=self.device)
        if n > 0:
            rows[-n:] = full[-n:]
            mask[-n:] = True
        return rows, mask

    def closed_episode_ids(self) -> list[int]:
        return sorted(self.episodes.keys())

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_steps(self) -> int:
        closed = sum(int(t.shape[0]) for t in self.episodes.values())
        return closed + sum(len(rows) for rows in self._open.values())
    
    def get_similars(self, state: np.ndarray, num_similars: int, superstate_size: int) -> torch.Tensor:
        """Nearest stored rows, each followed by the next rows in that episode.

        ``state`` is ``(row_dim,)`` or ``(B, row_dim)``: state and action concatenated.
        A match at time ``t`` contributes rows ``t .. t + superstate_size - 1`` from
        the same episode; starts that would run past the episode end are skipped.

        Returns ``(B, num_similars, superstate_size, row_dim)``. ``out[b, k, i]`` is
        row ``i`` inside that superstate, ``[state, action]``. Missing matches are
        zero. Very inefficient for now.
        """
        query = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.shape[-1] != self.row_dim:
            raise ValueError(f"query last dim must be {self.row_dim}, got {query.shape[-1]}")
        batch = int(query.shape[0])
        width = int(superstate_size)
        count = int(num_similars)
        out = torch.zeros(
            (batch, count, width, self.row_dim), dtype=torch.float32, device=self.device
        )
        if width <= 0 or count <= 0:
            return out

        windows: list[torch.Tensor] = []
        for episode_id in list(self.closed_episode_ids()) + list(self._open.keys()):
            episode_rows = self.episode_tensor(episode_id)
            length = int(episode_rows.shape[0])
            if length < width:
                continue
            offset = torch.arange(width, device=episode_rows.device)
            starts = torch.arange(length - width + 1, device=episode_rows.device)
            windows.append(episode_rows[starts[:, None] + offset])

        if not windows:
            return out

        window_bank = torch.cat(windows, dim=0)
        distances = torch.linalg.norm(window_bank[:, 0].unsqueeze(0) - query.unsqueeze(1), dim=-1)
        take = min(count, int(window_bank.shape[0]))
        nearest = torch.topk(distances, k=take, dim=-1, largest=False).indices
        out[:, :take] = window_bank[nearest]
        return out
