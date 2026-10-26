"""LLD mean rollout of a whole horizon as one CUDA graph.

The eager rollout is a chain of tiny ops per step and is bound by launch overhead, not arithmetic.
Here the 25-step loop (key encoding included) is captured once per batch size on a GPU copy of the
LLD and replayed with one launch. Inputs are copied into fixed buffers. The key bank is padded to
its full size with invalid keys: their logit is the dtype minimum, so their kernel is exactly zero
and they change nothing but the summation order.
"""

from __future__ import annotations

import torch

from TrainingLite.hwm.models.networks import build_models
from TrainingLite.hwm.shared.observation import _normalize_states, lld_rows_for_super
from utilities.Settings import Settings


class LLDGraphRollout:
    def __init__(self, lld_cpu: torch.nn.Module, bank_size: int, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.bank_size = int(bank_size)
        self.state_dim = int(Settings.HWM_STATE_DIM)
        self.lld = build_models()["LLD"].to(self.device).eval()
        for parameter in self.lld.parameters():
            parameter.requires_grad_(False)
        self.sync_weights(lld_cpu)
        self._graphs: dict[tuple[int, int], dict] = {}

    def sync_weights(self, lld_cpu: torch.nn.Module) -> None:
        """Copy the weights in place, so captured graphs keep pointing at them."""
        with torch.no_grad():
            self.lld.load_state_dict(lld_cpu.state_dict())

    def _rollout(self, s, actions, history, supers, deltas, density, valid):
        lld = self.lld
        key_emb = lld.attention_norm_projector(supers)
        tokens = deltas - lld._correct(supers)
        states, n_softs, n_soft_norms = [s], [], []
        for k in range(int(actions.shape[1])):
            current = lld_rows_for_super(torch.cat((_normalize_states(s), actions[:, k]), dim=-1), self.state_dim)
            query = torch.cat((history.flatten(start_dim=1), current), dim=-1)
            delta, n_soft, n_soft_norm = lld._mean_delta_core(query, key_emb, tokens, density, valid)
            s = lld.reconstruct_next_state(s, delta)
            history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
            states.append(s)
            n_softs.append(n_soft)
            n_soft_norms.append(n_soft_norm)
        return torch.stack(states, dim=1), torch.stack(n_softs, dim=1), torch.stack(n_soft_norms, dim=1)

    def _capture(self, batch: int, horizon: int, history_shape: tuple, super_dim: int, delta_dim: int) -> dict:
        dev = self.device
        buffers = {
            "s": torch.zeros(batch, self.state_dim, device=dev),
            "a": torch.zeros(batch, horizon, 2, device=dev),
            "h": torch.zeros(batch, *history_shape, device=dev),
            "supers": torch.zeros(self.bank_size, super_dim, device=dev),
            "deltas": torch.zeros(self.bank_size, delta_dim, device=dev),
            "density": torch.ones(self.bank_size, device=dev),
            "valid": torch.zeros(self.bank_size, dtype=torch.bool, device=dev),
        }
        args = tuple(buffers.values())
        with torch.no_grad():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):  # also builds the cached constants before capture
                    self._rollout(*args)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._rollout(*args)
        return {"graph": graph, "buffers": buffers, "outputs": outputs}

    @torch.no_grad()
    def run(self, s, actions, history, supers, deltas, density, valid):
        """Same as the eager loop. ``supers`` etc. are the bank views (count <= bank_size), ``valid`` ``(count,)``."""
        count = int(supers.shape[0])
        if count > self.bank_size:
            raise ValueError(f"bank has {count} keys, graph was sized for {self.bank_size}")
        batch, horizon = int(actions.shape[0]), int(actions.shape[1])
        key = (batch, horizon)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._graphs[key] = self._capture(
                batch, horizon, tuple(history.shape[1:]), int(supers.shape[1]), int(deltas.shape[1])
            )
        b = entry["buffers"]
        b["s"].copy_(s)
        b["a"].copy_(actions)
        b["h"].copy_(history)
        b["supers"][:count].copy_(supers)
        b["deltas"][:count].copy_(deltas)
        b["density"].fill_(1.0)
        b["density"][:count].copy_(density)
        b["valid"].zero_()
        b["valid"][:count].copy_(valid)
        entry["graph"].replay()
        states, n_soft, n_soft_norm = entry["outputs"]
        return states.cpu(), n_soft.cpu(), n_soft_norm.cpu()
