"""Whole-horizon LLD rollout: eager vs compiled vs CUDA graph, end to end (CPU tensors in, CPU tensors out).

    python scripts/bench_lld_graph.py --model hwm_weighted_SE_test_7 [--bank 600]

Also checks every variant against the eager CPU rollout (max abs difference of the state trajectory).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from TrainingLite.hwm.models.networks import build_models  # noqa: E402
from TrainingLite.hwm.paths import resolve_model_dir  # noqa: E402
from TrainingLite.hwm.shared.memory_manager import MemoryManager  # noqa: E402
from TrainingLite.hwm.shared.observation import _normalize_states, lld_rows_for_super  # noqa: E402
from utilities.Settings import Settings  # noqa: E402


def timeit(fn, repeat: int = 60, warm: int = 8) -> float:
    for _ in range(warm):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bank", type=int, default=600)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 50])
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    state_dim = int(Settings.HWM_STATE_DIM)
    width = int(Settings.HWM_SUPER_STATE_SIZE)

    models = build_models().to("cpu").eval()
    models.load(resolve_model_dir(args.model), device="cpu")
    lld = models["LLD"]
    flat = lld.attention_norm_projector[0].in_features
    Settings.HWM_MEMORY_BANK_SIZE = args.bank
    memory = MemoryManager(device="cpu")
    memory.attach_projector(lld.attention_norm_projector)
    memory.import_rows(torch.load(resolve_model_dir(args.model) / "memory.pt", weights_only=True))
    memory.refresh_embeddings()
    supers, deltas, density, _ = memory.lld_bank()
    print(f"bank keys: {supers.shape[0]}")

    def make_rollout(model):
        """Plain whole-horizon function ``(s, actions, history, supers, deltas, density, valid) -> states``."""

        def rollout(s, actions, history, supers, deltas, density, valid):
            key_emb = model.attention_norm_projector(supers)
            tokens = deltas - model._correct(supers)
            states = [s]
            for k in range(actions.shape[1]):
                current = lld_rows_for_super(torch.cat((_normalize_states(s), actions[:, k]), dim=-1), state_dim)
                query = torch.cat((history.flatten(start_dim=1), current), dim=-1)
                delta = model._mean_delta_core(query, key_emb, tokens, density, valid)[0]
                s = model.reconstruct_next_state(s, delta)
                history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
                states.append(s)
            return torch.stack(states, dim=1)

        return rollout

    def inputs(b, device):
        g = torch.Generator().manual_seed(0)
        s = torch.zeros(b, state_dim)
        s[:, 0] = 1.0
        s[:, 5] = 2.0
        actions = torch.randn(b, args.horizon, 2, generator=g) * 0.3
        history = supers[:1].reshape(1, width, flat // width)[:, : width - 1].expand(b, -1, -1).contiguous()
        return s.to(device), actions.to(device), history.to(device)

    valid_cpu = torch.ones(supers.shape[0], dtype=torch.bool)
    rows = []
    with torch.no_grad():
        eager_cpu = make_rollout(lld)
        reference = {}
        for b in args.batches:
            s, a, h = inputs(b, "cpu")
            reference[b] = eager_cpu(s, a, h, supers, deltas, density, valid_cpu)
            row = {"batch": b}
            row["eager_cpu"] = timeit(lambda: eager_cpu(s, a, h, supers, deltas, density, valid_cpu))
            rows.append(row)

        # Compiled horizon on CPU, static shapes.
        try:
            compiled = torch.compile(make_rollout(lld), dynamic=False)
            for row in rows:
                b = row["batch"]
                s, a, h = inputs(b, "cpu")
                out = compiled(s, a, h, supers, deltas, density, valid_cpu)
                row["compiled_cpu"] = timeit(lambda: compiled(s, a, h, supers, deltas, density, valid_cpu), repeat=30)
                row["err_compiled_cpu"] = float((out - reference[b]).abs().max())
        except Exception as error:
            print(f"compiled cpu failed: {type(error).__name__}: {str(error)[:300]}")

        # GPU: eager and CUDA graph.
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            gpu_lld = build_models().to(dev).eval()
            gpu_lld.load(resolve_model_dir(args.model), device=dev)
            gpu = gpu_lld["LLD"]
            g_supers, g_deltas, g_density = supers.to(dev), deltas.to(dev), density.to(dev)
            g_valid = valid_cpu.to(dev)
            rollout = make_rollout(gpu)
            for row in rows:
                b = row["batch"]
                s_cpu, a_cpu, h_cpu = inputs(b, "cpu")
                s, a, h = s_cpu.to(dev), a_cpu.to(dev), h_cpu.to(dev)

                def eager_gpu():
                    out = rollout(s_cpu.to(dev), a_cpu.to(dev), h_cpu.to(dev), g_supers, g_deltas, g_density, g_valid)
                    return out.cpu()

                row["eager_gpu"] = timeit(eager_gpu)
                row["err_eager_gpu"] = float((eager_gpu() - reference[b]).abs().max())

                # CUDA graph: static input buffers, one replay per rollout.
                static_s, static_a, static_h = s.clone(), a.clone(), h.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        rollout(static_s, static_a, static_h, g_supers, g_deltas, g_density, g_valid)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out = rollout(static_s, static_a, static_h, g_supers, g_deltas, g_density, g_valid)

                def graph_gpu():
                    static_s.copy_(s_cpu, non_blocking=False)
                    static_a.copy_(a_cpu, non_blocking=False)
                    static_h.copy_(h_cpu, non_blocking=False)
                    graph.replay()
                    return static_out.cpu()

                row["graph_gpu"] = timeit(graph_gpu, repeat=200)
                row["err_graph_gpu"] = float((graph_gpu() - reference[b]).abs().max())

    print("\nmedian ms for one rollout, CPU tensors in/out (lower is better); err = max abs diff vs eager CPU")
    keys = ["eager_cpu", "compiled_cpu", "eager_gpu", "graph_gpu", "err_compiled_cpu", "err_eager_gpu", "err_graph_gpu"]
    print(f"{'batch':>6} " + " ".join(f"{k:>18}" for k in keys))
    for row in rows:
        print(f"{row['batch']:>6} " + " ".join(f"{row.get(k, float('nan')):>18.4g}" for k in keys))


if __name__ == "__main__":
    main()
