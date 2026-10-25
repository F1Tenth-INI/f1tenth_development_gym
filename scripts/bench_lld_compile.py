"""Does compiling more of the LLD rollout help? Times a 25-step rollout at several batch sizes.

    python scripts/bench_lld_compile.py --model hwm_weighted_SE_test_7

Variants: eager everything; the current setup (only the attention core compiled); the whole rollout step
compiled (query build + attention + reconstruct + history update); the whole horizon compiled as one graph.
Single thread, real model and key bank. The last two are also run with the memory-free residual net.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

torch.set_num_threads(1)

from TrainingLite.hwm.models.networks import build_models  # noqa: E402
from TrainingLite.hwm.paths import resolve_model_dir  # noqa: E402
from TrainingLite.hwm.shared.memory_manager import MemoryManager  # noqa: E402
from TrainingLite.hwm.shared.observation import _normalize_states, lld_rows_for_super  # noqa: E402
from utilities.Settings import Settings  # noqa: E402


def timeit(fn, repeat: int = 40, warm: int = 5) -> float:
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
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 25, 50, 100])
    parser.add_argument("--horizon", type=int, default=25)
    args = parser.parse_args()

    models = build_models().to("cpu").eval()
    models.load(resolve_model_dir(args.model), device="cpu")
    lld = models["LLD"]
    state_dim = int(Settings.HWM_STATE_DIM)
    width = int(Settings.HWM_SUPER_STATE_SIZE)
    flat = lld.attention_norm_projector[0].in_features

    memory = MemoryManager(device="cpu")
    memory.attach_projector(lld.attention_norm_projector)
    memory.import_rows(torch.load(resolve_model_dir(args.model) / "memory.pt", weights_only=True))
    memory.refresh_embeddings()
    supers, deltas, density, _ = memory.lld_bank()

    with torch.no_grad():
        lld._compile_rollout_step(flat)
        key_emb = lld.attention_norm_projector(supers)
        tokens = deltas - lld._correct(supers)
        valid = torch.ones(supers.shape[0], dtype=torch.bool)
        empty = (key_emb[:0], tokens[:0], density[:0], valid[:0])

        def step(s, a, history, k, t, dn, v, memory_free: bool):
            current = lld_rows_for_super(torch.cat((_normalize_states(s), a), dim=-1), state_dim)
            query = torch.cat((history.flatten(start_dim=1), current), dim=-1)
            if memory_free:
                delta = lld._correct(query)
            else:
                delta = lld._mean_delta_core(query, k, t, dn, v)[0]
            s = lld.reconstruct_next_state(s, delta)
            history = torch.cat((history[:, 1:], current.unsqueeze(1)), dim=1)
            return s, history

        def horizon_fn(s, actions, history, k, t, dn, v, memory_free: bool):
            states = [s]
            for i in range(actions.shape[1]):
                s, history = step(s, actions[:, i], history, k, t, dn, v, memory_free)
                states.append(s)
            return torch.stack(states, dim=1)

        def make_inputs(b):
            s = torch.zeros(b, state_dim)
            s[:, 0] = 1.0
            s[:, 5] = 2.0  # some speed, arbitrary
            actions = torch.randn(b, args.horizon, 2) * 0.3
            history = supers[:1].reshape(1, width, flat // width)[:, : width - 1].expand(b, -1, -1).contiguous()
            return s, actions, history

        results = []
        compiled_step = torch.compile(step, dynamic=False)  # one graph per batch size; the planner only uses 2 sizes
        for b in args.batches:
            s, actions, history = make_inputs(b)
            row = {"batch": b}
            for label, memory_free in (("memory", False), ("free", True)):
                kt = (key_emb, tokens, density, valid)
                row[f"eager_{label}"] = timeit(lambda: horizon_fn(s, actions, history, *kt, memory_free))
                if not memory_free:
                    core = lld.mean_delta_compiled

                    def current_setup():
                        st, h = s, history
                        out = [st]
                        for i in range(args.horizon):
                            cur = lld_rows_for_super(torch.cat((_normalize_states(st), actions[:, i]), dim=-1), state_dim)
                            q = torch.cat((h.flatten(start_dim=1), cur), dim=-1)
                            delta = core(q, *kt)[0]
                            st = lld.reconstruct_next_state(st, delta)
                            h = torch.cat((h[:, 1:], cur.unsqueeze(1)), dim=1)
                            out.append(st)
                        return torch.stack(out, dim=1)

                    row["current_core_only"] = timeit(current_setup)

                def whole_step_compiled():
                    st, h = s, history
                    out = [st]
                    for i in range(args.horizon):
                        st, h = compiled_step(st, actions[:, i], h, *kt, memory_free)
                        out.append(st)
                    return torch.stack(out, dim=1)

                try:
                    row[f"compiled_step_{label}"] = timeit(whole_step_compiled)
                except Exception as error:  # report, do not hide
                    row[f"compiled_step_{label}"] = float("nan")
                    print(f"compiled step {label} failed: {type(error).__name__}: {str(error)[:200]}")
            results.append(row)

        try:
            compiled_horizon = torch.compile(horizon_fn, dynamic=False)
            for b in args.batches:
                s, actions, history = make_inputs(b)
                kt = (key_emb, tokens, density, valid)
                for row in results:
                    if row["batch"] == b:
                        row["compiled_horizon_memory"] = timeit(
                            lambda: compiled_horizon(s, actions, history, *kt, False), repeat=20
                        )
                        row["compiled_horizon_free"] = timeit(
                            lambda: compiled_horizon(s, actions, history, *kt, True), repeat=20
                        )
        except Exception as error:
            print(f"compiled horizon failed: {type(error).__name__}: {str(error)[:300]}")

    keys = [
        "eager_memory", "current_core_only", "compiled_step_memory", "compiled_horizon_memory",
        "eager_free", "compiled_step_free", "compiled_horizon_free",
    ]
    print("\nmedian ms for one 25-step rollout (lower is better)")
    print(f"{'batch':>6} " + " ".join(f"{k:>24}" for k in keys))
    for row in results:
        print(f"{row['batch']:>6} " + " ".join(f"{row.get(k, float('nan')):>24.2f}" for k in keys))


if __name__ == "__main__":
    main()
