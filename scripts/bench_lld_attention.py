"""Time the LLD memory-attention math in isolation (one rollout step), at several batch sizes and bank sizes.

    python scripts/bench_lld_attention.py --model hwm_weighted_SE_test_7

One rollout step = one ``mean_delta`` call: encode the query superstates, distance to every key, softmax-like
weights, weighted sum of the residual tokens, plus the residual net g(q). Single thread, like run.py.
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
from utilities.Settings import Settings  # noqa: E402


def timeit(fn, repeat: int = 300, warm: int = 30) -> float:
    for _ in range(warm):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e3  # median, ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 12, 25, 50, 100, 200])
    parser.add_argument("--banks", type=int, nargs="+", default=[1000, 4000])
    args = parser.parse_args()

    models = build_models().to("cpu").eval()
    models.load(resolve_model_dir(args.model), device="cpu")
    lld = models["LLD"]
    flat = lld.attention_norm_projector[0].in_features
    emb = int(lld.attention_norm_projector[-1].out_features)
    d = int(lld.prediction_dim)
    print(f"superstate dim {flat}, embedding dim {emb}, delta dim {d}, temperature {lld.attention_temperature}")

    # The model's saved memory: real embeddings matter, random ones underflow exp() and look slower than they are.
    memory = MemoryManager(device="cpu")
    memory.attach_projector(lld.attention_norm_projector)
    payload = torch.load(resolve_model_dir(args.model) / "memory.pt", weights_only=True)
    memory.import_rows(payload)
    memory.refresh_embeddings()
    supers, deltas, density_all, _ = memory.lld_bank()
    print(f"real key bank: {supers.shape[0]} keys")

    rows = []
    with torch.no_grad():
        lld._compile_rollout_step(flat)
        for n in args.banks:
            n = min(n, supers.shape[0])
            keys = lld.attention_norm_projector(supers[:n])
            tokens = deltas[:n] - lld._correct(supers[:n])
            density = density_all[:n]
            valid = torch.ones(n, dtype=torch.bool)
            for b in args.batches:
                pick = torch.randint(0, supers.shape[0], (b,))
                q = supers[pick] + 0.02 * torch.randn(b, flat)
                qe = lld.attention_norm_projector(q)
                logits = lld._memory_logits(qe, keys, valid)
                weights, _, _ = lld._key_weights(logits, density)
                r = {
                    "bank": n,
                    "batch": b,
                    "encode_query": timeit(lambda: lld.attention_norm_projector(q)),
                    "cdist_mask": timeit(lambda: lld._memory_logits(qe, keys, valid)),
                    "key_weights": timeit(lambda: lld._key_weights(logits, density)),
                    "weights_at_tokens": timeit(lambda: weights @ tokens),
                    "residual_net_g": timeit(lambda: lld._correct(q)),
                    "mean_delta_eager": timeit(lambda: lld._mean_delta_core(q, keys, tokens, density, valid)),
                    "mean_delta_compiled": timeit(lambda: lld.mean_delta_compiled(q, keys, tokens, density, valid)),
                    "memory_free_g_only": timeit(lambda: lld.residual_mean(q)),
                }
                s = torch.randn(b, int(Settings.HWM_STATE_DIM))
                delta = torch.randn(b, d) * 0.01
                r["reconstruct_next_state"] = timeit(lambda: lld.reconstruct_next_state(s, delta))
                rows.append(r)

    cols = [
        "encode_query", "cdist_mask", "key_weights", "weights_at_tokens", "residual_net_g",
        "mean_delta_eager", "mean_delta_compiled", "memory_free_g_only", "reconstruct_next_state",
    ]
    print("\nmedian ms per call (one rollout step)")
    print(f"{'bank':>6} {'batch':>6} " + " ".join(f"{c:>20}" for c in cols))
    for r in rows:
        print(f"{r['bank']:>6} {r['batch']:>6} " + " ".join(f"{r[c]:>20.4f}" for c in cols))


if __name__ == "__main__":
    main()
