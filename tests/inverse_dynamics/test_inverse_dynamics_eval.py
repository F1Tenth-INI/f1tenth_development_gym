# tests/inverse_dynamics/test_inverse_dynamics_eval.py
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tests.inverse_dynamics import config as CFG
from tests.inverse_dynamics.eval_utils import (
    gather_csv_files, load_states_controls,
    eval_fast, eval_refine, eval_hybrid
)

def _build_window(states: np.ndarray, Q: np.ndarray, T: int):
    assert len(states) >= T+1, "Not enough rows for chosen T"
    x_T = states[-1:,:]            # [1,10]
    Q_w = Q[-T:,:]                 # [T,2], old->new
    gt_newest_first = states[-(T+1):,:][::-1].copy()  # [T+1,10]
    return x_T.astype(np.float32), Q_w.astype(np.float32), gt_newest_first.astype(np.float32)

def _select_valid_Ts(n_rows: int, requested: list[int]) -> list[int]:
    max_T = n_rows - 1
    valid = [T for T in requested if 2 <= T <= max_T]
    if not valid and max_T >= 2:
        valid = [min(10, max_T)]
    return valid

def test_inverse_dynamics_recover_and_bench():
    files = gather_csv_files(CFG.DATA_DIR, max_files=CFG.MAX_FILES, all_files=CFG.PROCESS_ALL_FILES)
    assert len(files) >= 1, "No CSV files found or synthesized"
    file = files[0]
    states, Q, info = load_states_controls(file)
    dt = info.get("dt", 0.01)
    mu = info.get("mu", None)

    T_candidates = _select_valid_Ts(len(states), CFG.TRAJECTORY_LENGTHS)
    assert len(T_candidates) > 0, "No valid trajectory lengths for this file"

    results = []
    for T in T_candidates:
        x_T, Q_w, gt = _build_window(states, Q, T)

        for solver in CFG.SOLVERS:
            if solver == "fast":
                for init in CFG.INITS:
                    res = eval_fast(x_T, Q_w, gt, init_type=init, noise_scale=CFG.NOISE_SCALE, dt=dt, mu=mu)
                    res.update({"file": Path(file).name, "solver": solver, "T": T, "init": init})
                    results.append(res)
                    # Basic invariants
                    assert 0.0 <= res["conv_rate"] <= 1.0
                    assert np.isfinite(res["mae_mean"]) and np.isfinite(res["rmse_mean"])

            elif solver == "refine":
                for init in CFG.INITS:
                    res = eval_refine(x_T, Q_w, gt, init_type=init, noise_scale=CFG.NOISE_SCALE, dt=dt, mu=mu)
                    res.update({"file": Path(file).name, "solver": solver, "T": T, "init": init})
                    results.append(res)
                    assert 0.0 <= res["conv_rate"] <= 1.0
                    assert np.isfinite(res["mae_mean"]) and np.isfinite(res["rmse_mean"])

            elif solver == "hybrid":
                res = eval_hybrid(x_T, Q_w, gt, dt=dt, mu=mu)
                res.update({"file": Path(file).name, "solver": solver, "T": T, "init": "n/a"})
                results.append(res)
                assert 0.0 <= res["conv_rate"] <= 1.0
                assert np.isfinite(res["mae_mean"]) and np.isfinite(res["rmse_mean"])
            else:
                raise ValueError(f"Unknown solver: {solver}")

    # Optionally save to CSV (hard-coded path in CFG)
    if CFG.SAVE_CSV:
        df = pd.DataFrame(results)
        Path(CFG.SAVE_CSV).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CFG.SAVE_CSV, index=False)

    # Require that at least one experiment ran
    assert len(results) > 0, "No experiments executed (maybe trajectory lengths too long for this file?)"
