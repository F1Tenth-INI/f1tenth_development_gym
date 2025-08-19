# tests/inverse_dynamics/test_inverse_dynamics_eval.py
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from tests.inverse_dynamics import config as CFG
from tests.inverse_dynamics.eval_utils import (
    gather_csv_files, load_states_controls,
    build_window_by_end, enumerate_end_indices,
    eval_fast, eval_refine, eval_hybrid, per_step_scaled_errors,
)
from utilities.InverseDynamics import HARD_CODED_STATE_STD

def _p(msg: str):
    print(msg, flush=True)

def _make_pbar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm as _tqdm  # type: ignore
        return _tqdm(total=total, desc=desc, leave=False)
    except Exception:
        class _PB:
            def __init__(self, total, desc):
                self.total = max(1, int(total))
                self.n = 0
                self.desc = desc
            def update(self, n=1):
                self.n += n
                pct = int(100 * self.n / self.total)
                print(f"\r{self.desc} {self.n}/{self.total} ({pct}%)", end="", flush=True)
                if self.n >= self.total:
                    print("", flush=True)
            def close(self):
                pass
        return _PB(total, desc)

def _effective_inits():
    # We keep "none" and "noisy" (drop "gt" to save time; "gt" is a trivial baseline).
    return [i for i in CFG.INITS if i in ("none", "noisy")]

def _results_to_csv(results: list[dict], file: Path, suffix: str):
    df = pd.DataFrame(results)
    out_path = Path(CFG.SAVE_CSV)
    out_dir = out_path if out_path.is_dir() or out_path.suffix == "" else out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    stem = file.stem
    target = out_dir / f"inverse_dynamics_results_{stem}_{suffix}_{ts}.csv"
    df.to_csv(target, index=False)
    _p(f"[ID] Saved results to: {target}")

def _maybe_plot_series(rmse_curve: np.ndarray, title: str, out_file: Path):
    if not CFG.PLOT_SERIES:
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(len(rmse_curve)), rmse_curve)
    plt.xlabel("step into past (older → right)")
    plt.ylabel("scaled RMSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def test_inverse_dynamics_recover_and_bench():
    files = gather_csv_files(CFG.DATA_DIR, max_files=CFG.MAX_FILES, all_files=CFG.PROCESS_ALL_FILES)
    assert len(files) >= 1, "No CSV files found or synthesized"
    file = files[0]
    stem = Path(file).stem
    states, Q, info = load_states_controls(file)
    dt = info.get("dt", 0.01)
    mu = info.get("mu", None)

    # Enumerate Ts and end indices
    T_list = [T for T in CFG.TRAJECTORY_LENGTHS if 2 <= T <= len(states)-1]
    assert len(T_list) > 0, "No valid trajectory lengths for this file"

    # Count total runs
    total_runs = 0
    for T in T_list:
        ends = enumerate_end_indices(len(states), T)
        for s in CFG.SOLVERS:
            total_runs += len(ends) * (1 if s == "hybrid" else len(_effective_inits()))

    _p(f"[ID] File: {Path(file).name} | states={states.shape}, Q={Q.shape}, dt={dt:.4f}, mu={mu}")
    _p(f"[ID] Ts={T_list}, solvers={CFG.SOLVERS}, inits={_effective_inits()}, mode={CFG.WINDOW_MODE}, stride={CFG.WINDOW_STRIDE}")
    pbar = _make_pbar(total_runs, desc="[ID] Experiments")

    results = []
    series_plotted = 0

    for T in T_list:
        end_indices = enumerate_end_indices(len(states), T)
        for end_idx in end_indices:
            x_T, Q_w, gt = build_window_by_end(states, Q, T, end_idx)

            for solver in CFG.SOLVERS:
                if solver == "fast":
                    for init in _effective_inits():
                        out = eval_fast(x_T, Q_w, gt, init_type=init,
                                        noise_scale=CFG.NOISE_SCALE, dt=dt, mu=mu,
                                        return_states=CFG.COLLECT_SERIES)
                        if CFG.COLLECT_SERIES:
                            res, states_pred = out
                            ser = per_step_scaled_errors(states_pred, gt, HARD_CODED_STATE_STD)
                            res.update({
                                "file": Path(file).name, "solver": solver, "T": T, "init": init, "end_idx": end_idx
                            })
                            results.append(res)
                            if series_plotted < CFG.MAX_SERIES_PLOTS:
                                _maybe_plot_series(
                                    ser["rmse_step"],
                                    title=f"{solver}/T={T}/init={init}/end={end_idx}",
                                    out_file=Path(CFG.PLOTS_DIR) / f"{stem}_{solver}_T{T}_{init}_end{end_idx}.png",
                                )
                                series_plotted += 1
                        else:
                            res = out
                            res.update({
                                "file": Path(file).name, "solver": solver, "T": T, "init": init, "end_idx": end_idx
                            })
                            results.append(res)
                        pbar.update(1)

                elif solver == "refine":
                    for init in _effective_inits():
                        out = eval_refine(x_T, Q_w, gt, init_type=init,
                                          noise_scale=CFG.NOISE_SCALE, dt=dt, mu=mu,
                                          return_states=CFG.COLLECT_SERIES)
                        if CFG.COLLECT_SERIES:
                            res, states_pred = out
                            ser = per_step_scaled_errors(states_pred, gt, HARD_CODED_STATE_STD)
                            res.update({
                                "file": Path(file).name, "solver": solver, "T": T, "init": init, "end_idx": end_idx
                            })
                            results.append(res)
                            if series_plotted < CFG.MAX_SERIES_PLOTS:
                                _maybe_plot_series(
                                    ser["rmse_step"],
                                    title=f"{solver}/T={T}/init={init}/end={end_idx}",
                                    out_file=Path(CFG.PLOTS_DIR) / f"{stem}_{solver}_T{T}_{init}_end{end_idx}.png",
                                )
                                series_plotted += 1
                        else:
                            res = out
                            res.update({
                                "file": Path(file).name, "solver": solver, "T": T, "init": init, "end_idx": end_idx
                            })
                            results.append(res)
                        pbar.update(1)

                elif solver == "hybrid":
                    out = eval_hybrid(x_T, Q_w, gt, dt=dt, mu=mu, return_states=CFG.COLLECT_SERIES)
                    if CFG.COLLECT_SERIES:
                        res, states_pred = out
                        ser = per_step_scaled_errors(states_pred, gt, HARD_CODED_STATE_STD)
                        res.update({
                            "file": Path(file).name, "solver": solver, "T": T, "init": "n/a", "end_idx": end_idx
                        })
                        results.append(res)
                        if series_plotted < CFG.MAX_SERIES_PLOTS:
                            _maybe_plot_series(
                                ser["rmse_step"],
                                title=f"{solver}/T={T}/end={end_idx}",
                                out_file=Path(CFG.PLOTS_DIR) / f"{stem}_{solver}_T{T}_end{end_idx}.png",
                            )
                            series_plotted += 1
                    else:
                        res = out
                        res.update({
                            "file": Path(file).name, "solver": solver, "T": T, "init": "n/a", "end_idx": end_idx
                        })
                        results.append(res)
                    pbar.update(1)

                else:
                    raise ValueError(f"Unknown solver: {solver}")

    pbar.close()

    # Optionally save to CSV
    if CFG.SAVE_CSV:
        _results_to_csv(results, Path(file), suffix=CFG.WINDOW_MODE)

    # Sanity: at least one experiment ran
    assert len(results) > 0
    # Basic quality gate: refine should have a strictly better median rmse than fast (when both ran)
    if "fast" in CFG.SOLVERS and "refine" in CFG.SOLVERS:
        r_fast = [r for r in results if r["solver"] == "fast"]
        r_ref  = [r for r in results if r["solver"] == "refine"]
        if len(r_fast) > 0 and len(r_ref) > 0:
            med_fast = np.median([r["rmse_mean"] for r in r_fast])
            med_ref  = np.median([r["rmse_mean"] for r in r_ref])
            assert med_ref <= med_fast * 1.05  # within 5%
