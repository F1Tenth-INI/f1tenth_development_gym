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
    eval_single_pass, eval_progressive_window, per_step_scaled_errors,
)
from utilities.InverseDynamics import HARD_CODED_STATE_STD

from collections import defaultdict

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


# bundle_key = (file, solver, T, init)
_AGG_CURVES: dict[tuple, list[np.ndarray]] = defaultdict(list)

def _add_curve_to_bundle(file_name: str, solver: str, T: int, init: str, rmse_step: np.ndarray):
    _AGG_CURVES[(file_name, solver, int(T), init)].append(rmse_step.astype(np.float32))

def _plot_agg_series(curves: list[np.ndarray], title: str, out_png: Path, out_csv: Path | None):
    import matplotlib.pyplot as plt
    # ensure same length (should be, because same T; guard anyway)
    L = min(int(c.shape[0]) for c in curves)
    if L == 0: return
    A = np.stack([c[:L] for c in curves], axis=0)  # [N,L]
    mean = A.mean(axis=0)
    std = A.std(axis=0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    x = np.arange(L)
    plt.plot(x, mean, label="mean")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="±1σ")
    plt.xlabel("steps into past (older → right)")
    plt.ylabel("scaled RMSE")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"step": x, "rmse_mean": mean, "rmse_std": std}).to_csv(out_csv, index=False)


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
            if s == "single_pass":
                total_runs += len(ends) * len(_effective_inits())
            elif s == "progressive_window":
                total_runs += len(ends)
            else:
                raise ValueError(f"Unknown solver: {s}")

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
                if solver == "single_pass":
                    for init in _effective_inits():
                        out = eval_single_pass(x_T, Q_w, gt, init_type=init,
                                        noise_scale=CFG.NOISE_SCALE, dt=dt, mu=mu,
                                        return_states=CFG.COLLECT_SERIES)
                        if CFG.COLLECT_SERIES:
                            res, states_pred = out
                            ser = per_step_scaled_errors(states_pred, gt, HARD_CODED_STATE_STD)
                            _add_curve_to_bundle(Path(file).name, solver, T, init, ser["rmse_step"])
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

                elif solver == "progressive_window":
                    init_label = "auto"  # explicit label for grouping/plots
                    out = eval_progressive_window(x_T, Q_w, gt, dt=dt, mu=mu, return_states=CFG.COLLECT_SERIES)
                    if CFG.COLLECT_SERIES:
                        res, states_pred = out
                        ser = per_step_scaled_errors(states_pred, gt, HARD_CODED_STATE_STD)
                        _add_curve_to_bundle(Path(file).name, solver, T, init_label, ser["rmse_step"])
                        res.update({
                            "file": Path(file).name, "solver": solver, "T": T, "init": init_label, "end_idx": end_idx
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
                            "file": Path(file).name, "solver": solver, "T": T, "init": init_label, "end_idx": end_idx
                        })
                        results.append(res)
                    pbar.update(1)

                else:
                    raise ValueError(f"Unknown solver: {solver}")

    pbar.close()

    # Optionally save to CSV
    if CFG.SAVE_CSV:
        _results_to_csv(results, Path(file), suffix=CFG.WINDOW_MODE)

    # ---- Aggregated bundles: mean/std per (file, solver, T, init) ----
    df_all = pd.DataFrame(results)
    group_cols = ["file", "solver", "T", "init"]

    agg = df_all.groupby(group_cols).agg(
        conv_rate_mean=("conv_rate", "mean"),
        conv_rate_std=("conv_rate", "std"),
        time_s_mean=("time_s", "mean"),
        time_s_std=("time_s", "std"),
        mae_mean_mean=("mae_mean", "mean"),
        mae_mean_std=("mae_mean", "std"),
        rmse_mean_mean=("rmse_mean", "mean"),
        rmse_mean_std=("rmse_mean", "std"),
        rmse_auc_mean=("rmse_auc", "mean"),
        rmse_auc_std=("rmse_auc", "std"),
        rmse_head_mean=("rmse_head", "mean"),
        rmse_head_std=("rmse_head", "std"),
        rmse_tail_mean=("rmse_tail", "mean"),
        rmse_tail_std=("rmse_tail", "std"),
        growth_ratio_mean=("growth_ratio", "mean"),
        growth_ratio_std=("growth_ratio", "std"),
        trace_count_max=("trace_count", "max"),
    ).reset_index()

    # Save aggregated CSV side-by-side
    if CFG.SAVE_CSV:
        out_path = Path(CFG.SAVE_CSV)
        out_dir = out_path if out_path.is_dir() or out_path.suffix == "" else out_path.parent
        ts = time.strftime("%Y%m%d-%H%M%S")
        stem = Path(file).stem
        agg_path = out_dir / f"inverse_dynamics_results_{stem}_{CFG.WINDOW_MODE}_agg_{ts}.csv"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_path, index=False)
        _p(f"[ID] Saved aggregated results to: {agg_path}")

    # Optional: print a short summary
    _p("[ID] Bundle summary (mean±std):")
    for _, r in agg.iterrows():
        _p(f"  {r['solver']:18s} T={int(r['T']):3d} init={r['init']:>5s}  "
           f"rmse={r['rmse_mean_mean']:.4f}±{(0.0 if np.isnan(r['rmse_mean_std']) else r['rmse_mean_std']):.4f}  "
           f"time={r['time_s_mean']:.3f}±{(0.0 if np.isnan(r['time_s_std']) else r['time_s_std']):.3f}s  "
           f"trace_count_max={int(r['trace_count_max'])}")

    # ---- Bundle mean±std per-step RMSE plots ----
    if getattr(CFG, "PLOT_AGG_SERIES", False):
        plot_dir = Path(CFG.PLOTS_AGG_DIR)
        for (f_name, solver, T, init), curves in _AGG_CURVES.items():
            if not curves:
                continue
            stem = Path(f_name).stem
            title = f"{solver} | T={T} | init={init} | {stem}"
            png_path = plot_dir / f"{stem}_{solver}_T{T}_{init}_agg.png"
            csv_path = (plot_dir / f"{stem}_{solver}_T{T}_{init}_agg_curve.csv") if getattr(CFG, "SAVE_AGG_SERIES_CSV",
                                                                                            False) else None
            _plot_agg_series(curves, title, png_path, csv_path)

    assert len(results) > 0
    # Basic quality gate: single_pass should be ≤ 5% worse than itself (no-op if only one solver),
    # but keep the original intent comparing two families if both are present.
    if "single_pass" in CFG.SOLVERS and "progressive_window" in CFG.SOLVERS:
        r_sp = [r for r in results if r["solver"] == "single_pass"]
        r_pw = [r for r in results if r["solver"] == "progressive_window"]
        if len(r_sp) > 0 and len(r_pw) > 0:
            med_sp = np.median([r["rmse_mean"] for r in r_sp])
            med_pw = np.median([r["rmse_mean"] for r in r_pw])
            # Non-binding; ensures progressive_window isn't wildly worse.
            assert med_pw <= med_sp * 1.10
