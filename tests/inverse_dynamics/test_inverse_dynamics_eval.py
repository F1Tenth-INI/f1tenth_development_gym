# tests/inverse_dynamics/test_inverse_dynamics_eval.py
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pytest

# Force headless plotting for CI/offline
import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt

from tests.inverse_dynamics import config as CFG
from tests.inverse_dynamics.eval_utils import (
    gather_csv_files, load_states_controls,
    build_window_by_end, enumerate_end_indices,
    eval_single_pass, eval_progressive_window, per_step_scaled_errors,
)
from utilities.InverseDynamics import HARD_CODED_STATE_STD

from collections import defaultdict

# ---- Overlay plotting helpers ------------------------------------------------
# State layout used across the test harness and solvers
_STATE_NAMES = [
    "angular_vel_z","linear_vel_x","linear_vel_y","pose_theta",
    "pose_theta_sin","pose_theta_cos","pose_x","pose_y","slip_angle","steering_angle"
]
_NAME2IDX = {n: i for i, n in enumerate(_STATE_NAMES)}

def _time_vector(n: int, dt: float) -> np.ndarray:
    # Use GT time if you later expose it; for now we follow spec fallback to a uniform grid.
    return (np.arange(n, dtype=np.float64) * float(dt))

def _plot_overlay_feature(
    stem: str,
    solver: str,
    T: int,
    init: str,
    feature: str,
    states_gt: np.ndarray,
    dt: float,
    segments: list[dict],
    out_dir: Path,
):
    """
    Time-series overlay for a single feature.
    Non-obvious bits:
      • Predictions arrive newest→older; we store them older→newer and plot on [start..end].
      • Color encodes 'distance from anchor': green (near anchor, i.e., newest) → red (farthest past).
        Implemented via a per-trajectory gradient with cmap='RdYlGn_r' and parameter p∈[0,1].
    """
    idx = _NAME2IDX[feature]
    t = _time_vector(len(states_gt), dt)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}_{solver}_T{T}_{init}_{feature}.png"

    fig, ax = plt.subplots()
    # GT across full horizon
    ax.plot(t, states_gt[:, idx], linestyle=":", color="k", linewidth=1.0, label="GT")
    # All predicted segments (overlaps simply overdrawn; no merging)
    for s in segments:
        start, end = int(s["start"]), int(s["end"])
        pred = s["pred"]                      # shape (end-start+1, D), older→newer
        t_seg = t[start : end + 1]
        y_seg = pred[:, idx]
        # Gradient color: p=0 (green, near anchor/newest) at the last point; p=1 (red) at the first/oldest
        p = np.linspace(1.0, 0.0, len(t_seg), dtype=np.float64)
        ax.scatter(t_seg, y_seg, c=p, cmap="RdYlGn_r", s=8, linewidths=0)

    ax.set_xlabel("time [s]")
    ax.set_ylabel(feature)
    ax.set_title(f"{solver} | T={T} | init={init} | {stem}")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)

def _plot_overlay_xy(
    stem: str,
    solver: str,
    T: int,
    init: str,
    states_gt: np.ndarray,
    dt: float,  # unused but kept for symmetry with _plot_overlay_feature
    segments: list[dict],
    out_dir: Path,
):
    """
    XY path overlay.
    Non-obvious bits:
      • Equal aspect is forced so geometry is faithful.
      • Same per-trajectory green→red gradient (near→far from anchor).
    """
    xi, yi = _NAME2IDX["pose_x"], _NAME2IDX["pose_y"]
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}_{solver}_T{T}_{init}_XY.png"

    fig, ax = plt.subplots()
    # GT full path (thicker, solid)
    ax.plot(states_gt[:, xi], states_gt[:, yi], "k-", linewidth=2.0, label="GT")

    for s in segments:
        pred = s["pred"]  # older→newer
        x_seg, y_seg = pred[:, xi], pred[:, yi]
        p = np.linspace(1.0, 0.0, len(x_seg), dtype=np.float64)  # 1=oldest(red), 0=newest(green)
        ax.scatter(x_seg, y_seg, c=p, cmap="RdYlGn_r", s=6, linewidths=0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{solver} | T={T} | init={init} | {stem}")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)

def _render_overlays_for_bucket(
    stem: str,
    solver: str,
    T: int,
    init: str,
    states_gt: np.ndarray,
    dt: float,
    segments: list[dict],
):
    """Render all requested feature plots + the XY plot for a single (file, solver, T, init) bucket."""
    if not CFG.PLOT_OVERLAY or not segments:
        return
    out_dir = Path(CFG.PLOTS_OVERLAY_DIR) / f"{stem}_{solver}_T{T}_{init}"
    for feat in CFG.PLOT_OVERLAY_FEATURES:
        _plot_overlay_feature(stem, solver, T, init, feat, states_gt, dt, segments, out_dir)
    _plot_overlay_xy(stem, solver, T, init, states_gt, dt, segments, out_dir)


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
from collections import defaultdict
_AGG_CURVES: dict[tuple, list[np.ndarray]] = defaultdict(list)

def _add_curve_to_bundle(file_name: str, solver: str, T: int, init: str, rmse_step: np.ndarray):
    _AGG_CURVES[(file_name, solver, int(T), init)].append(rmse_step.astype(np.float32))

def _plot_agg_series(curves: list[np.ndarray], title: str, out_png: Path, out_csv: Path | None):
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

    # --- Overlay collection (per bucket = (file, solver, T, init)) ---
    segments_by_bucket: dict[tuple, list[dict]] = defaultdict(list)

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

                            # --- Collect one overlay segment for the current window ---------------------
                            start_idx = int(end_idx - T)
                            end_idx_i = int(end_idx)

                            # states_pred is newest→older of length T+1; plot wants older→newer
                            pred_old2new = states_pred[::-1].copy()

                            # (Optional but good): sanity guard against off-by-one
                            assert pred_old2new.shape[0] == (end_idx_i - start_idx + 1), \
                                f"Segment length mismatch: got {pred_old2new.shape[0]}, expected {(end_idx_i - start_idx + 1)}"

                            bucket_key = (Path(file).name, solver, int(T), str(init))
                            segments_by_bucket[bucket_key].append({
                                "start": start_idx,
                                "end": end_idx_i,  # inclusive
                                "pred": pred_old2new,  # shape (end-start+1, D), older→newer
                            })

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
                        # --- Collect one overlay segment for the current window ---------------------
                        start_idx = int(end_idx - T)
                        end_idx_i = int(end_idx)
                        pred_old2new = states_pred[::-1].copy()
                        assert pred_old2new.shape[0] == (end_idx_i - start_idx + 1)

                        bucket_key = (Path(file).name, solver, int(T), str(init_label))
                        segments_by_bucket[bucket_key].append({
                            "start": start_idx,
                            "end": end_idx_i,
                            "pred": pred_old2new,
                        })

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

    # ---- Overlay plots: GT vs. all predicted segments per (file, solver, T, init)
    if CFG.PLOT_OVERLAY and segments_by_bucket:
        stem = Path(file).stem
        dt_used = float(info.get("dt", 0.01))
        # Only render for the current 'file' (collector could hold other files in broader runs)
        for (fname, solver_name, T_val, init_val), segs in segments_by_bucket.items():
            if fname != Path(file).name:
                continue
            _render_overlays_for_bucket(
                stem=stem,
                solver=str(solver_name),
                T=int(T_val),
                init=str(init_val),
                states_gt=states,  # full-length GT states for this file
                dt=dt_used,
                segments=segs,
            )

