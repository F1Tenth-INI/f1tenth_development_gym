#!/usr/bin/env python3
"""
Create improved learning curve figures with better readability:
- larger figures and DPI
- colorblind-friendly palette
- rolling smoothing of medians and CIs
- thinner CI shading and stronger mean lines
- legend placed outside to avoid covering curves
- optional Plotly interactive export

Usage: run from repo root or adjust glob paths as needed.
"""
import argparse
import ast
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_hex


# --------- Configure experiments (reuse patterns from original script) ---------
EXPERIMENTS = {
    "uniform": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_CUSTOM_uniform*/learning_metrics.csv"),
    # "proportional (TD-error; alpha=0.3)": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_OneBatch_TD_A_0.3*/learning_metrics.csv"),
    # "proportional (TD-error; alpha=0.5)": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_OneBatch_TD_A_0.5*/learning_metrics.csv"),
    # "proportional (TD-error; alpha=0.8)": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_OneBatch_TD_A_0.8*/learning_metrics.csv"),
    "proportional (State; rew=3, d=3, e=5)": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_State_Wrew_3.0_Wd_3.0_We_5.0*/learning_metrics.csv"),
    "proportional (State; rew=7, d=3, e=5)": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_State_Wrew_7.0_Wd_3.0_We_5.0*/learning_metrics.csv"),
    "proportional (State; vel=3, rew=5, d=5, e=5)": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_State_VelW_3.0_Wrew_5_Wd_5_We_5*/learning_metrics.csv"),
    "proportional (State; rew=3, d=5, e=3)": glob.glob("TrainingLite/rl_racing/models/RCA2-ReportFinal_UTD125_State_Wrew_3.0_Wd_5.0_We_3.0*/learning_metrics.csv"),
}

METRICS = {
    "episode_mean_step_reward": {
        "columns": ["episode_mean_step_rewards"],
        "label": "Episode Mean Step Reward",
        "output": "learning_curve_episode_mean_step_reward.png",
    },
    "episode_reward": {
        "columns": ["episode_rewards"],
        "label": "Episode Reward",
        "output": "learning_curve_episode_reward.png",
    },
    "lap_time": {
        "columns": ["lap_times"],
        "label": "Lap Time (s)",
        "output": "learning_curve_lap_time.png",
    },
    "critic_loss": {
        "columns": ["critic_loss"],
        "label": "Critic Loss",
        "output": "learning_curve_critic_loss.png",
    },
    "actor_loss": {
        "columns": ["actor_loss"],
        "label": "Actor Loss",
        "output": "learning_curve_actor_loss.png",
    },
}


def parse_metric_column(series) -> np.ndarray:
    out = []
    for v in series:
        if v is None:
            out.append(np.nan)
            continue
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                out.append(np.nan)
                continue
            if s.startswith("[") and s.endswith("]"):
                try:
                    vals = ast.literal_eval(s)
                    out.append(np.mean(vals) if len(vals) else np.nan)
                except Exception:
                    out.append(np.nan)
                continue
            try:
                out.append(float(s))
            except Exception:
                out.append(np.nan)
            continue
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


def load_curve(csv_path: str, metric_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    x = df["total_timesteps"].to_numpy(dtype=float)
    y = None
    for column in metric_cfg["columns"]:
        if column in df.columns:
            y = parse_metric_column(df[column])
            break
    if y is None:
        raise ValueError(f"None of the metric columns {metric_cfg['columns']} exist in {csv_path}")
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def align_and_summarize(curves: List[Tuple[np.ndarray, np.ndarray]], x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = []
    for x, y in curves:
        if len(x) < 2:
            continue
        yi = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
        ys.append(yi)
    if not ys:
        return np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)
    Y = np.array(ys)
    median = np.nanmedian(Y, axis=0)
    q25 = np.nanpercentile(Y, 25, axis=0)
    q75 = np.nanpercentile(Y, 75, axis=0)
    return median, q25, q75


def smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    return pd.Series(arr).rolling(window=window, center=True, min_periods=1).median().to_numpy()


def infer_shared_prefix(experiments: Dict[str, List[str]]) -> str:
    model_names = []
    for files in experiments.values():
        for csv_path in files:
            model_dir = Path(csv_path).parent.name
            model_names.append(model_dir)

    if not model_names:
        return "all_experiments"

    inferred = os.path.commonprefix(model_names).strip("_- ")
    inferred = re.sub(r"[^A-Za-z0-9_\-]+", "_", inferred)
    return inferred or "all_experiments"


def make_plots(
    save_dir: str,
    output_name: str | None,
    dpi: int,
    figsize: Tuple[int, int],
    smooth_window: int,
    interactive: bool,
):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
    })

    run_name = output_name.strip() if output_name is not None else infer_shared_prefix(EXPERIMENTS)
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    raw_save_path = save_path / "plots"
    rolling_save_path = save_path / "plots_rolling_average"
    raw_save_path.mkdir(parents=True, exist_ok=True)
    rolling_save_path.mkdir(parents=True, exist_ok=True)

    # colorblind palette sized to experiments
    labels = list(EXPERIMENTS.keys())
    palette = sns.color_palette("colorblind", n_colors=max(2, len(labels)))
    colors = {lab: palette[i % len(palette)] for i, lab in enumerate(labels)}
    colors["uniform"] = "black"
    plotly_colors = {lab: to_hex(color) if not isinstance(color, str) else color for lab, color in colors.items()}

    plot_variants = [
        (False, raw_save_path, "raw", 2.0, 2.6),
        (True, rolling_save_path, f"roll{smooth_window}", 2.8, 3.4),
    ]

    for metric_name, cfg in METRICS.items():
        # collect curves per method
        all_x = []
        method_curves = {}
        for method, files in EXPERIMENTS.items():
            curves = []
            for csv_path in files:
                try:
                    x, y = load_curve(csv_path, cfg)
                    curves.append((x, y))
                except Exception as e:
                    print(f"Skipping {csv_path} for {metric_name}: {e}")
            method_curves[method] = curves
            for x, _ in curves:
                if len(x):
                    all_x.append(x.max())

        if not all_x:
            print(f"No valid data for metric {metric_name}")
            continue

        x_max = max(all_x)
        x_grid = np.linspace(0, x_max, 300)

        for apply_smoothing, variant_path, variant_suffix, other_width, uniform_width in plot_variants:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            for method, curves in method_curves.items():
                if not curves:
                    continue
                med, q25, q75 = align_and_summarize(curves, x_grid)
                if apply_smoothing:
                    med = smooth_series(med, smooth_window)
                    q25 = smooth_series(q25, max(1, smooth_window // 2))
                    q75 = smooth_series(q75, max(1, smooth_window // 2))
                linewidth = uniform_width if method == "uniform" else other_width

                ax.plot(x_grid, med, label=method, color=colors.get(method), linewidth=linewidth)
                ax.fill_between(x_grid, q25, q75, color=colors.get(method), alpha=0.12)

            ax.set_xlabel("training step")
            ax.set_ylabel(cfg["label"])
            ax.grid(alpha=0.25)
            suffix_label = "smoothed" if apply_smoothing else "raw"
            ax.set_title(f"Learning Curve: {cfg['label']} ({suffix_label})")

            # put legend on top-left with semi-transparent background
            ax.legend(loc='upper left', frameon=True, framealpha=0.95, fancybox=True)

            plt.tight_layout()
            out_png = variant_path / cfg["output"]
            plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
            print(f"Saved {out_png}")

            if interactive:
                try:
                    import plotly.graph_objects as go

                    figly = go.Figure()
                    for method, curves in method_curves.items():
                        if not curves:
                            continue
                        med, q25, q75 = align_and_summarize(curves, x_grid)
                        if apply_smoothing:
                            med = smooth_series(med, smooth_window)
                            q25 = smooth_series(q25, max(1, smooth_window // 2))
                            q75 = smooth_series(q75, max(1, smooth_window // 2))
                        line_width = uniform_width if method == "uniform" else other_width
                        line_color = plotly_colors.get(method, "black")
                        figly.add_trace(go.Scatter(x=x_grid, y=med, mode='lines', name=method, line=dict(color=line_color, width=line_width)))
                        figly.add_trace(go.Scatter(x=np.concatenate([x_grid, x_grid[::-1]]), y=np.concatenate([q75, q25[::-1]]), fill='toself', fillcolor='rgba(0,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False))

                    figly.update_layout(title=f"Learning Curve: {cfg['label']} ({suffix_label})", xaxis_title='training step', yaxis_title=cfg['label'])
                    out_html = variant_path / cfg["output"].replace(".png", ".html")
                    figly.write_html(str(out_html))
                    print(f"Saved interactive {out_html}")
                except Exception as e:
                    print(f"Plotly export failed: {e}")

            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create improved learning curve figures")
    parser.add_argument("--save-dir", default="learning_curves_better", help="Directory to save figures")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Name of the subfolder inside save-dir; use None to auto-pick a shared prefix from the selected models",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--width", type=int, default=12, help="Figure width in inches")
    parser.add_argument("--height", type=int, default=6, help="Figure height in inches")
    parser.add_argument("--smooth", type=int, default=51, help="Rolling median window for smoothing (odd recommended)")
    parser.add_argument("--interactive", action='store_true', help="Also save interactive Plotly HTML files")

    args = parser.parse_args()
    output_name = None if args.output_name in (None, "None", "none", "") else args.output_name
    make_plots(
        save_dir=args.save_dir,
        output_name=output_name,
        dpi=args.dpi,
        figsize=(args.width, args.height),
        smooth_window=args.smooth,
        interactive=args.interactive,
    )


if __name__ == '__main__':
    main()
