import glob
import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure your experiments here
# Each method has multiple learning_metrics.csv files (different seeds/runs)
EXPERIMENTS = {
    # "uniform": glob.glob("TrainingLite/rl_racing/models/RCA2-Fixed_maxFreq_70_noCustom*/learning_metrics.csv"),
    # "proportional (TD-error), alpha=0.8": glob.glob("TrainingLite/rl_racing/models/RCA2-Fixed_maxFreq_65_TD_A_0.8*/learning_metrics.csv"),
    # "proportional (TD-error), alpha=0.6": glob.glob("TrainingLite/rl_racing/models/RCA2-Fixed_maxFreq_65_TD_A_0.6*/learning_metrics.csv"),
    # "Proportional (State)": glob.glob("TrainingLite/rl_racing/models/RCA2-Fixed_maxFreq_65_State*/learning_metrics.csv"),

    # "uniform": (glob.glob("TrainingLite/rl_racing/models/Physical-20-CustomUniform-a/learning_metrics.csv") + glob.glob("TrainingLite/rl_racing/models/Physical-20b/learning_metrics.csv")),
    # "proportional (TD-error), alpha=0.6": (
    #     glob.glob("TrainingLite/rl_racing/models/Physical-20d_TD_A_06/learning_metrics.csv")
    #     + glob.glob("TrainingLite/rl_racing/models/Physical-20b_TD_A_06/learning_metrics.csv")
    # ),
    # "Proportional (State)": glob.glob("TrainingLite/rl_racing/models/Physical-20-State_Wrew_10_Wd_10_We_3/learning_metrics.csv"),

    # "uniform": (glob.glob("TrainingLite/rl_racing/models/RCA2-Results-F70_CUSTOM_uniform_A00*/learning_metrics.csv") 
    #             + 
    #             glob.glob("TrainingLite/rl_racing/models/RCA2-Results-F83_CUSTOM_uniform_A00*/learning_metrics.csv")),
    # "proportional (TD-error), alpha=0.6": (
    #     glob.glob("TrainingLite/rl_racing/models/RCA2-Results_noSepBatches-F83_TD_A_0.6_ActorInvTD_False*/learning_metrics.csv")
    #     # + glob.glob("TrainingLite/rl_racing/models/RCA2-Results-F65_TD_A_0.6_ActorInvTD_False*/learning_metrics.csv")
    # ),
    # "Proportional (State)": glob.glob("TrainingLite/rl_racing/models/RCA2-Results-F65_State_Wrew_5_Wd_5_We_5*/learning_metrics.csv"),

    # "uniform": glob.glob("TrainingLite/rl_racing/models/RCA2c-Results*/learning_metrics.csv"),
    # "proportional (TD-error)": glob.glob("TrainingLite/rl_racing/models/RCA2c-maxFreq_65_TD_A_0.6_ActorInvTD_False*/learning_metrics.csv"),
    # "proportional (TD-error, actor Inverse TD)": glob.glob("TrainingLite/rl_racing/models/RCA2c-maxFreq_65_TD_A_0.6_ActorInvTD_True*/learning_metrics.csv"),
}

# ----------------------------------------
#set to None to auto find common prefix 
save_dir_prefix = "Physical-2-uni-2-TD06"
# ----------------------------------------



def infer_save_dir_prefix(experiments):
    model_names = []
    for files in experiments.values():
        for csv_path in files:
            model_dir = os.path.dirname(csv_path)
            model_names.append(os.path.basename(model_dir))

    if not model_names:
        return "all_experiments"

    inferred = os.path.commonprefix(model_names)
    return inferred or "all_experiments"

if not save_dir_prefix:
    save_dir_prefix = infer_save_dir_prefix(EXPERIMENTS)

base_save_dir = "learning_curves"  # directory to save figures
current_save_dir = f"{base_save_dir}/{save_dir_prefix}"

if not os.path.exists(current_save_dir):
    os.makedirs(current_save_dir, exist_ok=True)

HUMAN_SCORE = 0.0  # replace per map/task
X_LABEL = "training step"

METRICS = {
    "episode_mean_step_reward": {
        "columns": ["episode_mean_step_rewards"],
        "label": "Episode Mean Step Reward",
        "output": "learning_curve_episode_mean_step_reward.png",
        "reference_line": 0.0,
        "reference_label": "reference",
    },
    "episode_reward": {
        "columns": ["episode_rewards"],
        "label": "Episode Reward",
        "output": "learning_curve_episode_reward.png",
        "reference_line": 0.0,
        "reference_label": "reference",
    },
    "lap_time": {
        "columns": ["lap_times"],
        "label": "Lap Time (s)",
        "output": "learning_curve_lap_time.png",
        "reference_line": None,
        "reference_label": "reference",
    },
    "critic_loss": {
        "columns": ["critic_loss"],
        "label": "Critic Loss",
        "output": "learning_curve_critic_loss.png",
        "reference_line": None,
        "reference_label": "reference",
    },
    "actor_loss": {
        "columns": ["actor_loss"],
        "label": "Actor Loss",
        "output": "learning_curve_actor_loss.png",
        "reference_line": None,
        "reference_label": "reference",
    },
}

def parse_metric_column(series):
    # Handles scalar values or stringified lists like "[120.0, 150.0]".
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
                except (ValueError, SyntaxError):
                    out.append(np.nan)
                continue
            try:
                out.append(float(s))
            except ValueError:
                out.append(np.nan)
            continue

        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)

    return np.array(out, dtype=float)

def load_curve(csv_path, metric_name, metric_cfg):
    df = pd.read_csv(csv_path)
    x = df["total_timesteps"].to_numpy(dtype=float)

    y = None
    for column in metric_cfg["columns"]:
        if column in df.columns:
            y = parse_metric_column(df[column])
            break

    if y is None:
        raise ValueError(
            f"None of the metric columns {metric_cfg['columns']} exist in {csv_path}"
        )

    # keep finite points only
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def align_and_summarize(curves, x_grid):
    ys = []
    for x, y in curves:
        if len(x) < 2:
            continue
        # interpolate each run to common grid
        yi = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
        ys.append(yi)
    Y = np.array(ys)
    median = np.nanmedian(Y, axis=0)
    q25 = np.nanpercentile(Y, 25, axis=0)
    q75 = np.nanpercentile(Y, 75, axis=0)
    return median, q25, q75

palette = ["black", "green", "red", "blue", "orange", "purple", "cyan", "magenta"]
colors = {
    method: palette[idx % len(palette)]
    for idx, method in enumerate(EXPERIMENTS.keys())
}

for metric_name, metric_cfg in METRICS.items():
    # Build common grid per metric
    all_x = []
    method_curves = {}

    for method, files in EXPERIMENTS.items():
        curves = []
        for csv_path in files:
            try:
                x, y = load_curve(csv_path, metric_name, metric_cfg)
                curves.append((x, y))
            except ValueError as exc:
                print(f"Skipping {csv_path} for metric '{metric_name}': {exc}")

        method_curves[method] = curves
        for x, _ in curves:
            if len(x):
                all_x.append(x.max())

    if not all_x:
        print(f"No valid data found for metric '{metric_name}'.")
        continue

    x_max = min(all_x)  # fair comparison: shared horizon
    x_grid = np.linspace(0, x_max, 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, curves in method_curves.items():
        if not curves:
            continue
        med, q25, q75 = align_and_summarize(curves, x_grid)
        ax.plot(x_grid, med, color=colors.get(method, None), label=method, linewidth=2)
        ax.fill_between(x_grid, q25, q75, color=colors.get(method, None), alpha=0.25)

    reference_line = metric_cfg.get("reference_line")
    if reference_line is not None:
        ax.axhline(
            reference_line,
            linestyle=(0, (3, 3)),
            color="green",
            alpha=0.8,
            label=metric_cfg.get("reference_label", "reference"),
        )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(metric_cfg["label"])
    ax.grid(alpha=0.3)
    ax.legend()
    plt.title(f"Learning Curve: {metric_cfg['label']}")
    plt.tight_layout()
    plt.savefig(f"{current_save_dir}/{metric_cfg['output']}", dpi=300)
    plt.show()