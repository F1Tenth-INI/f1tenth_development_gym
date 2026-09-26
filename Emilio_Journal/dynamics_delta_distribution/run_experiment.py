#!/usr/bin/env python3
"""Held-out diagnostics for p(s_{t+1} - s_t | history, a).

Collects Pure Pursuit transitions in the F1TENTH gym, plots the unconditioned
distribution of state increments, then fits a tiny deterministic MLP on a
train split only and plots held-out prediction errors.

One-step input: (s_t, a_t)
Three-step input: (s_{t-2}, a_{t-2}, s_{t-1}, a_{t-1}, s_t, a_t)

Re-run from this directory, or from anywhere:

    python Emilio_Journal/dynamics_delta_distribution/run_experiment.py

Outputs land in ./outputs next to this file.
"""

from __future__ import annotations

import sys

# Do not write .pyc files into the gym package tree.
sys.dont_write_bytecode = True

import builtins
import copy
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn

# Gym root is two levels up: Emilio_Journal/<this>/ -> f1tenth_development_gym
GYM_ROOT = Path(__file__).resolve().parents[2]
if str(GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(GYM_ROOT))

from utilities.Settings import Settings  # noqa: E402
from utilities.state_utilities import STATE_VARIABLES  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants. Changing these and re-running is the whole protocol.
# ---------------------------------------------------------------------------
SEED = 0
N_CONTROL_STEPS = 6000
CONTROLLER = "pp"
WAYPOINT_VEL_FACTOR = 0.8
TRAIN_FRACTION = 0.7
# Rows within this many steps of the train/test cut are dropped for the
# 3-step model so a test target never sees a train state.
HISTORY = 3

HIDDEN = 32
EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6

STATE_DIM = 10
ACTION_DIM = 2
STATE_NAMES = [str(name) for name in STATE_VARIABLES]
THETA_INDEX = STATE_NAMES.index("pose_theta")

OUT_DIR = Path(__file__).resolve().parent / "outputs"
JOURNAL_DIR = Path(__file__).resolve().parent


def _path_is_allowed(path: Path) -> bool:
    """Writes may land in this journal folder, or outside the gym repo entirely.

    Nothing under the gym except this folder may be created or modified.
    """
    resolved = path.resolve()
    try:
        resolved.relative_to(GYM_ROOT.resolve())
    except ValueError:
        return True
    try:
        resolved.relative_to(JOURNAL_DIR.resolve())
        return True
    except ValueError:
        return False


def install_repo_write_guard() -> None:
    """Refuse any write the simulator might make outside this journal folder."""
    real_open = builtins.open
    real_makedirs = os.makedirs

    def guarded_open(file, mode="r", *args, **kwargs):
        mode_text = mode if isinstance(mode, str) else ""
        if any(flag in mode_text for flag in ("w", "a", "x", "+")):
            if not _path_is_allowed(Path(file)):
                raise RuntimeError(f"refusing to write outside the journal folder: {file}")
        return real_open(file, mode, *args, **kwargs)

    def guarded_makedirs(name, mode=0o777, exist_ok=False):
        if not _path_is_allowed(Path(name)):
            raise RuntimeError(f"refusing to create a directory outside the journal folder: {name}")
        return real_makedirs(name, mode, exist_ok=exist_ok)

    builtins.open = guarded_open
    os.makedirs = guarded_makedirs


def snapshot_settings() -> dict:
    snap = {}
    for attr, value in vars(Settings).items():
        if attr.startswith("_") or isinstance(value, (classmethod, staticmethod, property)):
            continue
        if callable(value):
            continue
        snap[attr] = copy.deepcopy(value)
    return snap


def restore_settings(snap: dict) -> None:
    for attr, value in snap.items():
        setattr(Settings, attr, value)


def configure_sim() -> None:
    """Headless Pure Pursuit, zero delay, no recordings, no opponents.

    CONTROL_DELAY is set to 0 so the action stored with s_t is the action
    integrated over the whole control interval [t, t+1]. The gym default
    delay would apply a previous command instead.
    """
    Settings.CONTROLLER = CONTROLLER
    Settings.GLOBAL_WAYPOINT_VEL_FACTOR = WAYPOINT_VEL_FACTOR
    Settings.RENDER_MODE = None
    Settings.SAVE_RECORDINGS = False
    Settings.SAVE_PLOTS = False
    Settings.SAVE_REWARDS = False
    Settings.SAVE_VIDEOS = False
    Settings.CONTROL_DELAY = 0.0
    Settings.NUMBER_OF_OPPONENTS = 0
    Settings.MAX_SIM_FREQUENCY = None
    Settings.RESET_ON_DONE = True
    Settings.RESPAWN_ON_RESET = False
    Settings.START_FROM_RANDOM_POSITION = True
    Settings.CBF_SAFETY_FILTER = False
    Settings.MPC_SAFETY_FILTER = False
    Settings.NOISE_LEVEL_CAR_STATE = [0.0] * STATE_DIM
    Settings.NOISE_LEVEL_CONTROL = [0.0, 0.0]
    # Let episodes end on crash / leave-track rather than a short time cap,
    # so each episode is a real contiguous trajectory.
    Settings.EXPERIMENT_MAX_LENGTH = 10**9
    Settings.SIMULATION_LENGTH = 10**9
    Settings.MAX_EPISODE_LENGTH = 4096


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def state_delta(state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
    """Componentwise increment. Yaw is wrapped to [-pi, pi].

    pose_theta_sin / pose_theta_cos stay as raw differences. They are not
    angles; wrapping them would hide that they are bounded and redundant
    with yaw.
    """
    delta = next_state - state
    delta[..., THETA_INDEX] = wrap_angle(delta[..., THETA_INDEX])
    return delta


def collect(n_steps: int) -> dict[str, np.ndarray]:
    """Roll out the configured controller and keep in-episode transitions.

    A transition is dropped when the step ends in a reset: the pose after
    reset is not the physical successor of s_t. The action is the command
    the driver produced on that step (steering, acceleration), which with
    CONTROL_DELAY = 0 is what the simulator integrates.
    """
    from run.run_simulation import RacingSimulation

    sim = RacingSimulation()
    sim.prepare_simulation()
    sim.reset()

    states = []
    actions = []
    next_states = []
    episode_ids = []
    episode_id = 0

    for step in range(n_steps):
        agent = sim.world_sim.agents[0]
        driver = sim.drivers[0]
        s = np.asarray(agent.state, dtype=np.float64).copy()
        sim.simulation_step()
        # reset() zeroes control_index at the end of a terminal step.
        if int(driver.control_index) == 0:
            episode_id += 1
            continue
        s_next = np.asarray(agent.state, dtype=np.float64).copy()
        action = np.array(
            [float(driver.angular_control), float(driver.translational_control)],
            dtype=np.float64,
        )
        states.append(s)
        actions.append(action)
        next_states.append(s_next)
        episode_ids.append(episode_id)
        if (step + 1) % 1000 == 0:
            print(f"  collected step {step + 1}/{n_steps}  transitions={len(states)}  episodes={episode_id + 1}")

    if len(states) < 500:
        raise RuntimeError(f"only {len(states)} valid transitions; expected a few thousand")

    return {
        "state": np.stack(states),
        "action": np.stack(actions),
        "next_state": np.stack(next_states),
        "episode_id": np.asarray(episode_ids, dtype=np.int64),
    }


def chronological_split(episode_id: np.ndarray, train_fraction: float) -> np.ndarray:
    """True = train. Last (1 - train_fraction) of each episode is test.

    Splitting inside every episode keeps both sets populated even when the
    car only finishes one or two laps. The cut is by time, not by shuffle,
    so a test window cannot sit in the middle of training context.
    """
    train = np.zeros(len(episode_id), dtype=bool)
    for ep in np.unique(episode_id):
        idx = np.flatnonzero(episode_id == ep)
        cut = int(np.floor(train_fraction * len(idx)))
        cut = min(max(cut, 1), len(idx) - 1)
        train[idx[:cut]] = True
    return train


def windows(episode_id: np.ndarray, train: np.ndarray, history: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices t where t-history+1 ... t lie in one episode and one split."""
    train_idx = []
    test_idx = []
    n = len(episode_id)
    for t in range(history - 1, n):
        sl = slice(t - history + 1, t + 1)
        if np.unique(episode_id[sl]).size != 1:
            continue
        flags = train[sl]
        if flags.all():
            train_idx.append(t)
        elif (~flags).all():
            test_idx.append(t)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


class DeltaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, STATE_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stack_history(state: np.ndarray, action: np.ndarray, idx: np.ndarray, history: int) -> np.ndarray:
    """For each t in idx, concat (s, a) over t-history+1 ... t. Shape (N, history*12)."""
    pieces = []
    for k in range(history - 1, -1, -1):
        pieces.append(state[idx - k])
        pieces.append(action[idx - k])
    return np.concatenate(pieces, axis=1)


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[DeltaMLP, dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1e-8] = 1.0

    xt = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32)
    yt = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32)
    model = DeltaMLP(x_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    n = xt.shape[0]
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            batch = perm[start : start + BATCH_SIZE]
            pred = model(xt[batch])
            loss = loss_fn(pred, yt[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch + 1:3d}/{EPOCHS}  train_mse={total / n:.6f}")
    model.eval()
    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, stats


@torch.no_grad()
def predict(model: DeltaMLP, stats: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xt = torch.tensor((x - stats["x_mean"]) / stats["x_std"], dtype=torch.float32)
    y_hat = model(xt).numpy() * stats["y_std"] + stats["y_mean"]
    return y_hat


def summarize(delta: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": delta.mean(axis=0).tolist(),
        "std": delta.std(axis=0).tolist(),
        "min": delta.min(axis=0).tolist(),
        "max": delta.max(axis=0).tolist(),
    }


def rmse(err: np.ndarray) -> list[float]:
    return np.sqrt((err**2).mean(axis=0)).tolist()


def _hist_page(pdf: PdfPages, values: np.ndarray, title: str, xlabel: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2))
    fig.suptitle(title, fontsize=12)
    for i, ax in enumerate(axes.ravel()):
        col = values[:, i]
        ax.hist(col, bins=60, density=True, color="#3d5a80", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{STATE_NAMES[i]}\nμ={col.mean():.3g}  σ={col.std():.3g}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("density", fontsize=7)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _corr_page(pdf: PdfPages, delta: np.ndarray, title: str) -> None:
    corr = np.corrcoef(delta.T)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(STATE_DIM), STATE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(STATE_DIM), STATE_NAMES, fontsize=8)
    for i in range(STATE_DIM):
        for j in range(STATE_DIM):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    # In-memory only. Settings.py on disk is snapshotted and restored, and the
    # write guard rejects recordings, crash logs, and any other gym-tree write.
    install_repo_write_guard()
    # Numba's default cache directory is sim/f110_sim/envs/__pycache__.
    # Point it here so importing the simulator does not write into the gym tree.
    numba_cache = OUT_DIR / "numba_cache"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    settings_snapshot = snapshot_settings()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    try:
        _run()
    finally:
        restore_settings(settings_snapshot)


def _run() -> None:
    configure_sim()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {N_CONTROL_STEPS} control steps with controller={CONTROLLER}")
    data = collect(N_CONTROL_STEPS)
    delta_all = state_delta(data["state"], data["next_state"])
    train = chronological_split(data["episode_id"], TRAIN_FRACTION)
    test = ~train

    print(
        f"transitions={len(train)}  train={int(train.sum())}  test={int(test.sum())}  "
        f"episodes={len(np.unique(data['episode_id']))}"
    )

    # --- 1-step model: input (s_t, a_t), target Δs. Test rows never enter the fit.
    x1 = np.concatenate([data["state"], data["action"]], axis=1)
    print("Fitting 1-step MLP on train split")
    model1, stats1 = fit_mlp(x1[train], delta_all[train], seed=SEED)
    err1 = delta_all[test] - predict(model1, stats1, x1[test])

    # --- 3-step model: same target, longer history, windows that stay inside one split.
    tr_idx, te_idx = windows(data["episode_id"], train, HISTORY)
    if len(tr_idx) < 200 or len(te_idx) < 200:
        raise RuntimeError(f"not enough 3-step windows (train={len(tr_idx)} test={len(te_idx)})")
    x3_train = stack_history(data["state"], data["action"], tr_idx, HISTORY)
    x3_test = stack_history(data["state"], data["action"], te_idx, HISTORY)
    print(f"Fitting 3-step MLP  train_windows={len(tr_idx)}  test_windows={len(te_idx)}")
    model3, stats3 = fit_mlp(x3_train, delta_all[tr_idx], seed=SEED + 1)
    err3 = delta_all[te_idx] - predict(model3, stats3, x3_test)

    np.savez_compressed(
        OUT_DIR / "dataset.npz",
        state=data["state"],
        action=data["action"],
        next_state=data["next_state"],
        delta=delta_all,
        episode_id=data["episode_id"],
        train_mask=train,
        state_names=np.array(STATE_NAMES),
        action_names=np.array(["angular_control", "translational_control"]),
    )
    torch.save(
        {"state_dict": model1.state_dict(), "stats": stats1, "in_dim": x1.shape[1]},
        OUT_DIR / "mlp_1step.pt",
    )
    torch.save(
        {"state_dict": model3.state_dict(), "stats": stats3, "in_dim": x3_train.shape[1]},
        OUT_DIR / "mlp_3step.pt",
    )

    metrics = {
        "seed": SEED,
        "controller": CONTROLLER,
        "waypoint_vel_factor": WAYPOINT_VEL_FACTOR,
        "control_delay_s": 0.0,
        "timestep_control_s": float(Settings.TIMESTEP_CONTROL),
        "n_control_steps_requested": N_CONTROL_STEPS,
        "n_transitions": int(len(train)),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "n_episodes": int(len(np.unique(data["episode_id"]))),
        "n_train_3step": int(len(tr_idx)),
        "n_test_3step": int(len(te_idx)),
        "train_fraction_per_episode": TRAIN_FRACTION,
        "history": HISTORY,
        "mlp": {"hidden": HIDDEN, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
        "state_names": STATE_NAMES,
        "delta_test": summarize(delta_all[test]),
        "rmse_1step_test": rmse(err1),
        "rmse_3step_test": rmse(err3),
        "error_1step_test": summarize(err1),
        "error_3step_test": summarize(err3),
        "notes": [
            "delta pose_theta is wrapped to [-pi, pi]; other components are raw s_{t+1}-s_t.",
            "Unconditioned histograms and both error histograms use held-out rows only.",
            "1-step test rows are the last 30% of each episode.",
            "3-step test windows lie entirely inside that held-out suffix.",
            "Action is [angular_control, translational_control] with delay 0.",
            "Only the 10-d car state and the 2-d action are used. No lidar, waypoints, or walls.",
        ],
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pdf_path = OUT_DIR / "delta_histograms.pdf"
    with PdfPages(pdf_path) as pdf:
        _hist_page(
            pdf,
            delta_all[test],
            f"Unconditioned Δs on held-out steps (n={int(test.sum())}, {CONTROLLER}, dt={Settings.TIMESTEP_CONTROL}s)",
            "s_{t+1} − s_t",
        )
        _corr_page(
            pdf,
            delta_all[test],
            f"Correlation of Δs components, held-out (n={int(test.sum())})",
        )
        _hist_page(
            pdf,
            err1,
            f"1-step MLP error  Δs − f(s_t, a_t)   held-out n={err1.shape[0]}",
            "prediction error",
        )
        _hist_page(
            pdf,
            err3,
            f"3-step MLP error  Δs − f(s,a over last 3)   held-out n={err3.shape[0]}",
            "prediction error",
        )

    print(f"Wrote {OUT_DIR / 'dataset.npz'}")
    print(f"Wrote {pdf_path}")
    print("1-step test RMSE:", [round(v, 5) for v in metrics["rmse_1step_test"]])
    print("3-step test RMSE:", [round(v, 5) for v in metrics["rmse_3step_test"]])


if __name__ == "__main__":
    main()
