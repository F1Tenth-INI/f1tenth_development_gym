# tests/inverse_dynamics/eval_utils.py
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from tests.inverse_dynamics import config as CFG
from utilities.InverseDynamics import (
    CarInverseDynamicsFast,
    CarInverseDynamics,
    CarInverseDynamicsHybrid,
    HARD_CODED_STATE_STD,
)

# ------- Data loading / synthesis -------

REQUIRED_STATE_COLS = [
    "angular_vel_z","linear_vel_x","linear_vel_y","pose_theta","pose_x","pose_y","slip_angle","steering_angle"
]
REQUIRED_CONTROL_COLS = ["angular_control_calculated","translational_control_calculated"]

def synthesize_one_csv(path: Path, N: int, dt: float, mu: float) -> Path:
    """Create a small, consistent CSV with the required columns so tests always run."""
    t = np.arange(N, dtype=np.float64) * dt
    # Simple constant-accel, constant-yaw-rate motion
    psi0 = 3.0
    omega = 0.05  # rad/s
    a = 0.2      # m/s^2
    vx0 = 5.0

    psi = psi0 + omega * t
    vx  = vx0 + a * t
    vy  = np.zeros_like(vx)

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    for k in range(1, N):
        x[k] = x[k-1] + vx[k-1]*np.cos(psi[k-1]) * dt
        y[k] = y[k-1] + vx[k-1]*np.sin(psi[k-1]) * dt

    # Controls are "desired steering angle" and "translational control" (accel)
    desired_angle = np.clip(0.1*np.sin(0.1*t), -0.3, 0.3).astype(np.float32)
    translational = np.full_like(desired_angle, a, dtype=np.float32)

    df = pd.DataFrame({
        "time": t,
        "angular_vel_z": np.full_like(t, omega, dtype=np.float32),
        "linear_vel_x": vx.astype(np.float32),
        "linear_vel_y": vy.astype(np.float32),
        "pose_theta": psi.astype(np.float32),
        "pose_theta_cos": np.cos(psi).astype(np.float32),
        "pose_theta_sin": np.sin(psi).astype(np.float32),
        "pose_x": x.astype(np.float32),
        "pose_y": y.astype(np.float32),
        "slip_angle": np.zeros_like(t, dtype=np.float32),
        "steering_angle": desired_angle,
        "angular_control_calculated": desired_angle,
        "translational_control_calculated": translational,
        "mu": np.full_like(t, mu, dtype=np.float32),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

def gather_csv_files(folder: Optional[str], max_files: Optional[int]=None, all_files: bool=False) -> List[Path]:
    p = Path(folder) if folder else None
    if p and p.exists():
        files = sorted(list(p.glob("*.csv")))
        if files:
            if not all_files and max_files is not None and len(files) > max_files:
                files = files[:max_files]
            return files
    # Fallback: synthesize one file
    synth_path = Path("tests/_synth_data/synth.csv")
    return [synthesize_one_csv(synth_path, N=CFG.SYNTH_N, dt=CFG.SYNTH_DT, mu=CFG.SYNTH_MU)]

def _ensure_cols(df: pd.DataFrame, needed: List[str]):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def load_states_controls(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    df = pd.read_csv(path, comment="#")
    _ensure_cols(df, REQUIRED_STATE_COLS + REQUIRED_CONTROL_COLS)

    theta = df["pose_theta"].to_numpy(np.float32)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # Build [ωz, vx, vy, ψ, sinψ, cosψ, x, y, β, δ]
    states = np.stack([
        df["angular_vel_z"].to_numpy(np.float32),
        df["linear_vel_x"].to_numpy(np.float32),
        df["linear_vel_y"].to_numpy(np.float32),
        theta.astype(np.float32),
        sin_th.astype(np.float32),
        cos_th.astype(np.float32),
        df["pose_x"].to_numpy(np.float32),
        df["pose_y"].to_numpy(np.float32),
        df["slip_angle"].to_numpy(np.float32),
        df["steering_angle"].to_numpy(np.float32),
    ], axis=1)

    # Controls [desired_angle, translational_control]
    Q = np.stack([
        df["angular_control_calculated"].to_numpy(np.float32),
        df["translational_control_calculated"].to_numpy(np.float32),
    ], axis=1)

    # derive dt and mu if available
    info: Dict[str, float] = {}
    if "time" in df.columns and len(df["time"]) >= 2:
        t = df["time"].to_numpy(np.float64)
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0: dt = 0.01
        info["dt"] = dt
    else:
        info["dt"] = 0.01

    if "mu" in df.columns:
        mu_vals = df["mu"].to_numpy(np.float32)
        info["mu"] = float(np.median(mu_vals))
    else:
        info["mu"] = None

    return states.astype(np.float32), Q.astype(np.float32), info

# ------- Errors / metrics -------

POSE_THETA_IDX = 3
POSE_THETA_SIN_IDX = 4
POSE_THETA_COS_IDX = 5

def _wrap_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))

def _fix_sin_cos_np(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    th = x[..., POSE_THETA_IDX]
    x[..., POSE_THETA_SIN_IDX] = np.sin(th)
    x[..., POSE_THETA_COS_IDX] = np.cos(th)
    return x

def scaled_state_errors(pred_states_newest_first: np.ndarray,
                        gt_states_newest_first: np.ndarray,
                        scale: np.ndarray,
                        drop_sincos: bool=True) -> Dict[str, np.ndarray]:
    """Return per-channel MAE and RMSE in *scaled* units. Excludes the pinned newest state (index 0)."""
    assert pred_states_newest_first.shape == gt_states_newest_first.shape
    # Exclude pinned newest state
    P = _fix_sin_cos_np(pred_states_newest_first[1:].copy())
    G = _fix_sin_cos_np(gt_states_newest_first[1:].copy())

    diff = P - G
    # Wrap heading
    diff[:, POSE_THETA_IDX] = _wrap_angle_diff(P[:, POSE_THETA_IDX], G[:, POSE_THETA_IDX])
    # Optionally drop sin/cos residuals
    if drop_sincos:
        diff[:, [POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX]] = 0.0

    z = diff / scale[None, :]
    mae = np.mean(np.abs(z), axis=0)
    rmse = np.sqrt(np.mean(z**2, axis=0))
    return {"mae": mae, "rmse": rmse}

# ------- Evaluation harness -------

def eval_fast(x_T: np.ndarray,
              Q_old_to_new: np.ndarray,
              gt_newest_first: np.ndarray,
              init_type: str,
              noise_scale: float,
              dt: float,
              mu: Optional[float]) -> Dict[str, float]:
    # Build initializer
    if init_type == "none":
        x_init_seq = None
    else:
        x_init_seq = gt_newest_first[1:].copy()
        if init_type == "noisy":
            rng = np.random.default_rng(1234)
            noise = rng.normal(0.0, 1.0, size=x_init_seq.shape).astype(np.float32)
            x_init_seq = x_init_seq + noise_scale * HARD_CODED_STATE_STD[None, :] * noise
        elif init_type == "gt":
            pass
        else:
            raise ValueError("init_type must be one of: none, gt, noisy")

    solver = CarInverseDynamicsFast(mu=mu, controls_are_pid=True, dt=dt, controls_time_order="old_to_new")
    t0 = time.perf_counter()
    states_pred, conv_flags = solver.inverse_entire_trajectory(x_T, Q_old_to_new, x_init_sequence=x_init_seq)
    dt_sec = time.perf_counter() - t0
    metrics = scaled_state_errors(states_pred, gt_newest_first, HARD_CODED_STATE_STD)
    out = {
        "conv_rate": float(np.mean(conv_flags.astype(np.float32))),
        "time_s": float(dt_sec),
        "mae_mean": float(np.mean(metrics["mae"])),
        "rmse_mean": float(np.mean(metrics["rmse"])),
    }
    return out

def eval_refine(x_T: np.ndarray,
                Q_old_to_new: np.ndarray,
                gt_newest_first: np.ndarray,
                init_type: str,
                noise_scale: float,
                dt: float,
                mu: Optional[float]) -> Dict[str, float]:
    T = Q_old_to_new.shape[0]
    # Build initializer (X_init for refine is newest-first without pinned latest)
    if init_type == "none":
        X_init = np.repeat(x_T, repeats=T, axis=0)
    else:
        X_init = gt_newest_first[1:].copy()
        if init_type == "noisy":
            rng = np.random.default_rng(1234)
            noise = rng.normal(0.0, 1.0, size=X_init.shape).astype(np.float32)
            X_init = X_init + noise_scale * HARD_CODED_STATE_STD[None, :] * noise
        elif init_type == "gt":
            pass
        else:
            raise ValueError("init_type must be one of: none, gt, noisy")

    solver = CarInverseDynamics(mu=mu, controls_are_pid=True, dt=dt)
    t0 = time.perf_counter()
    states_pred, conv_flags = solver.inverse_entire_trajectory(x_T, Q_old_to_new, X_init=X_init)
    dt_sec = time.perf_counter() - t0
    metrics = scaled_state_errors(states_pred, gt_newest_first, HARD_CODED_STATE_STD)
    out = {
        "conv_rate": float(np.mean(conv_flags.astype(np.float32))),
        "time_s": float(dt_sec),
        "mae_mean": float(np.mean(metrics["mae"])),
        "rmse_mean": float(np.mean(metrics["rmse"])),
    }
    return out

def eval_hybrid(x_T: np.ndarray,
                Q_old_to_new: np.ndarray,
                gt_newest_first: np.ndarray,
                dt: float,
                mu: Optional[float]) -> Dict[str, float]:
    solver = CarInverseDynamicsHybrid(mu=mu, controls_are_pid=True, dt=dt, controls_time_order="old_to_new")
    t0 = time.perf_counter()
    states_pred, conv_flags = solver.inverse_entire_trajectory(x_T, Q_old_to_new)
    dt_sec = time.perf_counter() - t0
    metrics = scaled_state_errors(states_pred, gt_newest_first, HARD_CODED_STATE_STD)
    out = {
        "conv_rate": float(np.mean(conv_flags.astype(np.float32))),
        "time_s": float(dt_sec),
        "mae_mean": float(np.mean(metrics["mae"])),
        "rmse_mean": float(np.mean(metrics["rmse"])),
    }
    return out
