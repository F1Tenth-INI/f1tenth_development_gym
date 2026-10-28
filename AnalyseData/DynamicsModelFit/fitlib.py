"""Shared code: load real-car recordings and score the sim car model on them.

Each 1 s window starts from the measured state. The steering servo is pre-rolled from the
previous commands, then the model is driven open-loop with the executed commands
(through the same control delay the sim applies) at the sim timestep of 0.01 s.
"""
import glob
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # for utilities.*
DEFAULT_RECORDINGS = os.path.join(ROOT, "AnalyseData", "Rosbags", "2026-10-26")
DEFAULT_MODEL = os.path.join(ROOT, "sim", "f110_sim", "envs", "dynamic_model_pacejka_jax.py")

DT_REC, DT_SIM = 0.04, 0.01
SUB = int(round(DT_REC / DT_SIM))  # sim steps per recorded sample
H = 25                             # horizon: 25 samples = 1.0 s
PRE = 10                           # servo pre-roll samples (0.4 s)
BLOCK = 250                        # 10 s blocks for the interleaved train / validation split

# Index of each fitted parameter in VehicleParameters.to_np_array()
PARAM_IDX = dict(mu=0, I_z=5, B_f=7, C_f=8, B_r=11, C_r=12, servo_p=15, s_max=17, a_max=21,
                 c_rr=26, curve_resistance_factor=28, brake_multiplier=29,
                 accel_multiplier=30, steering_multiplier=31)
REQUIRED_COLUMNS = ["angular_vel_z", "linear_vel_x", "pose_x", "pose_y", "pose_theta",
                    "angular_control_executed", "translational_control_executed"]


def load_model(path=DEFAULT_MODEL):
    spec = importlib.util.spec_from_file_location("dynamics_model_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_recordings(directory=DEFAULT_RECORDINGS):
    """All recording CSVs below `directory` that contain the required columns."""
    out = {}
    for f in sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)):
        header = pd.read_csv(f, comment="#", nrows=0).columns
        if not all(c in header for c in REQUIRED_COLUMNS):
            continue  # waypoint / map files etc.
        d = pd.read_csv(f, comment="#")
        lidar = [c for c in d.columns if c.startswith("LIDAR_")]
        out[os.path.basename(f)] = dict(
            r=d.angular_vel_z.values, vx=d.linear_vel_x.values,
            x=d.pose_x.values, y=d.pose_y.values, th=np.unwrap(d.pose_theta.values),
            steer=d.angular_control_executed.values, acc=d.translational_control_executed.values,
            wall=(d[lidar].values.min(1) < 0.15) if lidar else np.zeros(len(d), bool))
    return out


def make_windows(rec, delay_s, stride=2, blocks=(0, 1, 2, 3)):
    """Window arrays: initial measurement, per-substep controls, pre-roll steering, targets."""
    n = len(rec["vx"])
    D = int(round(delay_s / DT_SIM))
    W = dict(s0=[], u=[], pre=[], tgt=[])
    for i in range(PRE + SUB, n - H - 1, stride):
        if (i // BLOCK) % 4 not in blocks:
            continue  # interleaved 10 s blocks for train / validation split
        if np.any(np.abs(rec["steer"][i - PRE - SUB:i + H + 1]) > 0.45):
            continue  # recovery manoeuvres with out-of-range servo command
        if np.any(rec["wall"][i - 2:i + H + 1]):
            continue  # wall contact
        # sim step j of the window is at t_i + j*dt; applied command = sample floor((t - delay)/0.04)
        k = i + np.floor((np.arange(H * SUB) - D) / SUB).astype(int)
        kp = i + np.floor((np.arange(-PRE * SUB, 0) - D) / SUB).astype(int)
        W["u"].append(np.stack([rec["steer"][k], rec["acc"][k]], 1))
        W["pre"].append(rec["steer"][kp])
        W["s0"].append([rec["r"][i], rec["vx"][i], rec["th"][i], rec["x"][i], rec["y"][i]])
        sl = slice(i + 1, i + H + 1)
        W["tgt"].append(np.stack([rec["r"][sl], rec["vx"][sl], rec["th"][sl], rec["x"][sl], rec["y"][sl]], 1))
    return {k: jnp.array(np.array(v), dtype=jnp.float32) for k, v in W.items()}


def concat(ws):
    return {k: jnp.concatenate([w[k] for w in ws]) for k in ws[0]}


def build_rollout(mod, ode_model="ODE:ks_pacejka"):
    """vmapped open-loop rollout over windows: (s0, u, pre, params) -> states at the samples."""
    step = mod.car_dynamics_pacejka_jax

    def one(s0, u, pre, p):
        lf, lr = p[1], p[2]
        servo_p, s_min, s_max, sv_min, sv_max, mult = p[15], p[16], p[17], p[18], p[19], p[31]

        # servo pre-roll gives the wheel angle at the window start
        def servo(delta, cmd):
            dd = jnp.clip(servo_p * (mult * cmd - delta), sv_min, sv_max)
            return jnp.clip(delta + DT_SIM * dd, s_min, s_max), None
        delta0, _ = jax.lax.scan(servo, jnp.clip(mult * pre[0], s_min, s_max), pre)

        r, vx, th, x, y = s0
        vy = vx * jnp.tan(delta0) * lr / (lf + lr)  # measured v_y is unreliable -> kinematic guess
        state = jnp.array([r, vx, vy, th, jnp.cos(th), jnp.sin(th), x, y, 0.0, delta0], dtype=jnp.float32)

        def body(s, uu):
            s = step(s, uu, p, DT_SIM, 1, ode_model=ode_model)
            return s, s
        _, traj = jax.lax.scan(body, state, u)
        return traj[SUB - 1::SUB]

    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None)))


def errors(rollout, W, p):
    traj = rollout(W["s0"], W["u"], W["pre"], p)
    t = W["tgt"]
    e_r = traj[..., 0] - t[..., 0]
    e_vx = traj[..., 1] - t[..., 1]
    # model yaw is wrapped; compare heading change since the window start
    psi0 = W["s0"][:, 2:3]
    d_model = jnp.arctan2(jnp.sin(traj[..., 3] - psi0), jnp.cos(traj[..., 3] - psi0))
    d_meas = t[..., 2] - psi0
    e_th = jnp.arctan2(jnp.sin(d_model - d_meas), jnp.cos(d_model - d_meas))
    e_pos = jnp.hypot(traj[..., 6] - t[..., 3], traj[..., 7] - t[..., 4])
    finite = jnp.all(jnp.isfinite(traj), axis=(1, 2))
    return e_r, e_vx, e_th, e_pos, finite


def huber(x, d=1.0):
    a = jnp.abs(x)
    return jnp.where(a < d, 0.5 * x * x, d * (a - 0.5 * d))


def _loss(rollout, W, p):
    e_r, e_vx, e_th, e_pos, finite = errors(rollout, W, p)
    # error scales: 0.3 rad/s, 0.2 m/s, 0.05 rad, 0.1 m
    l = huber(e_r / 0.3) + huber(e_vx / 0.2) + huber(e_th / 0.05) + huber(e_pos / 0.1)
    return jnp.mean(jnp.where(finite[:, None], l, 1e3))


def loss(rollout, W, p):
    return float(_loss(rollout, W, p))


def batch_loss_fn(rollout):
    """Loss for a batch of parameter vectors (population-vectorised)."""
    return jax.jit(jax.vmap(lambda p, W: _loss(rollout, W, p), in_axes=(0, None)))


def report(rollout, W, p, label=""):
    e_r, e_vx, e_th, e_pos, finite = [np.array(e) for e in errors(rollout, W, p)]
    rm = lambda e, k: np.sqrt(np.nanmean(e[finite, k] ** 2))
    a, b = H // 2 - 1, H - 1  # 0.5 s and 1.0 s
    return (f"{label:30s} n={len(finite):5d}  "
            f"r[rad/s] {rm(e_r, a):.3f}/{rm(e_r, b):.3f}  vx[m/s] {rm(e_vx, a):.3f}/{rm(e_vx, b):.3f}  "
            f"yaw[deg] {np.degrees(rm(e_th, a)):.1f}/{np.degrees(rm(e_th, b)):.1f}  "
            f"pos[m] {rm(e_pos, a):.3f}/{rm(e_pos, b):.3f}  (at 0.5s/1.0s)  nonfinite={(~finite).sum()}")


def car_params(yml, overrides=None):
    """VehicleParameters array for `yml` with optional {name: value} overrides (s_max is symmetric)."""
    from utilities.car_files.vehicle_parameters import VehicleParameters
    p = VehicleParameters(yml).to_np_array()
    for n, v in (overrides or {}).items():
        p[PARAM_IDX[n]] = v
    p[16] = -p[17]
    return p
