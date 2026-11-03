"""Fit car model parameters to SysId recordings (prediction-error method).

The recording is cut into short overlapping windows (default 1 s). Each window
starts from the measured state and is simulated with the *applied* commands
(recorded ``angular_control`` / ``translational_control``, i.e. after any safety
filter) through the same JAX model the simulator and the MPCs use. The parameters
minimize the error of the predicted yaw rate, v_x and v_y over all windows.
Short windows keep the fit well posed although the drive is closed loop and the
state estimate drifts, and they are what an MPC needs anyway.

The fit is a bounded nonlinear least squares in log-parameter space with
finite-difference Jacobians (the model has stop_gradient on its resistance terms,
so autodiff would not move ``c_rr``). A small prior pulls parameters the data
does not determine back to the nominal file. The control delay is chosen by a
grid search.

    python -m Control_Toolkit_ASF.Controllers.SysId.fit_parameters \\
        --csv ExperimentRecordings/Sysid/<run>.csv \\
        --car-params yokomo1_car_parameters.yml \\
        [--truth sysid_truth_car_parameters.yml]    # simulation test: compare with the known answer
        [--out yokomo1_sysid]                       # writes utilities/car_files/yokomo1_sysid_car_parameters.yml

On the real car v_y comes from the EKF and is poor; use ``--vy-weight 0`` there.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GYM_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
if GYM_ROOT not in sys.path:
    sys.path.insert(0, GYM_ROOT)
os.chdir(GYM_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax  # noqa: E402
from utilities.car_files.vehicle_parameters import VehicleParameters  # noqa: E402
from utilities.Settings import Settings  # noqa: E402
from utilities.state_utilities import STATE_VARIABLES  # noqa: E402

CAR_FILES = os.path.join(GYM_ROOT, "utilities", "car_files")

# name -> (index in VehicleParameters.to_np_array(), log-bound factor)
FIT_PARAMETERS = {
    "mu": (0, 2.0),
    "I_z": (5, 4.0),
    "B_f": (7, 3.0),
    "C_f": (8, 1.6),
    "E_f": (10, None),  # Pacejka curvature (fall-off past the peak); nominal 0, so additive
    "B_r": (11, 3.0),
    "C_r": (12, 1.6),
    "E_r": (14, None),
    "servo_p": (15, 4.0),
    "sv_max": (19, 4.0),  # servo rate limit; sv_min is scaled by the same factor
    "s_max": (17, 2.0),  # s_min is scaled by the same factor (keeps an asymmetric lock)
    "c_rr": (26, 4.0),
    "brake_multiplier": (29, 3.0),
    "accel_multiplier": (30, 3.0),
    "steering_multiplier": (31, 2.0),
}
# A parameter whose nominal value is 0 cannot be fitted in log space: give it bound None to fit it
# additively, p = base + ADDITIVE_SCALE * theta, kept >= 0 (e.g. "steering_diff_low": (25, None)).
ADDITIVE_SCALE = 0.3
ADDITIVE_MAX = 3.0  # theta upper bound: base + 0.9
ADDITIVE_RANGE = {"E_f": (-2.0, 0.9), "E_r": (-2.0, 0.9)}  # absolute ranges where 0 is not the lower limit
YAML_KEYS = {"B_f": ("C_Pf", 0), "C_f": ("C_Pf", 1), "E_f": ("C_Pf", 3),
             "B_r": ("C_Pr", 0), "C_r": ("C_Pr", 1), "E_r": ("C_Pr", 3)}

OUTPUTS = ["angular_vel_z", "linear_vel_x", "linear_vel_y"]
OUTPUT_SCALE = np.array([0.3, 0.15, 0.15])  # [rad/s, m/s, m/s]: errors of this size weigh the same
STATE_COLUMNS = list(STATE_VARIABLES)
I_STEER = STATE_COLUMNS.index("steering_angle")
OUT_IDX = [STATE_COLUMNS.index(c) for c in OUTPUTS]


# ----------------------------------------------------------------------- data
def load_recordings(paths):
    runs = []
    for p in paths:
        d = pd.read_csv(p, comment="#")
        dt = float(np.median(np.diff(d["time"].to_numpy())))
        x = np.column_stack([d[c].to_numpy(dtype=np.float64) for c in STATE_COLUMNS])
        # On the car /driving/execute is what reached the VESC (manual driving, the stop when R1 is
        # released). The planner's command (angular_control) keeps running while the car is not
        # autonomous. The simulator has no executed columns.
        executed = ("angular_control_executed" in d and "translational_control_executed" in d
                    and d["angular_control_executed"].abs().max() > 0)
        cols = (("angular_control_executed", "translational_control_executed") if executed
                else ("angular_control", "translational_control"))
        u = np.column_stack([d[c].to_numpy(dtype=np.float64) for c in cols])
        print(f"  {os.path.basename(p)}: inputs from {cols[0]} / {cols[1]}")
        # Smoothed copy for initial states (the measured outputs stay raw)
        xs = x.copy()
        for c in OUTPUTS:
            i = STATE_COLUMNS.index(c)
            xs[:, i] = savgol_filter(x[:, i], 7, 2)
        runs.append({"path": p, "dt": dt, "x": x, "x_smooth": xs, "u": u, "t": d["time"].to_numpy()})
        if "vesc_voltage_input" in d and (d["vesc_voltage_input"] > 1.0).any():  # real car only
            vv = d["vesc_voltage_input"].to_numpy()
            vv = vv[vv > 1.0]
            print(f"  {os.path.basename(p)}: battery {vv.max():.2f} -> {vv.min():.2f} V (mean {vv.mean():.2f})")
    return runs


def make_windows(runs, horizon, stride, delay, val_block_s=10.0, val_every=5):
    """Window starts, split into train / validation by alternating time blocks."""
    W = {"train": [], "val": []}
    for ri, r in enumerate(runs):
        n = len(r["u"])
        for k0 in range(max(delay, 1) + 1, n - horizon - 1, stride):
            if r["x_smooth"][k0, STATE_COLUMNS.index("linear_vel_x")] < 0.4:
                continue
            block = int(r["t"][k0] // val_block_s)
            split = "val" if block % val_every == val_every - 1 else "train"
            W[split].append((ri, k0))
    return W


def window_arrays(runs, windows, horizon, delay):
    X0, U, Y = [], [], []
    for ri, k0 in windows:
        r = runs[ri]
        x0 = r["x_smooth"][k0].copy()
        x0[STATE_COLUMNS.index("pose_x")] = 0.0
        x0[STATE_COLUMNS.index("pose_y")] = 0.0
        X0.append(x0)
        U.append(r["u"][k0 - delay:k0 - delay + horizon])
        U_prev = r["u"][k0 - delay - 1]
        X0[-1][I_STEER] = U_prev[0]  # replaced by the model's steering below (not measured on the car)
        Y.append(r["x"][k0 + 1:k0 + 1 + horizon][:, OUT_IDX])
    return (jnp.asarray(np.array(X0), dtype=jnp.float32), jnp.asarray(np.array(U), dtype=jnp.float32),
            np.array(Y))


# ---------------------------------------------------------------------- model
def build_simulator(dt, ode_model):
    def rollout(params, x0, U):
        # The steering angle is not measured on the car: start at the steady servo angle of the last command
        delta0 = jnp.clip(x0[I_STEER] * params[31], params[16], params[17])
        x0 = x0.at[I_STEER].set(delta0)

        def step(x, u):
            x = car_dynamics_pacejka_jax(x, u, params, dt, intermediate_steps=4, ode_model=ode_model)
            return x, x[jnp.array(OUT_IDX)]

        _, ys = jax.lax.scan(step, x0, U)
        return ys

    return jax.jit(jax.vmap(rollout, in_axes=(None, 0, 0)))


class Problem:
    def __init__(self, base_params, names, sim, data, vy_weight, prior_weight):
        self.base = np.asarray(base_params, dtype=np.float64)
        self.names = names
        self.sim = sim
        self.X0, self.U, self.Y = data
        self.scale = OUTPUT_SCALE / np.array([1.0, 1.0, max(vy_weight, 1e-9)])
        self.use = np.array([1.0, 1.0, 1.0 if vy_weight > 0 else 0.0])
        self.prior_weight = prior_weight

    def params(self, theta):
        p = self.base.copy()
        for name, th in zip(self.names, theta):
            idx, bound = FIT_PARAMETERS[name]
            p[idx] = self.base[idx] + ADDITIVE_SCALE * th if bound is None else self.base[idx] * np.exp(th)
            if name == "s_max":  # scale both locks by the same factor: keeps a measured asymmetry
                p[16] = self.base[16] * np.exp(th)
            if name == "sv_max":
                p[18] = self.base[18] * np.exp(th)
        return p

    def predict(self, theta, X0=None, U=None):
        X0 = self.X0 if X0 is None else X0
        U = self.U if U is None else U
        return np.asarray(self.sim(jnp.asarray(self.params(theta), dtype=jnp.float32), X0, U), dtype=np.float64)

    def residuals(self, theta):
        e = (self.predict(theta) - self.Y) / self.scale * self.use
        e = np.nan_to_num(e, nan=1e3, posinf=1e3, neginf=-1e3)
        n = e.size
        prior = np.sqrt(self.prior_weight * n) * np.asarray(theta)  # per-parameter pull to nominal
        return np.concatenate([e.ravel() / np.sqrt(n), prior / np.sqrt(n)])

    def rmse(self, theta, X0, U, Y):
        e = self.predict(theta, X0, U) - Y
        return np.sqrt(np.mean(e ** 2, axis=(0, 1)))


# ---------------------------------------------------------------- yaml output
def write_parameter_file(nominal_file, out_file, fitted: dict, delay_s: float, source: str):
    text = open(os.path.join(CAR_FILES, nominal_file)).read()
    lines = text.splitlines()
    for name, value in fitted.items():
        if name in YAML_KEYS:
            block, pos = YAML_KEYS[name]
            start = next(i for i, l in enumerate(lines) if l.startswith(block + ":"))
            row = start + 1 + pos
            lines[row] = f"  - {value:.4g}  # {name}, sysid fit"
            continue
        for key in [name]:
            v = value
            for i, l in enumerate(lines):
                if re.match(rf"^{key}\s*:", l):
                    lines[i] = f"{key}: {v:.4g}  # sysid fit"
                    break
            else:  # optional parameter not in the file yet
                lines.append(f"{key}: {v:.4g}  # sysid fit")
    header = (f"# SysId fit {time.strftime('%Y-%m-%d %H:%M')} from {source}\n"
              f"# Estimated control delay: {delay_s:.2f} s (set Settings.CONTROL_DELAY / CONTROL_DELAY_COMPENSATION_STEPS)\n")
    with open(os.path.join(CAR_FILES, out_file), "w") as f:
        f.write(header + "\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--car-params", default=Settings.CONTROLLER_CAR_PARAMETER_FILE, help="nominal file (start point and prior)")
    ap.add_argument("--truth", default=None, help="simulation test only: the ENV_CAR_PARAMETER_FILE that generated the data")
    ap.add_argument("--out", default=None, help="name for <out>_car_parameters.yml in utilities/car_files")
    ap.add_argument("--params", default=",".join(FIT_PARAMETERS), help="comma separated subset of: " + ",".join(FIT_PARAMETERS))
    ap.add_argument("--horizon", type=float, default=1.0, help="[s] prediction window")
    ap.add_argument("--stride", type=int, default=3, help="[control steps] between window starts")
    ap.add_argument("--delay", default="auto", help="control delay in control steps, or 'auto' (grid 0..5)")
    ap.add_argument("--vy-weight", type=float, default=1.0, help="0 on the real car if v_y is not trustworthy")
    ap.add_argument("--prior", type=float, default=1e-3, help="pull towards the nominal parameters")
    ap.add_argument("--plot", default=None, help="write a validation figure to this png")
    args = ap.parse_args()

    names = [n.strip() for n in args.params.split(",") if n.strip()]
    runs = load_recordings(args.csv)
    dt = runs[0]["dt"]
    H = max(2, int(round(args.horizon / dt)))
    ode_model = Settings.ODE_MODEL_OF_CAR_DYNAMICS
    sim = build_simulator(dt, ode_model)
    nominal = VehicleParameters(args.car_params).to_np_array().astype(np.float64)
    truth = VehicleParameters(args.truth).to_np_array().astype(np.float64) if args.truth else None
    total = sum(len(r["u"]) for r in runs) * dt
    print(f"{len(runs)} recording(s), {total:.0f} s at {1 / dt:.0f} Hz, windows of {H} steps, model {ode_model}")

    def problem_for(delay):
        W = make_windows(runs, H, args.stride, delay)
        train = window_arrays(runs, W["train"], H, delay)
        val = window_arrays(runs, W["val"], H, delay)
        return Problem(nominal, names, sim, train, args.vy_weight, args.prior), val, W

    def bounds(n):
        idx, f = FIT_PARAMETERS[n]
        if f is not None:
            return -np.log(f), np.log(f)
        lo, hi = ADDITIVE_RANGE.get(n, (0.0, nominal[idx] + ADDITIVE_SCALE * ADDITIVE_MAX))
        return (lo - nominal[idx]) / ADDITIVE_SCALE, (hi - nominal[idx]) / ADDITIVE_SCALE

    lb = np.array([bounds(n)[0] for n in names])
    ub = np.array([bounds(n)[1] for n in names])

    def fit(delay, max_nfev):
        prob, val, W = problem_for(delay)
        t0 = time.time()
        if not names:  # evaluation only
            from types import SimpleNamespace
            return prob, val, W, SimpleNamespace(x=np.zeros(0), nfev=0, message="no parameters fitted (evaluation only)"), 0.0
        # The variable is 1 + log(p / p_nominal): scipy scales the finite-difference step with |x|, so x must not be 0
        res = least_squares(lambda z: prob.residuals(z - 1.0), np.ones(len(names)), bounds=(lb + 1.0, ub + 1.0),
                            diff_step=1e-2, x_scale=1.0, max_nfev=max_nfev, verbose=0)
        res.x = res.x - 1.0
        return prob, val, W, res, time.time() - t0

    if args.delay == "auto":
        print("delay search (short fits):")
        scores = {}
        for d in range(0, 6):
            prob, val, W, res, el = fit(d, 15)
            scores[d] = float(np.sum(prob.rmse(res.x, *val) / OUTPUT_SCALE * prob.use))
            print(f"  delay {d} steps ({d * dt:.2f} s): validation score {scores[d]:.3f}  [{el:.0f} s]")
        delay = min(scores, key=scores.get)
    else:
        delay = int(args.delay)

    prob, val, W, res, el = fit(delay, 200)
    print(f"\nfit with delay {delay} steps ({delay * dt:.2f} s): {len(W['train'])} train / {len(W['val'])} validation windows, "
          f"{res.nfev} evaluations, {el:.0f} s, status: {res.message}")

    fitted_params = prob.params(res.x)
    # No per-parameter error bars: B, C, D and mu trade off along flat valleys, so a local Hessian
    # estimate is far too optimistic. Judge the fit by the validation RMSE and the tyre curves below.
    print(f"\n{'parameter':22s} {'nominal':>9s} {'fitted':>9s}" + (f" {'truth':>9s} {'nom err':>8s} {'fit err':>8s}" if truth is not None else ""))
    fitted = {}
    for n in names:
        i = FIT_PARAMETERS[n][0]
        fitted[n] = float(fitted_params[i])
        row = f"{n:22s} {nominal[i]:9.4g} {fitted_params[i]:9.4g}"
        if truth is not None:
            row += (f" {truth[i]:9.4g} {100 * (nominal[i] / truth[i] - 1):+7.1f}% {100 * (fitted_params[i] / truth[i] - 1):+7.1f}%")
        print(row)

    print(f"\nopen-loop prediction RMSE over {H * dt:.1f} s windows, validation data [{', '.join(OUTPUTS)}]:")
    rows = [("nominal", nominal), ("fitted", fitted_params)]
    if truth is not None:
        rows.append(("truth", truth))
    for label, p in rows:
        pr = Problem(p, [], sim, val, args.vy_weight, 0.0)
        print(f"  {label:8s} " + "  ".join(f"{v:.3f}" for v in pr.rmse(np.zeros(0), *val)))

    tire_report(runs, delay, rows, args.plot)

    if args.out:
        out_file = f"{args.out}_car_parameters.yml"
        if "s_max" in fitted:
            fitted["s_min"] = float(fitted_params[16])
        if "sv_max" in fitted:
            fitted["sv_min"] = float(fitted_params[18])
        write_parameter_file(args.car_params, out_file, fitted, delay * dt, ", ".join(os.path.basename(c) for c in args.csv))
        print(f"\nwrote utilities/car_files/{out_file}")

    if args.plot:
        plot_validation(args.plot, sim, val, H, dt, [(l, p) for l, p in rows])
        print(f"wrote {args.plot}")


def slip_angles(runs, delay, p):
    """Front / rear slip angles visited in the recordings (steering from the commands, as on the car)."""
    lf, lr = p[1], p[2]
    out = []
    for r in runs:
        x = r["x_smooth"]
        vx = x[:, STATE_COLUMNS.index("linear_vel_x")]
        vy = x[:, STATE_COLUMNS.index("linear_vel_y")]
        yr = x[:, STATE_COLUMNS.index("angular_vel_z")]
        u = np.roll(r["u"][:, 0], delay)
        delta = np.clip(u * p[31], p[16], p[17])
        ok = vx > 1.5  # below ~1.75 m/s the model is mostly kinematic: no tyre forces
        a_f = delta - np.arctan((vy + lf * yr) / np.maximum(vx, 0.1))
        a_r = -np.arctan((vy - lr * yr) / np.maximum(vx, 0.1))
        out.append(np.column_stack([a_f[ok], a_r[ok]]))
    return np.concatenate(out)


def tire_curve(p, alpha, axle):
    mu = p[0]
    B, C, D, E = (p[7], p[8], p[9], p[10]) if axle == 0 else (p[11], p[12], p[13], p[14])
    Ba = B * alpha
    return mu * D * np.sin(C * np.arctan(Ba - E * (Ba - np.arctan(Ba))))


def tire_report(runs, delay, models, plot_path):
    """Compare the tyre force curves (mu * D * sin(C atan(B alpha))) over the slip range the drive visited.

    B, C, D and mu trade off against each other; what the data determines is the curve in the visited range.
    """
    fitted = dict(models)["fitted"]
    alphas = np.abs(slip_angles(runs, delay, fitted))
    print("\ntyre force / (m g share) over the visited slip range (|alpha| percentiles 50 / 90 / 99):")
    for axle, name in ((0, "front"), (1, "rear")):
        q = np.percentile(alphas[:, axle], [50, 90, 99])
        print(f"  {name}: alpha = " + " / ".join(f"{v:.3f}" for v in q) + " rad")
        for label, p in models:
            print(f"    {label:8s} " + "  ".join(f"{tire_curve(p, a, axle):.3f}" for a in q))
    if not plot_path:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    a = np.linspace(0, 0.5, 200)
    for axle, ax in enumerate(axes):
        for label, p in models:
            ax.plot(a, tire_curve(p, a, axle), label=label)
        ax2 = ax.twinx()
        ax2.hist(alphas[:, axle], bins=60, range=(0, 0.5), color="gray", alpha=0.3)
        ax2.set_yticks([])
        ax.set_title(("front" if axle == 0 else "rear") + " tyre (gray: slip angles visited above 1.5 m/s)")
        ax.set_xlabel("|slip angle| [rad]")
        ax.set_ylabel("lateral force / normal load")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plot_path.replace(".png", "_tyres.png")
    fig.savefig(path, dpi=80)
    print(f"wrote {path}")


def plot_validation(path, sim, val, H, dt, models, n_examples=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X0, U, Y = val
    idx = np.linspace(0, len(Y) - 1, n_examples).astype(int)
    preds = {l: np.asarray(sim(jnp.asarray(p, dtype=jnp.float32), X0[idx], U[idx])) for l, p in models}
    fig, axes = plt.subplots(len(OUTPUTS), n_examples, figsize=(3 * n_examples, 7), sharex=True)
    t = dt * (1 + np.arange(H))
    for j, w in enumerate(idx):
        for i, name in enumerate(OUTPUTS):
            ax = axes[i, j]
            ax.plot(t, Y[w, :, i], "k.", ms=3, label="measured")
            for l, _ in models:
                ax.plot(t, preds[l][j, :, i], label=l)
            if j == 0:
                ax.set_ylabel(name)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("prediction time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=80)


if __name__ == "__main__":
    main()
