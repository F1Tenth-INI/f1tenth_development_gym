"""Fit car parameters of the sim dynamics model to real-car recordings.

Global search (differential evolution) followed by a local Powell refinement, inside
physically plausible bounds. Prints the fitted values, the errors on training and
held-out data and the YAML lines to paste. It does not modify the YAML file.

Run from the repo root:
    python AnalyseData/DynamicsModelFit/fit.py --car yokomo1_car_parameters.yml --delay 0.04
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from fitlib import *  # noqa: E402,F401
from scipy.optimize import differential_evolution, minimize  # noqa: E402

# physically plausible ranges (Pacejka C in [1, 1.8]: no force collapse at large slip)
BOUNDS = dict(mu=(0.5, 1.2), I_z=(0.02, 0.08), s_max=(0.25, 0.6), B_f=(2, 20), C_f=(1.0, 1.8),
              B_r=(2, 25), C_r=(1.0, 1.8), servo_p=(5, 40), c_rr=(0.0, 0.3),
              curve_resistance_factor=(0.0, 1.0), brake_multiplier=(0.5, 10),
              accel_multiplier=(0.5, 2.5), steering_multiplier=(0.9, 1.8))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--car", default="yokomo1_car_parameters.yml", help="start values / fixed params (utilities/car_files)")
ap.add_argument("--recordings", default=DEFAULT_RECORDINGS, help="folder searched recursively for recording CSVs")
ap.add_argument("--test-file", default="2026-10-26_09-31-34_.csv", help="recording kept out of the fit entirely")
ap.add_argument("--delay", type=float, default=0.04, help="control delay [s] applied to the executed commands")
ap.add_argument("--maxiter", type=int, default=60, help="differential evolution generations")
ap.add_argument("--model", default=DEFAULT_MODEL, help="dynamics model file to fit")
ap.add_argument("--out", default=None, help="write the result as JSON here")
args = ap.parse_args()

roll = build_rollout(load_model(args.model))
batch_loss = batch_loss_fn(roll)
p0 = car_params(args.car)
recs = load_recordings(args.recordings)
fit_files = [k for k in recs if k != args.test_file]
print("fit recordings:", fit_files, "| test recording:", args.test_file if args.test_file in recs else "-")
W_train = concat([make_windows(recs[k], args.delay, stride=2, blocks=(0, 1, 2)) for k in fit_files])
W_val = concat([make_windows(recs[k], args.delay, blocks=(3,)) for k in fit_files])
W_test = make_windows(recs[args.test_file], args.delay) if args.test_file in recs else None

names = list(BOUNDS)
idx = np.array([PARAM_IDX[n] for n in names])


def to_P(Z):  # (S, n_fitted) -> (S, 32)
    P = np.repeat(p0[None], len(Z), 0)
    P[:, idx] = Z
    P[:, 16] = -P[:, 17]
    return jnp.array(P, dtype=jnp.float32)


def f_vec(X):  # scipy vectorised call: X (n_fitted, S)
    Z = np.atleast_2d(X.T)
    return np.concatenate([np.array(batch_loss(to_P(Z[k:k + 16]), W_train)) for k in range(0, len(Z), 16)])


z_start = np.array([np.clip(p0[PARAM_IDX[n]], *BOUNDS[n]) for n in names])
print(f"train windows {len(W_train['s0'])}, start loss {f_vec(z_start[:, None])[0]:.4f}", flush=True)
t = time.time()
res = differential_evolution(f_vec, [BOUNDS[n] for n in names], vectorized=True, updating="deferred",
                             popsize=8, maxiter=args.maxiter, tol=1e-6, seed=0, polish=False)
print(f"differential evolution: loss {res.fun:.4f} ({time.time() - t:.0f}s)", flush=True)
res2 = minimize(lambda z: float(f_vec(z[:, None])[0]), res.x, method="Powell",
                bounds=[BOUNDS[n] for n in names], options=dict(maxfev=800, xtol=1e-4, ftol=1e-6))
z = res2.x if res2.fun < res.fun else res.x
print(f"after Powell refinement: loss {min(res.fun, res2.fun):.4f}")

fitted = {n: float(v) for n, v in zip(names, z)}
at_bound = [n for n, v in fitted.items() if min(v - BOUNDS[n][0], BOUNDS[n][1] - v) < 1e-3 * (BOUNDS[n][1] - BOUNDS[n][0])]
print("parameters at a bound (check if physically sensible):", at_bound or "none")
p = to_P(z[None])[0]
print(report(roll, W_train, p, "train"))
print(report(roll, W_val, p, "validation (held-out blocks)"))
if W_test is not None:
    print(report(roll, W_test, p, "test (" + args.test_file + ")"))

print("\nYAML values:")
for n, v in fitted.items():
    if n in ("B_f", "C_f", "B_r", "C_r"):
        continue
    print(f"  {n}: {v:.4g}" + (f"\n  s_min: {-v:.4g}" if n == "s_max" else ""))
print(f"  C_Pf: [{fitted['B_f']:.4g}, {fitted['C_f']:.4g}, {p0[9]:.4g}, {p0[10]:.4g}]")
print(f"  C_Pr: [{fitted['B_r']:.4g}, {fitted['C_r']:.4g}, {p0[13]:.4g}, {p0[14]:.4g}]")
if args.out:
    json.dump(dict(delay=args.delay, car=args.car, params=fitted, loss=float(min(res.fun, res2.fun))),
              open(args.out, "w"), indent=2)
