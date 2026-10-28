"""Model-free checks on the recordings: steering gain / lock, understeer, grip, drivetrain, ESC.

These do not use the dynamics model, so they are an independent sanity check of the fit.

Run from the repo root:
    python AnalyseData/DynamicsModelFit/steady_state_analysis.py [--recordings DIR] [--wheelbase 0.25]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--recordings", default=os.path.join(ROOT, "AnalyseData", "Rosbags", "2026-10-26"))
ap.add_argument("--wheelbase", type=float, default=0.25)
ap.add_argument("--lag", type=int, default=3, help="steering command lag in samples for quasi-steady analysis")
args = ap.parse_args()
L = args.wheelbase

recs = []
for f in sorted(glob.glob(os.path.join(args.recordings, "**", "*.csv"), recursive=True)):
    if "angular_control_executed" in pd.read_csv(f, comment="#", nrows=0).columns:
        recs.append((os.path.basename(f), pd.read_csv(f, comment="#")))

print("== Signal checks against the scan-matched pose")
for name, d in recs:
    t, th = d.time.values, np.unwrap(d.pose_theta.values)
    vxw, vyw = np.gradient(d.pose_x.values, t), np.gradient(d.pose_y.values, t)
    vx_pose = np.cos(th) * vxw + np.sin(th) * vyw
    m = np.abs(vx_pose) > 0.3
    r_pose = np.gradient(th, t)
    mr = np.abs(r_pose) > 0.5
    print(f"  {name}: linear_vel_x / pose v_x = {np.median(d.linear_vel_x.values[m] / vx_pose[m]):.3f}, "
          f"v_x_ekf / pose v_x = {np.median(d.v_x_ekf.values[m] / vx_pose[m]):.3f}, "
          f"gyro / pose yaw rate = {np.median(d.angular_vel_z.values[mr] / r_pose[mr]):.3f}, "
          f"gyro noise = {np.std(d.angular_vel_z - savgol_filter(d.angular_vel_z, 7, 2)):.3f} rad/s")

print("\n== Drivetrain: rear/front wheel speed ratio vs. acceleration (equal -> locked 4WD)")
for name, d in recs:
    v = d.linear_vel_x.values
    a = savgol_filter(v, 9, 2, deriv=1, delta=0.04)
    ratio = d.wheel_speed_rear_rad_s / d.wheel_speed_front_rad_s
    m = (v > 1) & (np.abs(d.angular_vel_z) < 0.3)
    cells = [f"a in [{lo},{hi}]: {np.median(ratio[m & (a > lo) & (a < hi)]):.4f}"
             for lo, hi in [(-9, -1), (-1, 1), (1, 3)] if np.any(m & (a > lo) & (a < hi))]
    print(f"  {name}: " + ", ".join(cells))

print("\n== Longitudinal: a = k+ u+ + k- u- + a0 + kv v  (u = executed accel command, lag 2 samples)")
for name, d in recs:
    v = d.linear_vel_x.values
    a = savgol_filter(v, 9, 2, deriv=1, delta=0.04)
    u = np.roll(d.translational_control_executed.values, 2)
    m = v > 0.8
    X = np.c_[np.clip(u, 0, None), np.clip(u, None, 0), np.ones_like(v), v][m]
    c, *_ = np.linalg.lstsq(X, a[m], rcond=None)
    print(f"  {name}: k+ {c[0]:.2f}  k- {c[1]:.2f} (n u<-0.5: {(u < -0.5).sum()})  a0 {c[2]:.2f} m/s^2  kv {c[3]:.3f}")

print("\n== Quasi-steady cornering: curvature r/v [1/m] by |steering command| and speed")
rows = []
for name, d in recs:
    r, v = d.angular_vel_z.values, d.linear_vel_x.values
    u = np.roll(d.angular_control_executed.values, args.lag)
    ok = ((uniform_filter1d(np.abs(np.gradient(u)), 10) < 0.01) & (uniform_filter1d(np.abs(np.gradient(r)), 10) < 0.05)
          & (v > 0.8) & (np.abs(u) < 0.45))
    rows.append(np.c_[v[ok], u[ok], r[ok]])
v, u, r = np.concatenate(rows).T
k = r / v
vb = [(0.8, 1.5), (1.5, 2.2), (2.2, 3.0), (3.0, 4.5)]
print("  |cmd|      [kinematic]  " + "  ".join(f"v {a}-{b}".ljust(13) for a, b in vb))
for lo, hi in [(0.02, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.22), (0.22, 0.32)]:
    line = f"  {lo:.2f}-{hi:.2f}  [{np.tan((lo + hi) / 2) / L:.2f}]      "
    for a, b in vb:
        m = (np.abs(u) >= lo) & (np.abs(u) < hi) & (v >= a) & (v < b)
        line += f"  {np.median(np.abs(k[m])):.2f} (n={m.sum():3d})" if m.sum() >= 5 else "  -            "
    print(line)
m = np.abs(u) > 0.02
fit = least_squares(lambda z: z[0] * u[m] / (L * (1 + z[1] * v[m] ** 2)) - k[m], [1.3, 0.0], loss="soft_l1", f_scale=0.1)
print(f"  fit kappa = c*cmd / (L (1 + K v^2)): steering gain c = {fit.x[0]:.2f}, K = {fit.x[1]:.3f} s^2/m^2 "
      f"(curvature at 3 m/s reduced by {100 * (1 - 1 / (1 + 9 * fit.x[1])):.0f}%)")
print("  98th percentile steady |v*r| per speed bin [m/s^2] (grip limit ~ mu*g):",
      [round(float(np.percentile(np.abs(v * r)[(v >= a) & (v < b)], 98)), 2) for a, b in vb])
