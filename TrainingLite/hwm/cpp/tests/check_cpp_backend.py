"""Compare the C++ backend (hw_cpp) with the JAX code of the HW planner, and time it.

    python TrainingLite/hwm/cpp/tests/check_cpp_backend.py [--threads 1 2 4 8] [--batch 1050]

Run from f1tenth_development_gym. Rollouts start from random racing states (icra2022 map for the costs).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

from utilities.Settings import Settings


class _Stub:
    from TrainingLite.hwm.hirarchical_planner import HirarchicalPlanner as _H
    _pacejka_batch_step = _H._pacejka_batch_step
    _pacejka_scan_rollout = _H._pacejka_scan_rollout

    def __init__(self):
        self.action_denormalization_array = np.asarray(Settings.HWM_ACTION_DENORM, dtype=np.float32)


def random_inputs(rng, batch, horizon, delay):
    s = np.zeros((batch, 10), np.float32)
    s[:, 1] = rng.uniform(0.0, 8.0, batch)            # vx
    s[:, 2] = rng.normal(0, 0.3, batch)                # vy
    s[:, 0] = rng.normal(0, 1.5, batch)                # yaw rate
    s[:, 3] = rng.uniform(-np.pi, np.pi, batch)        # heading
    s[:, 4], s[:, 5] = np.cos(s[:, 3]), np.sin(s[:, 3])
    s[:, 6] = rng.uniform(-2, 2, batch)
    s[:, 7] = rng.uniform(-2, 2, batch)
    s[:, 9] = rng.uniform(-0.4, 0.4, batch)            # steering
    s[:, 8] = np.arctan(s[:, 2] / np.maximum(s[:, 1], 1e-3))
    s[: batch // 10, 1] = rng.uniform(-0.5, 1.0, batch // 10)  # slow cars: kinematic blend
    a = np.clip(rng.normal(0, 0.5, (batch, horizon, 2)), -1, 1).astype(np.float32)
    p = np.clip(rng.normal(0, 0.5, (batch, delay, 2)), -1, 1).astype(np.float32)
    return s, a, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--batch", type=int, default=1050)
    ap.add_argument("--horizon", type=int, default=int(Settings.MPC_TRAJECTORY_LENGTH))
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    from TrainingLite.hwm.hw_cpp import HwCpp
    from utilities.car_files.vehicle_parameters import VehicleParameters

    dt = float(Settings.TIMESTEP_CONTROL)
    substeps = int(round(dt / float(Settings.TIMESTEP_SIM)))
    delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))
    car = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_np_array()
    cpp = HwCpp(args.threads[0], car, Settings.ODE_MODEL_OF_CAR_DYNAMICS, dt, substeps,
                Settings.HWM_ACTION_DENORM, float(Settings.HWM_MAX_TRANSLATIONAL_CONTROL))

    rng = np.random.default_rng(0)
    s, a, p = random_inputs(rng, args.batch, args.horizon, delay)
    ref = np.asarray(_Stub()._pacejka_scan_rollout()(s, a, p))
    out = cpp.rollout(s, a, p)
    err = np.abs(out - ref)
    names = ["yaw_rate", "vx", "vy", "theta", "cos", "sin", "x", "y", "slip", "steer"]
    err[:, :, 3] = np.abs(np.arctan2(np.sin(out[:, :, 3] - ref[:, :, 3]), np.cos(out[:, :, 3] - ref[:, :, 3])))
    print("rollout |cpp - jax| over", args.batch, "rollouts of", args.horizon, "steps:")
    for i, n in enumerate(names):
        e = err[:, :, i]
        print(f"  {n:9s} max {e.max():.2e}  p99 {np.quantile(e, 0.99):.2e}  at step 1 max {e[:, 1].max():.2e}")
    pos = np.hypot(err[:, -1, 6], err[:, -1, 7])
    print(f"  end position error: median {np.median(pos):.2e} m, p99 {np.quantile(pos, 0.99):.2e} m, max {pos.max():.2e} m")

    # Costs and freeze on the icra2022 map, against drive_costs_jax and the torch freeze.
    from TrainingLite.hwm.shared.wall_geometry import WallGeometry
    from TrainingLite.hwm.drive_costs_jax import drive_costs, squared_distances
    from jax.experimental import enable_x64
    geo = WallGeometry()
    cpp.set_map(geo)
    wp = geo.raceline_abs.numpy()
    start = 40
    count = 30
    width = 32
    idx = (start + np.arange(width)) % len(wp)
    nxt = wp[(idx + 1) % len(wp)]
    heading = np.arctan2(nxt[:, 1] - wp[idx, 1], nxt[:, 0] - wp[idx, 0])
    rows = np.stack([wp[idx, 0], wp[idx, 1], heading, np.full(width, 5.0), np.zeros(width)], 1).astype(np.float32)
    s2 = np.repeat(s[:1], args.batch, 0)
    s2[:, 6], s2[:, 7] = wp[start, 0], wp[start, 1]
    s2[:, 3] = heading[0]
    s2[:, 4], s2[:, 5] = np.cos(heading[0]), np.sin(heading[0])
    s2[:, 1] = 4.0
    s2[:, 0] = s2[:, 2] = s2[:, 9] = 0
    traj = np.asarray(_Stub()._pacejka_scan_rollout()(s2, a, p)).astype(np.float32)
    border = float(Settings.HWM_MIN_BORDER_DISTANCE)
    consts = (1.0, 1.0, 0.6, border, 0.10, 0.05, geo.origin_x, geo.origin_y, geo.origin_cos, geo.origin_sin,
              geo.resolution, float(geo.map_width), float(geo.map_height), float(traj.shape[1]))
    with enable_x64():
        jx = drive_costs(traj, squared_distances(traj, rows[:, :2]), rows, np.int32(count), geo.wall_distance.numpy(),
                         geo._corner_x.numpy(), geo._corner_y.numpy(), consts)
        jx = [np.asarray(x) for x in jx]
    cx = cpp.drive_costs(traj, rows[:, :2], rows, count, consts[:6], traj.shape[1])
    for n, j, c in zip(["drive", "active", "late", "closeness", "depth"], jx, cx):
        d = np.abs(j.astype(np.float64) - c.astype(np.float64))
        print(f"  cost {n:9s} max |cpp - jax| {d.max():.2e} (values up to {np.abs(j).max():.3g}), crashes {int((np.asarray(jx[2]) > 0).sum())}")
    end = 20
    t_ref = torch.from_numpy(traj.copy())
    crashed = geo.in_wall(t_ref[:, :, [6, 7, 3]])
    first = torch.where(crashed.any(1), crashed.float().argmax(1), torch.full((args.batch,), traj.shape[1]))
    late = (first >= end) & (first < traj.shape[1])
    for k in torch.nonzero(late).flatten().tolist():
        f = int(first[k]); fr = t_ref[k, f - 1].clone(); fr[[1, 2, 0]] = 0; t_ref[k, f:] = fr
    t_c = traj.copy()
    cpp.freeze_late(t_c, end, border)
    print(f"  freeze: {int(late.sum())} rollouts frozen, max |cpp - torch| {np.abs(t_c - t_ref.numpy()).max():.2e}")

    # Timing: one control step's drive rollouts plus a window update's, then the costs.
    print(f"\ntiming, {args.batch} rollouts x {args.horizon} steps x {substeps} substeps:")
    for th in args.threads:
        cpp.lib.hw_set_threads(th)
        for _ in range(3):
            cpp.rollout(s, a, p)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            cpp.rollout(s, a, p)
        t_roll = (time.perf_counter() - t0) / args.reps
        t0 = time.perf_counter()
        for _ in range(args.reps):
            cpp.drive_costs(traj, rows[:, :2], rows, count, consts[:6], traj.shape[1])
        t_cost = (time.perf_counter() - t0) / args.reps
        print(f"  {th} threads: rollout {1e3 * t_roll:6.2f} ms, costs {1e3 * t_cost:5.2f} ms")


if __name__ == "__main__":
    main()
