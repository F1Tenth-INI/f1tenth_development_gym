#!/usr/bin/env python3
"""Commands of the frozen HW planner and of MPPI at fixed car states on icra2022.

Probe states: a grid over the drivable area, the car at rest, heading along the nearest
waypoint of the frozen HW table (nearest one in line of sight, not one behind a wall).

    python probe_policies.py states                  # from anywhere: writes probes/states.npz
    cd ../code && python ../thesis_icra2022/probe_policies.py hw    # re-runs hw50 seed 4 to the end, then probes
    cd ../code && python ../thesis_icra2022/probe_policies.py mppi  # mppi 2000, GPU

Each controller is reset per state and called ``ITERATIONS`` times at that state with
nothing in the control-delay queue. Between the calls the one-step shift of its warm
start is undone (the car does not move), so it refines the plan for this state.
The last command is stored.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.dirname(HERE)
CODE = os.path.join(GRID, "code")
MAP_DIR = os.path.join(CODE, "utilities", "maps", "icra2022")
FROZEN = os.path.join(GRID, "runs", "hw50_icra2022_raceline_mu0.5_noiseon_seed4", "final_raceline.npz")
PROBES = os.path.join(HERE, "probes")
SPACING = 0.3       # [m] grid spacing
CLEARANCE = 0.25    # [m] minimum distance of the car centre to a wall
ITERATIONS = 50
SEED = 4
# Smoke tests: PROBE_LIMIT=20 PROBE_SIM_SECONDS=300 (the HW freeze is at 276 s)
LIMIT = int(os.environ.get("PROBE_LIMIT", "0")) or None
SIM_SECONDS = os.environ.get("PROBE_SIM_SECONDS", "1000")


def free_space():
    import yaml
    from PIL import Image

    with open(os.path.join(MAP_DIR, "icra2022.yaml")) as f:
        meta = yaml.safe_load(f)
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    img = np.asarray(Image.open(os.path.join(MAP_DIR, "icra2022.png")).convert("L"), dtype=np.float64)
    return img > 128.0, res, ox, oy


def make_states():
    from scipy.ndimage import distance_transform_edt, label

    free, res, ox, oy = free_space()
    h, w = free.shape
    clearance = distance_transform_edt(free) * res

    def to_px(x, y):
        col = np.floor((x - ox) / res).astype(int)
        row = (h - 1 - np.floor((y - oy) / res)).astype(int)
        return np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)

    table = np.load(FROZEN)
    wxy, wheading, wspeed = table["xy"], table["heading"], table["speed"]
    # The track is the free component that holds the raceline.
    comp, _ = label(free)
    r, c = to_px(wxy[:, 0], wxy[:, 1])
    track = comp == np.bincount(comp[r, c]).argmax()

    x0, y0 = wxy.min(axis=0) - 2.0
    x1, y1 = wxy.max(axis=0) + 2.0
    gx, gy = np.meshgrid(np.arange(x0, x1, SPACING), np.arange(y0, y1, SPACING))
    gx, gy = gx.ravel(), gy.ravel()
    r, c = to_px(gx, gy)
    keep = track[r, c] & (clearance[r, c] >= CLEARANCE)
    gx, gy = gx[keep], gy[keep]

    def line_of_sight(p, q, n=40):
        t = np.linspace(0.0, 1.0, n)[:, None]
        pts = p[None] + t * (q - p)[None]
        rr, cc = to_px(pts[:, 0], pts[:, 1])
        return bool(free[rr, cc].all())

    idx = np.empty(len(gx), dtype=int)
    for i, (x, y) in enumerate(zip(gx, gy)):
        p = np.array([x, y])
        order = np.argsort(np.hypot(*(wxy - p).T))
        idx[i] = next((j for j in order[:20] if line_of_sight(p, wxy[j])), order[0])
    os.makedirs(PROBES, exist_ok=True)
    np.savez_compressed(os.path.join(PROBES, "states.npz"), x=gx, y=gy, theta=wheading[idx], wp=idx,
                        v_wp=wspeed[idx])
    print(f"{len(gx)} probe states")


def state_vector(x, y, theta, v=0.0):
    from utilities.state_utilities import STATE_INDICES

    s = np.zeros(len(STATE_INDICES), dtype=np.float32)
    s[STATE_INDICES["pose_x"]] = x
    s[STATE_INDICES["pose_y"]] = y
    s[STATE_INDICES["pose_theta"]] = theta
    s[STATE_INDICES["pose_theta_cos"]] = np.cos(theta)
    s[STATE_INDICES["pose_theta_sin"]] = np.sin(theta)
    s[STATE_INDICES["linear_vel_x"]] = v
    return s


def bench_args(controller, rollouts, gpu):
    import argparse

    return argparse.Namespace(out=os.path.join(PROBES, controller), controller=controller, rollouts=rollouts,
                              map="icra2022", wp="raceline", mu=0.5, noise="on", seed=SEED, sim_seconds=1000.0,
                              gpu=gpu, initial_svg_only=False)


def save(name, commands, extra=None):
    out = os.path.join(PROBES, f"{name}.npz")
    np.savez_compressed(out, steer=commands[:, 0], accel=commands[:, 1], **(extra or {}))
    print(f"wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# HW: re-run the benchmark run to its end, then probe the frozen planner
# ---------------------------------------------------------------------------

def probe_hw(planner):
    import torch
    from utilities.Settings import Settings

    st = {k: v[:LIMIT] for k, v in np.load(os.path.join(PROBES, "states.npz")).items()}
    assert planner._permanently_frozen(), "waypoints are not frozen"
    table_before = {k: getattr(planner, k).clone() for k in ("waypoints", "_wp_actions", "_wp_action_known")}
    scalars = {k: v for k, v in vars(planner).items() if isinstance(v, (bool, int, float)) or v is None}
    delay = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_CONTROL))

    original_rollout = planner.rollout_pacejka

    def rollout_pacejka(state, actions, use_control_delay=True, pending_actions=None):
        # The car stands still: no commands in the delay queue.
        if pending_actions is None and use_control_delay:
            batch = 1 if np.ndim(actions) == 2 else int(np.shape(actions)[0])
            pending_actions = np.zeros((batch, delay, 2), dtype=np.float32)
        return original_rollout(state, actions, use_control_delay, pending_actions)

    planner.rollout_pacejka = rollout_pacejka
    last = {}
    original_select = planner.select_action

    def select_action(obs, state):
        a = original_select(obs, state)
        last["action"] = torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(1, 2))
        return a

    planner.select_action = select_action

    out = np.zeros((len(st["x"]), 2), dtype=np.float32)
    recovery = np.zeros(len(st["x"]), dtype=bool)
    backtrack = np.zeros(len(st["x"]), dtype=bool)
    t0 = time.time()
    for i in range(len(st["x"])):
        for k, v in scalars.items():
            setattr(planner, k, v)
        planner.reset()
        torch.manual_seed(i)
        s = state_vector(st["x"][i], st["y"][i], st["theta"][i])
        for _ in range(ITERATIONS):
            planner._mean_state_seed = None
            cmd = planner.process_observation({"car_state": s})
            planner.action_buffer = torch.cat((last["action"], planner.action_buffer[:-1]), dim=0)
        out[i] = cmd
        recovery[i] = bool(planner.recover_mode)
        backtrack[i] = planner._stall_phase == 1
        if i % 100 == 0:
            print(f"hw probe {i}/{len(out)} {time.time() - t0:.0f}s", flush=True)
    for k, v in table_before.items():
        assert torch.equal(v, getattr(planner, k)), f"{k} changed while probing"
    save("hw50", out, {"recovery": recovery, "backtrack": backtrack})


def run_hw():
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "benchmark"))
    import bench_run

    captured = {}
    original_configure = bench_run.configure_settings

    def configure_settings(args):
        Settings = original_configure(args)
        from utilities.car_system import CarSystem

        original_init = CarSystem.initialize_controller

        def initialize_controller(self, name):
            original_init(self, name)
            captured.setdefault("planner", self.planner)

        CarSystem.initialize_controller = initialize_controller
        real_exit = os._exit

        def _exit(code):
            try:
                probe_hw(captured["planner"])
            finally:
                sys.stdout.flush()
                real_exit(code)

        os._exit = _exit
        return Settings

    bench_run.configure_settings = configure_settings
    sys.argv = [sys.argv[0], "--out", os.path.join(PROBES, "hw50_run"), "--controller", "hw", "--rollouts", "50",
                "--map", "icra2022", "--wp", "raceline", "--mu", "0.5", "--noise", "on", "--seed", str(SEED),
                "--sim-seconds", SIM_SECONDS]
    os.environ["JAX_PLATFORMS"] = "cpu"
    bench_run.main()


# ---------------------------------------------------------------------------
# MPPI: no simulation needed, the planner has no state besides its warm start
# ---------------------------------------------------------------------------

def run_mppi():
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "benchmark"))
    import bench_run

    Settings = bench_run.configure_settings(bench_args("mppi", 2000, gpu=True))
    import jax
    from utilities.planner_factory import initialize_planner
    from utilities.waypoint_utils import WaypointUtils

    planner = initialize_planner(Settings.CONTROLLER)
    planner.batch_size = 2000
    planner.key = jax.random.PRNGKey(SEED)
    wpu = WaypointUtils()
    map_xy = np.asarray(wpu.waypoints)[:, [1, 2]]
    st = {k: v[:LIMIT] for k, v in np.load(os.path.join(PROBES, "states.npz")).items()}
    free, res, ox, oy = free_space()
    h = free.shape[0]

    def visible(p, q, n=40):
        pts = p[None] + np.linspace(0, 1, n)[:, None] * (q - p)[None]
        col = np.floor((pts[:, 0] - ox) / res).astype(int)
        row = (h - 1 - np.floor((pts[:, 1] - oy) / res)).astype(int)
        return bool(free[row, col].all())

    out = np.zeros((len(st["x"]), 2), dtype=np.float32)
    t0 = time.time()
    for i in range(len(st["x"])):
        p = np.array([st["x"][i], st["y"][i]])
        order = np.argsort(np.hypot(*(map_xy - p).T))
        wpu.nearest_waypoint_index = int(next((j for j in order[:20] if visible(p, map_xy[j])), order[0]))
        s = state_vector(st["x"][i], st["y"][i], st["theta"][i])
        wpu.update_next_waypoints(s)
        obs = {"car_state": s, "next_waypoints": np.asarray(wpu.next_waypoints, dtype=np.float32)}
        planner.reset()
        for _ in range(ITERATIONS):
            planner.sent_controls[...] = 0.0
            cmd = planner.process_observation(obs)
            q = planner.last_Q_sq
            planner.last_Q_sq = np.concatenate([q[:1], q[:-1]], axis=0)
        out[i] = cmd
        if i % 200 == 0:
            print(f"mppi probe {i}/{len(out)} {time.time() - t0:.0f}s", flush=True)
    save("mppi2000", out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "states":
        make_states()
    elif mode == "hw":
        run_hw()
    elif mode == "mppi":
        run_mppi()
    else:
        raise SystemExit(f"unknown mode {mode}")
