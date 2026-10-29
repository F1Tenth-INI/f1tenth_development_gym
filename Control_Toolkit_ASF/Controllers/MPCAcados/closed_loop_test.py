#!/usr/bin/env python3
"""Headless closed-loop test of the safe MPC on the simulator dynamics (with control delay).

Usage::

    JAX_PLATFORMS=cpu python -m Control_Toolkit_ASF.Controllers.MPCAcados.closed_loop_test --steps 1500
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from utilities.Settings import Settings
from utilities.state_utilities import create_car_state
from utilities.waypoint_utils import WP_D_LEFT_IDX, WP_D_RIGHT_IDX, WP_GLOBID_IDX

from .mpc_acados_planner import SafeMPCPlanner
from .plant import border_clearance, gym_to_frenet


def load_waypoints(map_name: str) -> np.ndarray:
    raw = np.genfromtxt(f"utilities/maps/{map_name}/{map_name}_wp.csv", delimiter=",", skip_header=1)
    wps = np.zeros((len(raw), 10))
    wps[:, :7] = raw[:, :7]
    wps[:, WP_GLOBID_IDX] = np.arange(len(raw))
    wps[:, WP_D_RIGHT_IDX] = raw[:, 7]
    wps[:, WP_D_LEFT_IDX] = raw[:, 8]
    return wps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--map", default=Settings.MAP_NAME)
    args = ap.parse_args()

    wps = load_waypoints(args.map)
    planner = SafeMPCPlanner()
    planner._ensure_track(wps)
    track = planner.track

    i0 = args.start_index % track.n_points
    x0, y0 = track.xy[i0]
    state = create_car_state({"pose_x": x0, "pose_y": y0, "pose_theta": track.tangent[i0]}).astype(np.float64)
    sim_buffer = deque([np.zeros(2)] * planner.delay_steps)

    s_hint, s_travel, prev_s = None, 0.0, None
    min_clear, n_out = np.inf, 0
    t_wall = time.time()
    for k in range(args.steps):
        u = np.array(planner.process_observation({"car_state": state, "waypoints": wps}))
        for _ in range(planner.n_sub):
            sim_buffer.append(u)
            state = planner.plant.rollout(state, sim_buffer.popleft()[None], n_sub=1)[-1].astype(np.float64)
        s, n, mu = gym_to_frenet(track, state, s_hint)
        s_hint = float(s[0])
        clear = float(border_clearance(track, planner.veh, s, n, mu)[0])
        min_clear = min(min_clear, clear)
        n_out += clear < 0.0
        if prev_s is not None:
            s_travel += (s_hint - prev_s + 0.5 * track.length) % track.length - 0.5 * track.length
        prev_s = s_hint

    sim_time = args.steps * planner.dt
    st = planner.stats
    print("\n=== closed-loop summary ===")
    print(f"simulated {sim_time:.1f}s in {time.time() - t_wall:.1f}s wall, distance {s_travel:.1f} m "
          f"({s_travel / track.length:.2f} laps), mean speed {s_travel / sim_time:.2f} m/s")
    print(f"min corner clearance to border {min_clear:.3f} m, steps with a corner outside: {n_out}")
    print(f"applied: solver output {st['solver']}, shifted previous solution {st['shifted']}; "
          f"mean acados time {1000 * st['solve_time'] / max(st['steps'], 1):.1f} ms, "
          f"mean planner step {1000 * st['step_time'] / max(st['steps'], 1):.1f} ms")


if __name__ == "__main__":
    main()
