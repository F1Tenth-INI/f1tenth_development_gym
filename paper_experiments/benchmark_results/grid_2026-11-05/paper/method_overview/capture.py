#!/usr/bin/env python3
"""Run the paper's icra2022 example (soft25w02, raceline, mu 0.5, no noise, seed 1) and save the
planner's internals at every control step with an accepted window update in a range of steps.

    python capture.py [first_step last_step]

Writes ``steps/step_<n>.npz``: car state, driving rollouts and weights, the mean plan, the retrieval
points the step's nominal sequence was read at, the window candidates rolled out from the stored
state behind the car, the stored line before and after the update, and the cross-track lines.
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys

import numpy as np

os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", TQDM_DISABLE="1")
HERE = os.path.dirname(os.path.abspath(__file__))
TUNE = os.path.join(HERE, "..", "..", "..", "tune_noise_2026-11-08")
# CAPTURE_DIR: where steps/ and run/ go (default: this folder). CODE_DIR (read by tune_run.py): the code to run.
DIR = os.environ.get("CAPTURE_DIR", HERE)
OUT = os.path.join(DIR, "steps")
FIRST, LAST = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (2400, 3300)
sys.argv = ["tune_run.py", "soft25w02", "--out", os.path.join(DIR, "run"), "--controller", "hw", "--rollouts", "50",
            "--map", "icra2022", "--wp", "raceline", "--mu", "0.5", "--noise", "off", "--seed", "1",
            "--sim-seconds", str((LAST + 50) * 0.04)]
os.makedirs(OUT, exist_ok=True)


def xy(states):
    from utilities.state_utilities import POSE_X_IDX, POSE_Y_IDX
    return states[..., [POSE_X_IDX, POSE_Y_IDX]].detach().cpu().numpy().astype(np.float32)


def patch(P):
    global torch, WP_TARGET_SPEED_IDX
    import torch
    from TrainingLite.hwm.hirarchical_planner import WP_TARGET_SPEED_IDX
    select_action, weighted, seed, window_update = (
        P.select_action, P._weighted_state_trajectory, P._seed_from_states, P._window_update)

    def _weighted(self, trajectories, weights):
        mean = weighted(self, trajectories, weights)
        self._cap.update(rollouts=xy(trajectories), weights=weights.cpu().numpy(), mean=xy(mean))
        return mean

    def _seed(self, path):
        self._cap["next_retrieval"] = xy(path)
        return seed(self, path)

    def _window(self, window, states):
        before = self._waypoint_xy.clone().numpy()
        accepted = self._wp_window_accepted
        window_update(self, window, states)
        if self._wp_window_accepted > accepted:
            self._cap.update(w=int(window[0]), candidates=xy(states), stored_before=before,
                             stored_after=self._waypoint_xy.clone().numpy())

    def _select(self, state):
        self._cap = {}
        retrieval = getattr(self, "_cap_retrieval", None)
        car_waypoint = self._car_waypoint
        action = select_action(self, state)
        step = int(self.total_timesteps)
        if FIRST <= step <= LAST and "w" in self._cap and retrieval is not None and "rollouts" in self._cap:
            np.savez_compressed(
                os.path.join(OUT, f"step_{step}.npz"), step=step, state=np.asarray(state, dtype=np.float32),
                car_waypoint=-1 if car_waypoint is None else car_waypoint, retrieval=retrieval,
                anchor=self._wp_anchor_xy.numpy(), ortho=self._wp_ortho.numpy(),
                known=self._waypoint_known(torch.arange(self._waypoints_n)).numpy(),
                speed=self.waypoints[:, WP_TARGET_SPEED_IDX].numpy(),
                **{k: v for k, v in self._cap.items() if k != "next_retrieval"})
        self._cap_retrieval = self._cap.get("next_retrieval")
        if step > LAST:
            os._exit(0)
        return action

    P.select_action, P._weighted_state_trajectory, P._seed_from_states, P._window_update = (
        _select, _weighted, _seed, _window)


_reload = importlib.reload


def reload(module):
    module = _reload(module)
    if hasattr(module, "HirarchicalPlanner"):
        patch(module.HirarchicalPlanner)
    return module


importlib.reload = reload
runpy.run_path(os.path.join(TUNE, "tune_run.py"), run_name="__main__")
