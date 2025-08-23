# HistoryForger.py

import os
import numpy as np

from utilities.waypoint_utils import get_nearest_waypoint
from utilities.InverseDynamics import ProgressiveWindowRefiner
from utilities.Settings import Settings

HISTORY_LENGTH = 20  # in 'controller updates'
TIMESTEP_CONTROL = float(Settings.TIMESTEP_CONTROL)
TIMESTEP_ENVIRONMENT = float(Settings.TIMESTEP_SIM)
timesteps_per_controller_update = max(
    1, int(round(TIMESTEP_CONTROL / TIMESTEP_ENVIRONMENT))
)

START_AFTER_X_STEPS = 100

WINDOW_SIZE = 20
OVERLAP = WINDOW_SIZE//3
SMOOTHING_WINDOW = 40
SMOOTHING_OVERLAP = SMOOTHING_WINDOW//3

# ---- Online ID Diagnostics --------------------------------------------------

def _angle_wrap_diff_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def _scaled_rmse_curve(pred: np.ndarray, gt: np.ndarray, state_std: np.ndarray) -> np.ndarray:
    """
    pred, gt: [L,10] chronological (older->newer)
    Returns per-step RMSE over kept dims, scaled by HARD_CODED_STATE_STD.
    """
    from utilities.InverseDynamics import KEEP_IDX, POSE_THETA_IDX
    dif = pred.copy() - gt.copy()
    dif[:, POSE_THETA_IDX] = _angle_wrap_diff_np(pred[:, POSE_THETA_IDX], gt[:, POSE_THETA_IDX])
    keep = KEEP_IDX.numpy() if hasattr(KEEP_IDX, "numpy") else KEEP_IDX
    Z = dif[:, keep] / state_std[keep][None, :]
    rmse = np.sqrt(np.mean(Z * Z, axis=1))
    return rmse

def _curve_head_tail_stats(curve: np.ndarray) -> tuple[float,float,float]:
    """
    Returns (mean, head_mean, tail_mean) with head/tail = first/last 25%.
    """
    L = int(curve.shape[0])
    if L == 0: return 0.0, 0.0, 0.0
    k = max(1, L // 4)
    return float(np.mean(curve)), float(np.mean(curve[:k])), float(np.mean(curve[-k:]))


class HistoryForger:
    def __init__(self):
        self.previous_control_inputs = []
        self.previous_measured_states = []  # Only for debugging if needed

        # Progressive, overlapped refiner with smoothing (reused instance → no retracing)
        self.refiner = ProgressiveWindowRefiner(
            mu=Settings.FRICTION_FOR_CONTROLLER if hasattr(Settings, "FRICTION_FOR_CONTROLLER") else None,
            controls_are_pid=True,
            dt=TIMESTEP_ENVIRONMENT,
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
            smoothing_window=SMOOTHING_WINDOW,
            smoothing_overlap=SMOOTHING_OVERLAP,
        )

        self._warmed_once = False
        self.forged_history_applied = False
        self.counter = 0

        # --- Diagnostics / logging (env-controlled) ---
        self.diag_every_n = int(os.getenv("ID_DIAG_EVERY_N", "25"))  # print every N calls
        self.diag_csv = os.getenv("ID_DIAG_CSV", "")                  # optional CSV path
        self._diag_counter = 0
        self._diag_csv_header_written = False
        self._diag_last = {}  # last stats snapshot


    def update_control_history(self, u):
        self.counter += 1
        self.previous_control_inputs.append(u)
        max_len = HISTORY_LENGTH * timesteps_per_controller_update + 1
        if len(self.previous_control_inputs) > max_len:
            self.previous_control_inputs.pop(0)

    def update_state_history(self, s):
        self.previous_measured_states.append(s)
        max_len = HISTORY_LENGTH * timesteps_per_controller_update + 1
        if len(self.previous_measured_states) > max_len:
            self.previous_measured_states.pop(0)

    def _warm_once(self, x_T, Q):
        if self._warmed_once:
            return
        # Tiny subset to force tracing quickly
        T_small = min(5, Q.shape[0])
        if T_small >= 1:
            Qs = Q[-T_small:]  # old->new last few controls
            _ = self.refiner.refine(x_T, Qs)  # warm the internal optimizer
        self._warmed_once = True

    def get_forged_history(self, car_state, waypoint_utils):
        """
        Computes a forged history behind the current `car_state`
        by running inverse dynamics backward. Returns None if not enough data
        or no convergence.
        """
        if self.counter < START_AFTER_X_STEPS:
            return None

        required_length = HISTORY_LENGTH * timesteps_per_controller_update + 1
        if len(self.previous_control_inputs) < required_length:
            self.forged_history_applied = False
            return None

        controls = np.array(self.previous_control_inputs, dtype=np.float32)[-required_length:]

        # Flatten [T,1,2] → [T,2] or accept [T,2] as-is
        if controls.ndim == 3 and controls.shape[1] == 1:
            controls = controls[:, 0, :]
        elif controls.ndim == 1:
            controls = controls[None, :]

        Q_np = controls.astype(np.float32)              # [T, 2], old→new
        x_next = np.asarray(car_state, np.float32)[None]  # [1, 10]

        # Warm once to exclude tracing from timings
        self._warm_once(x_next, Q_np)

        # ---- Refine with stats ----
        states_all, converged_flags, stats = self.refiner.refine_stats(x_next, Q_np)

        # ---- Compare to true measured past (online quality) ----
        try:
            from utilities.InverseDynamics import HARD_CODED_STATE_STD
            # Take last 'required_length' measured states; exclude current (last)
            if len(self.previous_measured_states) >= (required_length + 1):
                gt_past = np.array(self.previous_measured_states[-(required_length+1):-1], dtype=np.float32)
                # predicted 'past_states' chronological (older->newer)
                states_at_control_times = states_all[::timesteps_per_controller_update, :]
                past_states_backwards = states_at_control_times[1:, :]
                pred_past = past_states_backwards[::-1, :]
                rmse_curve = _scaled_rmse_curve(pred_past, gt_past, HARD_CODED_STATE_STD)
                rmse_mean, rmse_head, rmse_tail = _curve_head_tail_stats(rmse_curve)
                stats.update({
                    "rmse_mean": rmse_mean, "rmse_head": rmse_head, "rmse_tail": rmse_tail,
                    "rmse_auc": float(np.trapz(rmse_curve, dx=1.0))
                })

                from utilities.InverseDynamics import KEEP_IDX, HARD_CODED_STATE_STD, POSE_THETA_IDX

                def _scaled(A, B):
                    D = A.copy() - B.copy()
                    D[:, POSE_THETA_IDX] = _angle_wrap_diff_np(A[:, POSE_THETA_IDX], B[:, POSE_THETA_IDX])
                    kept = KEEP_IDX.numpy() if hasattr(KEEP_IDX, "numpy") else KEEP_IDX
                    return D[:, kept] / HARD_CODED_STATE_STD[kept][None, :]

                def _rmse(Z): return float(np.sqrt(np.mean(Z * Z)))

                rmse0 = _rmse(_scaled(pred_past, gt_past))
                rmse_m = _rmse(_scaled(pred_past[:-1], gt_past[1:]))  # pred shifted one step older
                rmse_p = _rmse(_scaled(pred_past[1:], gt_past[:-1]))  # pred shifted one step newer

                stats.update(
                    {"rmse_unshifted": rmse0, "rmse_shift_pred_back_1": rmse_m, "rmse_shift_pred_fwd_1": rmse_p})

        except Exception as _:
            pass

        stats["conv_rate"] = float(np.mean(converged_flags.astype(np.float32)))
        stats["applied"] = float(1.0 if np.all(converged_flags) else 0.0)
        self._diag_last = stats

        # optional periodic print
        self._diag_counter += 1
        if self.diag_every_n > 0 and (self._diag_counter % self.diag_every_n == 0):
            print(
                f"[ID] T={int(stats.get('T', -1))} | total={stats['total_ms']:6.1f} ms "
                f"(inv={stats['inv_ms']:5.1f}, smooth={stats['smooth_ms']:5.1f}, "
                f"prep={stats.get('last_prep_ms', 0):5.1f}, opt={stats.get('last_opt_ms', 0):5.1f}) | "
                f"rmse={stats.get('rmse_mean', float('nan')):.4f} (head={stats.get('rmse_head', float('nan')):.4f}, tail={stats.get('rmse_tail', float('nan')):.4f}) "
                f"| conv={stats['conv_rate']:.2f} | traces={int(stats.get('last_traces', -1))}"
            )

        # optional CSV
        if self.diag_csv:
            import csv, os
            os.makedirs(os.path.dirname(self.diag_csv), exist_ok=True) if os.path.dirname(self.diag_csv) else None
            write_header = (not self._diag_csv_header_written) or (not os.path.exists(self.diag_csv))
            with open(self.diag_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=sorted(stats.keys()))
                if write_header:
                    w.writeheader()
                    self._diag_csv_header_written = True
                w.writerow({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in stats.items()})

        if not np.all(converged_flags):
            self.forged_history_applied = False
            return None

        # states_all[0] is current state, states_all[1:] older states
        # Keep only states at multiples of timesteps_per_controller_update
        states_at_control_times = states_all[::timesteps_per_controller_update, :]
        # Discard the current state (index 0)
        past_states_backwards = states_at_control_times[1:, :]
        # Reverse them to get chronological order
        past_states = past_states_backwards[::-1, :]

        # Next, find nearest waypoint indices for each older state
        nearest_waypoint_indices = [waypoint_utils.nearest_waypoint_index]
        for i in range(len(past_states_backwards)):
            idx, _ = get_nearest_waypoint(
                past_states_backwards[i],
                waypoint_utils.waypoints,
                nearest_waypoint_indices[-1],
                lower_search_limit=-5,
                upper_search_limit=3
            )
            nearest_waypoint_indices.append(idx)

        nearest_waypoint_indices.pop(0)  # remove first (for current state)
        nearest_waypoints_indices = np.array(nearest_waypoint_indices)[::-1]

        look_len = waypoint_utils.look_ahead_steps + waypoint_utils.ignore_steps
        n_wp = len(waypoint_utils.waypoints)
        idx_array = (
            (nearest_waypoints_indices[:, None] + np.arange(look_len)) % n_wp
        )
        next_waypoints_including_ignored = waypoint_utils.waypoints[idx_array]
        # discard the first 'ignore_steps'
        nearest_waypoints = next_waypoints_including_ignored[:, waypoint_utils.ignore_steps:]

        self.forged_history_applied = True
        return past_states, nearest_waypoints

    def feed_planner_forged_history(self, car_state, ranges, waypoint_utils, planner, render_utils, interpolate_local_wp):
        """
        Feeds the "synthetic past" states + waypoints into the planner if forging works.
        """
        history = self.get_forged_history(car_state, waypoint_utils)
        if history is not None:
            past_car_states, all_past_next_waypoints = history

            for past_car_state, past_next_waypoints in zip(past_car_states, all_past_next_waypoints):
                # no obstacles, no new LiDAR
                obstacles = np.array([])
                ranges_   = ranges

                # Interpolate local waypoints for the planner
                next_wps_interp = waypoint_utils.get_interpolated_waypoints(
                    past_next_waypoints, interpolate_local_wp
                )
                planner.pass_data_to_planner(next_wps_interp, past_car_state, obstacles)
                planner.process_observation(ranges_, past_car_state)

            # Optionally update rendering or debugging
            render_utils.update(
                past_car_states_alternative=past_car_states,
            )
        else:
            print("Not enough data for forging history.")
