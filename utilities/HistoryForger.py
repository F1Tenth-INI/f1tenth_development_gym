# HistoryForger.py

import os
import numpy as np

from utilities.waypoint_utils import get_nearest_waypoint
from utilities.InverseDynamics import ProgressiveWindowRefiner
from utilities.Settings import Settings

HISTORY_LENGTH = 50  # in 'controller updates'
TIMESTEP_CONTROL = float(Settings.TIMESTEP_CONTROL)
TIMESTEP_ENVIRONMENT = float(Settings.TIMESTEP_SIM)
timesteps_per_controller_update = max(
    1, int(round(TIMESTEP_CONTROL / TIMESTEP_ENVIRONMENT))
)
FORGE_AT_CONTROLLER_RATE = True

START_AFTER_X_STEPS = HISTORY_LENGTH

# HistoryForger.py (top-level constants)
WINDOW_SIZE         = 15
OVERLAP             = 10
SMOOTHING_WINDOW    = None
SMOOTHING_OVERLAP   = 15


# ---- Online ID Diagnostics --------------------------------------------------

def _angle_wrap_diff_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def _scaled_rmse_curve(pred: np.ndarray, gt: np.ndarray, state_std: np.ndarray) -> np.ndarray:
    """
    pred, gt: [L,10] chronological (older->newer)
    Returns per-step RMSE over kept dims, scaled by HARD_CODED_STATE_STD.
    """
    from utilities.InverseDynamics import KEEP_IDX, POSE_THETA_IDX

    pred = pred.copy()
    gt   = gt.copy()

    # Wrapped heading error; sin/cos excluded by KEEP_IDX anyway
    pred[:, POSE_THETA_IDX] = np.unwrap(pred[:, POSE_THETA_IDX])
    gt[:,   POSE_THETA_IDX] = np.unwrap(gt[:,   POSE_THETA_IDX])
    ang_err = _angle_wrap_diff_np(pred[:, POSE_THETA_IDX], gt[:, POSE_THETA_IDX])

    dif = pred - gt
    dif[:, POSE_THETA_IDX] = ang_err

    keep = KEEP_IDX.numpy() if hasattr(KEEP_IDX, "numpy") else KEEP_IDX
    # Guard against zero/denorm stds
    eps = 1e-12
    scale = np.maximum(np.asarray(state_std, dtype=np.float32), eps)[keep]

    Z = dif[:, keep] / scale[None, :]  # [L, |keep|]

    # NaN-safe per-step RMSE: return NaN for rows with any non-finite entry
    finite_rows = np.isfinite(Z).all(axis=1)
    rmse = np.full((Z.shape[0],), np.nan, dtype=np.float32)
    if np.any(finite_rows):
        Zf = Z[finite_rows]
        rmse[finite_rows] = np.sqrt(np.mean(Zf * Zf, axis=1)).astype(np.float32)
    return rmse

def _curve_head_tail_stats(curve: np.ndarray) -> tuple[float,float,float]:
    """
    Returns (mean, head_mean, tail_mean) with head/tail = first/last 25%.
    """
    L = int(curve.shape[0])
    if L == 0: return float('nan'), float('nan'), float('nan')
    k = max(1, L // 4)
    mean_all  = float(np.nanmean(curve)) if np.any(np.isfinite(curve))          else float('nan')
    mean_head = float(np.nanmean(curve[:k])) if np.any(np.isfinite(curve[:k]))  else float('nan')
    mean_tail = float(np.nanmean(curve[-k:])) if np.any(np.isfinite(curve[-k:])) else float('nan')
    return mean_all, mean_head, mean_tail



class HistoryForger:
    def __init__(self):
        self.previous_control_inputs = []
        self.previous_measured_states = []  # Only for debugging if needed

        # --- Cadence selection (env-rate forge vs ctrl-rate forge) -----------
        # If ID_FORGE_AT_CONTROLLER_RATE is True, the internal inverse solver runs
        # at controller dt (fewer steps); otherwise it runs at env dt (finer grid).
        # Regardless of this, we will ALWAYS output (and compute RMSE) at the
        # controller cadence for comparability and planner feeding.
        self.dt_env = TIMESTEP_ENVIRONMENT
        self.dt_ctrl = TIMESTEP_CONTROL
        self.ts = timesteps_per_controller_update
        self.forge_ctrl_rate = FORGE_AT_CONTROLLER_RATE
        # How many solver steps correspond to one controller decision in the current forge mode:
        #  - env-rate forge → stride = ts (collapse each control-hold)
        #  - ctrl-rate forge → stride = 1 (each step is a control boundary already)
        self.out_stride_ctrl = 1 if self.forge_ctrl_rate else self.ts

        # Progressive, overlapped refiner with smoothing (reused instance → no retracing)
        self.refiner = ProgressiveWindowRefiner(
            mu=Settings.FRICTION_FOR_CONTROLLER if hasattr(Settings, "FRICTION_FOR_CONTROLLER") else None,
            controls_are_pid=False,
            dt=(self.dt_ctrl if self.forge_ctrl_rate else self.dt_env),  # CHANGED: forge cadence dt
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

        # CHANGED: enable CSV by default (path can be overridden via env)
        self.diag_csv = os.getenv("ID_DIAG_CSV", "logs/id_diag_online.csv")

        self._diag_counter = 0
        self._diag_csv_header_written = False
        self._diag_last = {}  # last stats snapshot

        # --- For optional rendering overlays (controller cadence, oldest→newest) ---
        self._last_gt_past = None      # np.ndarray [H,10]
        self._last_prior_past = None   # np.ndarray [H,10]
        self._last_prior_full_newest_first = None  # [T+1,10], whole horizon, 0 == x_T, newest→older



    def update_control_history(self, u):
        self.counter += 1
        self.previous_control_inputs.append(u)
        max_len = HISTORY_LENGTH * timesteps_per_controller_update
        if len(self.previous_control_inputs) > max_len:
            self.previous_control_inputs.pop(0)

    def update_state_history(self, s):
        self.previous_measured_states.append(s)
        max_len = HISTORY_LENGTH * timesteps_per_controller_update
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

    def _compress_env_controls_to_ctrl_rate(self, env_controls: np.ndarray) -> np.ndarray:
        """
        Convert env-rate control log (old->new, length = H*ts+1) into controller-rate
        sequence (old->new, length = H+1) by taking the LAST env sample of each hold.
        This aligns with sampling the predicted states at control boundaries.
        """
        H = HISTORY_LENGTH
        ts = self.ts
        L = int(env_controls.shape[0])

        assert L == H * ts, f"Control span must be H*ts (= {H * ts}); got {L}."

        blocks = env_controls.reshape(H, ts, 2)  # old->new blocks
        per_decision = blocks[:, -1, :]  # last env tick in each control-hold -> [H,2]
        return per_decision

    def get_forged_history(self, car_state, waypoint_utils):
        """
        Computes a forged history behind the current `car_state`
        by running inverse dynamics backward. Returns None if not enough data
        or no convergence.
        """
        if self.counter < START_AFTER_X_STEPS:
            self._last_gt_past = None
            self._last_prior_past = None
            return None

        required_length = HISTORY_LENGTH * timesteps_per_controller_update
        if len(self.previous_control_inputs) < required_length:
            self.forged_history_applied = False
            self._last_gt_past = None
            self._last_prior_past = None
            return None

        controls = np.array(self.previous_control_inputs, dtype=np.float32)[-required_length:]

        # Flatten [T,1,2] → [T,2] or accept [T,2] as-is
        if controls.ndim == 3 and controls.shape[1] == 1:
            controls = controls[:, 0, :]
        elif controls.ndim == 1:
            controls = controls[None, :]

        # CHANGED: choose forge cadence for controls fed to the solver
        if self.forge_ctrl_rate:
            Q_np = self._compress_env_controls_to_ctrl_rate(controls.astype(np.float32))  # [H,2]
        else:
            Q_np = controls.astype(np.float32)  # [H*ts,2] (fine-grain)

        x_next = np.asarray(car_state, np.float32)[None]  # [1, 10]

        # Warm once to exclude tracing from timings
        self._warm_once(x_next, Q_np)

        # ---- Refine with stats ----
        # ---- Refine with stats ----
        states_all, converged_flags, stats = self.refiner.refine_stats(x_next, Q_np)

        # ---- Build controller-cadence predicted past ONCE (reused everywhere) ----
        states_at_ctrl = states_all[::self.out_stride_ctrl, :]  # newest→older at controller boundaries
        past_states_backwards = states_at_ctrl[1:, :]  # drop current; newest→older
        past_states = past_states_backwards[::-1, :].astype(np.float32)  # oldest→newest

        # ---- Compare to true measured past (online quality) ----
        try:
            ts = timesteps_per_controller_update
            H = HISTORY_LENGTH
            need_env = H * ts

            if len(self.previous_measured_states) >= need_env:
                idxs = [-(k * ts) for k in range(H, 0, -1)]
                gt_past = np.array([self.previous_measured_states[i] for i in idxs], dtype=np.float32)

                # Expose ground truth (controller cadence, oldest→newest)
                self._last_gt_past = gt_past.copy()

                # Sanity check (helps catch future refactors)
                assert past_states.shape == gt_past.shape == (H, states_all.shape[1]), \
                    f"Shape mismatch: pred {past_states.shape}, gt {gt_past.shape}"

                # Per-step scaled RMSE curve and summary stats
                from utilities.InverseDynamics import HARD_CODED_STATE_STD, KEEP_IDX, POSE_THETA_IDX

                rmse_curve = _scaled_rmse_curve(past_states, gt_past, HARD_CODED_STATE_STD)
                rmse_mean, rmse_head, rmse_tail = _curve_head_tail_stats(rmse_curve)
                stats.update({
                    "rmse_mean": rmse_mean,
                    "rmse_head": rmse_head,
                    "rmse_tail": rmse_tail,
                    "rmse_auc": float(np.trapz(np.nan_to_num(rmse_curve, nan=0.0), dx=1.0)),

                })

                def _scaled(A, B):
                    D = A.copy() - B.copy()
                    D[:, POSE_THETA_IDX] = _angle_wrap_diff_np(A[:, POSE_THETA_IDX], B[:, POSE_THETA_IDX])
                    kept = KEEP_IDX.numpy() if hasattr(KEEP_IDX, "numpy") else KEEP_IDX
                    eps = 1e-12
                    scale = np.maximum(HARD_CODED_STATE_STD.astype(np.float32), eps)
                    return D[:, kept] / scale[kept][None, :]

                def _rmse(Z: np.ndarray) -> float:
                    if Z.size == 0:
                        return float('nan')
                    mask = np.isfinite(Z).all(axis=1)
                    if not np.any(mask):
                        return float('nan')
                    Zf = Z[mask]
                    return float(np.sqrt(np.nanmean(Zf * Zf)))

                rmse0 = _rmse(_scaled(past_states, gt_past))
                rmse_m = _rmse(_scaled(past_states[:-1], gt_past[1:]))  # pred shifted older by 1 ctrl step
                rmse_p = _rmse(_scaled(past_states[1:], gt_past[:-1]))  # pred shifted newer by 1 ctrl step

                stats.update({
                    "rmse_unshifted": rmse0,
                    "rmse_shift_pred_back_1": rmse_m,
                    "rmse_shift_pred_fwd_1": rmse_p,
                })

        except Exception as _:
            self._last_gt_past = None
            pass

        ...
        # --- extra diagnostics (optional; controlled by env) ---
        try:
            import os
            if bool(int(os.getenv("ID_EXTRA_DIAG", "1"))):
                from utilities import id_diagnostics as IDD

                # 1) Time-shift / alignment probe (requires GT)
                if self._last_gt_past is not None:
                    stats.update(IDD.probe_time_shift(past_states, self._last_gt_past))

                # 2) Per-dimension residuals & scales (requires GT)
                if self._last_gt_past is not None:
                    stats.update(IDD.per_dim_residuals(past_states, self._last_gt_past))

                # 3) Control-hold aggregation mismatch check
                if len(self.previous_control_inputs) >= required_length:
                    controls_env = np.array(self.previous_control_inputs, dtype=np.float32)[-required_length:]
                    stats.update(IDD.compare_control_aggregations(self.refiner, x_next, controls_env, self.ts,
                                                                  gt_past=(
                                                                      self._last_gt_past if self._last_gt_past is not None else None)))

                # 4) dt / rounding sanity
                stats.update(IDD.dt_alignment_check(self.dt_ctrl, self.dt_env, self.ts))

                # 5) Prior influence audit (requires prior snapshots + GT)
                if self._last_gt_past is not None:
                    stats.update(IDD.prior_influence_audit(self.refiner, self._last_gt_past, self.out_stride_ctrl))

                # 6) Dynamics consistency (forward model, newest->older)
                stats.update(IDD.dynamics_consistency(states_all, Q_np, self.refiner))

                # Decide once per call
                do_sweep_windows = os.getenv("ID_SWEEP_WINDOWS", "1").strip() not in ("", "0")
                do_sweep_model = os.getenv("ID_SWEEP_MODEL", "1").strip() not in ("", "0")

                # Start only when forging has enough data and we actually produced a full past window
                sweeps_ready = (past_states is not None) and (past_states.shape[0] == HISTORY_LENGTH)

                # 7) Window & overlap sweep (default ON; gated)
                if do_sweep_windows and sweeps_ready:
                    stats.update(IDD.window_overlap_sweep(x_next, Q_np, self.refiner, self._last_gt_past))

                # 8) Model mismatch sweep (default ON; gated)
                if do_sweep_model and sweeps_ready:
                    stats.update(IDD.model_param_sweep(x_next, Q_np, self._last_gt_past))

                # 9) Angle/sin-cos/slip sanity (requires GT)
                if self._last_gt_past is not None:
                    stats.update(IDD.angle_and_slip_sanity(past_states, self._last_gt_past))

                # 10) Waypoint association & tangent alignment
                try:
                    stats.update(IDD.waypoint_alignment_check(past_states, waypoint_utils))
                except Exception:
                    pass
        except Exception as e:
            stats["diag_error"] = str(e)[:120]

        # optional CSV (start only once we had enough data to build past_states)
        if self.diag_csv and (past_states is not None) and (past_states.shape[0] == HISTORY_LENGTH):
            import csv, os
            os.makedirs(os.path.dirname(self.diag_csv), exist_ok=True) if os.path.dirname(self.diag_csv) else None
            write_header = (not self._diag_csv_header_written) or (not os.path.exists(self.diag_csv))
            with open(self.diag_csv, "a", newline="") as f:
                # tiny robustness: ignore unexpected keys if future rows differ slightly
                w = csv.DictWriter(f, fieldnames=sorted(stats.keys()), extrasaction="ignore", restval="")
                if write_header:
                    w.writeheader()
                    self._diag_csv_header_written = True
                w.writerow({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in stats.items()})

        if not np.all(converged_flags):
            self.forged_history_applied = False
            return None

        # --- Prior used during optimization (already assembled newest→older inside refiner) ---
        try:
            prior_all = getattr(self.refiner, "_last_prior_newest_first", None)  # [T+1,10] newest→older
            if prior_all is not None and isinstance(prior_all, np.ndarray):
                prior_at_ctrl = prior_all[::self.out_stride_ctrl, :]   # sample at controller boundaries
                prior_past_backwards = prior_at_ctrl[1:, :]            # drop current
                self._last_prior_past = prior_past_backwards[::-1, :].astype(np.float32)  # oldest→newest
            else:
                self._last_prior_past = None
        except Exception:
            self._last_prior_past = None

        # Whole-horizon anchored prior produced by refiner (render-only baseline)
        try:
            prior_full_all = getattr(self.refiner, "_last_prior_full_newest_first", None)  # [T+1,10], newest→older
            if prior_full_all is not None and isinstance(prior_full_all, np.ndarray):
                # Sample at controller boundaries to stay comparable with GT/progressive overlays.
                prior_full_at_ctrl = prior_full_all[::self.out_stride_ctrl, :]  # [H+1,10] newest→older
                prior_full_past_backwards = prior_full_at_ctrl[1:, :]           # drop current, keep H, newest→older (older as index grows)
                self._last_prior_full_past = prior_full_past_backwards[::-1, :].astype(np.float32)  # oldest→newest
            else:
                self._last_prior_full_past = None
        except Exception:
            self._last_prior_full_past = None



        # states_all[0] is current state, states_all[1:] older states
        states_at_control_times = states_all[::self.out_stride_ctrl, :]
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
                gt_past_car_states=self._last_gt_past,
                prior_past_car_states=self._last_prior_past,
                prior_full_past_car_states=self._last_prior_full_past,
            )

        else:
            print("Not enough data for forging history.")
