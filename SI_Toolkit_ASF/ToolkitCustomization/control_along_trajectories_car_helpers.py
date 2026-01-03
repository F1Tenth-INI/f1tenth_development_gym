import re

import numpy as np
import pandas as pd

from utilities.Settings import Settings
from utilities.car_system import initialize_planner, if_mpc_define_cs_variables
from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX
from utilities.state_utilities import STATE_VARIABLES
from utilities.lidar_utils import LidarHelper
from utilities.waypoint_utils import WaypointUtils


class PlannerAsController:
    def __init__(self, controller_config, initial_environment_attributes):
        controller_name = controller_config["controller_name"]
        self.planner = initialize_planner(controller_name)
        self.lidar_utils = LidarHelper()
        self.waypoint_utils = WaypointUtils()
        # Backwards-compatible alias: some code paths expect `self.LIDAR`.
        self.LIDAR = self.lidar_utils
        if hasattr(self.planner, 'lidar_utils'):
            self.planner.lidar_utils = self.lidar_utils
        if hasattr(self.planner, 'waypoint_utils'):
            self.planner.waypoint_utils = self.waypoint_utils
        self.angular_control_dict, self.translational_control_dict = if_mpc_define_cs_variables(self.planner)

        # Optional: forge a fake history using the backward predictor (matches online history-forge experiments).
        # Enabled by Settings.FORGE_HISTORY.
        self.backward_predictor = None
        self._prev_recorded_u = None  # u_{t-1} in controller cadence
        self._state_hist = []         # list[np.ndarray]
        self._wpt_hist = []           # list[np.ndarray]
        self._lidar_hist = []         # list[np.ndarray] (full scan length)

        self._forge_source = str(getattr(Settings, "FORGE_PAST_SOURCE", "backward")).lower()
        self._first_warmup_done = False  # Track if first warmup has been done (for "first_only" mode)
        self._step_counter = 0  # For output stride
        self._output_stride = int(getattr(Settings, "OUTPUT_STRIDE", 1))
        if getattr(Settings, "FORGE_HISTORY", False) and self._forge_source == "backward":
            from utilities.BackwardPredictor import BackwardPredictor
            self.backward_predictor = BackwardPredictor()

    def reset(self):
        # Reset internal state to improve solver stability when re-running a trajectory
        # (e.g. when evaluating the same trajectory for multiple mu values).
        if hasattr(self.lidar_utils, 'reset'):
            self.lidar_utils.reset()
        if hasattr(self.waypoint_utils, 'reset'):
            self.waypoint_utils.reset()
        if hasattr(self.planner, 'reset'):
            self.planner.reset()
        if self.backward_predictor is not None:
            # BackwardPredictor has no explicit reset in current code; just drop history by re-instantiating.
            try:
                from utilities.BackwardPredictor import BackwardPredictor
                self.backward_predictor = BackwardPredictor()
            except Exception:
                pass
        self._prev_recorded_u = None
        self._state_hist = []
        self._wpt_hist = []
        self._lidar_hist = []
        self._first_warmup_done = False  # Track if first warmup has been done (for "first_only" mode)
        self._step_counter = 0  # Reset step counter for stride

    def _reset_rnn_hidden_states(self):
        """Reset LSTM/GRU hidden states to zeros so warmup starts from a clean slate."""
        try:
            # Navigate: planner -> nni -> net_evaluator -> net (TF model)
            nni = getattr(self.planner, "nni", None)
            if nni is None:
                return
            net_evaluator = getattr(nni, "net_evaluator", None)
            if net_evaluator is None:
                return
            net = getattr(net_evaluator, "net", None)
            if net is None:
                return
            # Reset each recurrent layer's states to zeros
            for layer in net.layers:
                layer_name = getattr(layer, "name", "").lower()
                if "lstm" in layer_name or "gru" in layer_name or "rnn" in layer_name:
                    try:
                        layer.reset_states()  # resets to zeros (initial_state default)
                    except Exception:
                        pass
            # Also reset the planner's control_history to zeros (initial state)
            if hasattr(self.planner, "control_history"):
                self.planner.control_history.clear()
                for _ in range(self.planner.control_history_size):
                    self.planner.control_history.append((0.0, 0.0))
            # Reset simulation_index so acceleration warmup logic matches online behavior
            if hasattr(self.planner, "simulation_index"):
                self.planner.simulation_index = 0
        except Exception:
            pass

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, object]" = {}):

        # Some planner/controller stacks (notably the "neural" planner) read mu from Settings.SURFACE_FRICTION.
        # If the dataset provides mu, keep Settings in sync per-step.
        if isinstance(updated_attributes, dict) and "mu" in updated_attributes:
            mu_val = updated_attributes.get("mu", None)
            if mu_val is not None:
                try:
                    Settings.SURFACE_FRICTION = float(mu_val)
                except Exception:
                    pass

        if hasattr(self.planner, 'mpc'):
            self.planner.mu = updated_attributes['mu']

        # Keep planner state in sync with the recorded trajectory sample.
        if hasattr(self.planner, 'car_state'):
            self.planner.car_state = np.array(s, dtype=np.float64)
        if time is not None and hasattr(self.planner, 'time'):
            self.planner.time = time

        self.waypoint_utils.next_waypoints = updated_attributes['next_waypoints']
        lidar_in = np.asarray(updated_attributes['lidar'], dtype=np.float32)
        # Accept either:
        # - full scan of length 1080 (already indexed)
        # - processed-only scan of length len(processed_scan_indices)
        if lidar_in.shape[0] == self.LIDAR.num_scans_total:
            lidar_full = lidar_in
        elif lidar_in.shape[0] == len(self.LIDAR.processed_scan_indices):
            lidar_full = np.zeros((self.LIDAR.num_scans_total,), dtype=np.float32)
            lidar_full[self.LIDAR.processed_scan_indices] = lidar_in
        else:
            raise ValueError(
                f"Unexpected lidar length {lidar_in.shape[0]}; expected {self.LIDAR.num_scans_total} "
                f"(full) or {len(self.LIDAR.processed_scan_indices)} (processed-only)."
            )

        self.LIDAR.update_ranges(lidar_full, np.array(s, dtype=np.float64))

        # ---- Output stride optimization ----
        # With reset_mode="every_step", we can skip warmup+inference for non-stride rows.
        # We only need to update history buffers and return NaN.
        current_step = self._step_counter
        self._step_counter += 1
        skip_inference = (self._output_stride > 1 and (current_step % self._output_stride) != 0)

        if skip_inference:
            # Still update history buffers (needed for future warmups in oracle mode)
            try:
                self._state_hist.append(np.array(s, dtype=np.float32))
                self._wpt_hist.append(np.array(updated_attributes.get("next_waypoints"), dtype=np.float32))
                self._lidar_hist.append(np.array(lidar_full, dtype=np.float32))
            except Exception:
                pass
            # Update BackwardPredictor's internal history FIRST (before updating _prev_recorded_u!)
            # This ensures we feed u(t-1) when processing x(t), matching: x(t) = f(x(t-1), u(t-1))
            if self.backward_predictor is not None:
                if self._prev_recorded_u is not None:
                    try:
                        self.backward_predictor.update_control_history(self._prev_recorded_u)
                    except Exception:
                        pass
                try:
                    self.backward_predictor.update_state_history(np.array(s, dtype=np.float32))
                except Exception:
                    pass
            # NOW update _prev_recorded_u with current control (for next step)
            if isinstance(updated_attributes, dict) and ("angular_control" in updated_attributes) and ("translational_control" in updated_attributes):
                try:
                    a_rec = float(updated_attributes["angular_control"])
                    v_rec = float(updated_attributes["translational_control"])
                    self._prev_recorded_u = np.array([a_rec, v_rec], dtype=np.float32)
                except Exception:
                    pass
            return np.array([np.nan, np.nan], dtype=np.float32)

        # ---- Forged history priming (BEFORE computing current control) ----
        # This is the critical change for RNNs: we want to set a consistent hidden state
        # based on either the BackwardPredictor's forged past, or the true past from the CSV ("oracle").
        if getattr(Settings, "FORGE_HISTORY", False):
            source = str(getattr(Settings, "FORGE_PAST_SOURCE", "backward")).lower()

            # Oracle mode: replay true past states/waypoints/lidar from the CSV to prime hidden state.
            if source == "oracle":
                # Need at least HISTORY_LENGTH past samples.
                H = int(getattr(Settings, "FORGE_HISTORY_LENGTH", 50))
                if len(self._state_hist) >= H:
                    # Check reset mode: "every_step" (reset before each warmup) vs "first_only" (reset only at first warmup)
                    reset_mode = str(getattr(Settings, "FORGE_RESET_MODE", "every_step")).lower()
                    should_reset = (reset_mode == "every_step") or (reset_mode == "first_only" and not self._first_warmup_done)

                    if should_reset:
                        # Reset LSTM/GRU hidden states to zeros before warmup.
                        self._reset_rnn_hidden_states()

                    # Mark first warmup as done (for "first_only" mode)
                    if not self._first_warmup_done:
                        self._first_warmup_done = True

                    past_states = self._state_hist[-H:]
                    past_wpts = self._wpt_hist[-H:]
                    past_lidars = self._lidar_hist[-H:]
                    for ps, pw, pl in zip(past_states, past_wpts, past_lidars):
                        try:
                            if hasattr(self.planner, "car_state"):
                                self.planner.car_state = np.array(ps, dtype=np.float64)
                            self.waypoint_utils.next_waypoints = pw
                            self.LIDAR.update_ranges(pl, np.array(ps, dtype=np.float64))
                            _ = self.planner.process_observation()
                        except Exception:
                            pass

                    # CRITICAL: Restore CURRENT state/waypoints/lidar after warmup,
                    # so process_observation() below computes control for the CURRENT step, not step H-1!
                    if hasattr(self.planner, "car_state"):
                        self.planner.car_state = np.array(s, dtype=np.float64)
                    self.waypoint_utils.next_waypoints = updated_attributes['next_waypoints']
                    self.LIDAR.update_ranges(lidar_full, np.array(s, dtype=np.float64))

            # BackwardPredictor mode (B): use BackwardPredictor forged past, but feed it with RECORDED online controls.
            elif source == "backward" and self.backward_predictor is not None:
                # 1) Update control history with u_{t-1} (recorded) like online sim does.
                if self._prev_recorded_u is not None:
                    try:
                        self.backward_predictor.update_control_history(self._prev_recorded_u)
                    except Exception:
                        pass
                # 2) Update state history with current state (online sim updates state history after control computation;
                # we do it here so get_forged_history can see the newest state).
                try:
                    self.backward_predictor.update_state_history(np.array(s, dtype=np.float32))
                except Exception:
                    pass

                # 3) If forging is ready, replay forged past to prime planner hidden state.
                try:
                    history = self.backward_predictor.get_forged_history(np.array(s, dtype=np.float32), self.waypoint_utils)
                except Exception:
                    history = None
                if history is not None:
                    # Check reset mode: "every_step" (reset before each warmup) vs "first_only" (reset only at first warmup)
                    reset_mode = str(getattr(Settings, "FORGE_RESET_MODE", "every_step")).lower()
                    should_reset = (reset_mode == "every_step") or (reset_mode == "first_only" and not self._first_warmup_done)

                    if should_reset:
                        # Reset LSTM/GRU hidden states to zeros before warmup.
                        self._reset_rnn_hidden_states()

                    # Mark first warmup as done (for "first_only" mode)
                    if not self._first_warmup_done:
                        self._first_warmup_done = True

                    past_car_states, all_past_next_waypoints = history
                    for past_car_state, past_next_waypoints in zip(past_car_states, all_past_next_waypoints):
                        try:
                            if hasattr(self.planner, "car_state"):
                                self.planner.car_state = np.array(past_car_state, dtype=np.float64)
                            self.waypoint_utils.next_waypoints = past_next_waypoints
                            # BackwardPredictor intentionally uses the same current scan (no past lidar reconstruction).
                            self.LIDAR.update_ranges(lidar_full, np.array(past_car_state, dtype=np.float64))
                            _ = self.planner.process_observation()
                        except Exception:
                            pass

                    # CRITICAL: Restore CURRENT state/waypoints/lidar after warmup
                    if hasattr(self.planner, "car_state"):
                        self.planner.car_state = np.array(s, dtype=np.float64)
                    self.waypoint_utils.next_waypoints = updated_attributes['next_waypoints']
                    self.LIDAR.update_ranges(lidar_full, np.array(s, dtype=np.float64))

        # Match `CarSystem` behavior: planners read required inputs from their bound utils/state.
        new_controls = self.planner.process_observation()

        # Normalize output type to plain floats (important for TF models returning tf.Tensor scalars).
        def _to_float(x):
            if hasattr(x, "numpy"):
                x = x.numpy()
            if isinstance(x, np.ndarray):
                return float(x.reshape(-1)[0]) if x.size else float("nan")
            return float(x)

        try:
            a, v = new_controls
            out = np.array([_to_float(a), _to_float(v)], dtype=np.float32)
        except Exception:
            # Fallback: attempt to coerce any array-like into float vector
            arr = np.asarray(new_controls)
            if arr.size >= 2:
                out = np.array([_to_float(arr.flat[0]), _to_float(arr.flat[1])], dtype=np.float32)
            raise

        # ---- Update local histories (for oracle + for predictor control feed) ----
        # Store true past signals from the CSV (state, waypoints, lidar).
        try:
            self._state_hist.append(np.array(s, dtype=np.float32))
            self._wpt_hist.append(np.array(updated_attributes.get("next_waypoints"), dtype=np.float32))
            self._lidar_hist.append(np.array(lidar_full, dtype=np.float32))
        except Exception:
            pass

        # Store recorded online control from CSV for feeding into BackwardPredictor at the next step.
        # Use APPLIED controls (angular_control, translational_control), not raw network output.
        if isinstance(updated_attributes, dict) and ("angular_control" in updated_attributes) and ("translational_control" in updated_attributes):
            try:
                a_rec = float(updated_attributes["angular_control"])
                v_rec = float(updated_attributes["translational_control"])
                self._prev_recorded_u = np.array([a_rec, v_rec], dtype=np.float32)
            except Exception:
                self._prev_recorded_u = out.copy()
        else:
            # Fallback: use offline-computed control
            self._prev_recorded_u = out.copy()

        return out


def controller_creator(controller_config, initial_environment_attributes):
    # For the neural controller, the planner reads mu from Settings.SURFACE_FRICTION.
    # Set it from the dataset if provided, so offline replay matches the recorded run.
    if isinstance(initial_environment_attributes, dict) and "mu" in initial_environment_attributes:
        mu_val = initial_environment_attributes.get("mu", None)
        if mu_val is not None and not (isinstance(mu_val, float) and np.isnan(mu_val)):
            try:
                Settings.SURFACE_FRICTION = float(mu_val)
            except Exception:
                pass
    controller_instance = PlannerAsController(controller_config, initial_environment_attributes)
    return controller_instance


def df_modifier(df):
    # all_waypoints has shape (num_rows, num_waypoints, 8)
    all_waypoints, waypoints_cols = get_waypoints_from_dataset(df)

    lidar, lidar_cols = get_lidar_from_dataset(df)

    all_columns = df.columns
    exclude_cols = ['time'] + list(STATE_VARIABLES) + list(waypoints_cols) + list(lidar_cols)

    # Calculate the remaining columns that are not in the exclusion list.
    remaining_columns = [col for col in all_columns if col not in exclude_cols]

    df_temp = pd.DataFrame(
        {
            'time': df['time'].values,
            'state': list(df[STATE_VARIABLES].values),
            'next_waypoints': list(all_waypoints),
            'lidar': list(lidar),
        }
    )

    df_temp = pd.concat([df_temp, df[remaining_columns]], axis=1)


    # all_waypoints_relative = []
    # for i in range(len(df)):
    #     all_waypoints_relative.append(waypoint_utils.get_relative_positions(all_waypoints[i, :, :], all_car_states[i, :]))
    # all_waypoints_relative = np.array(all_waypoints_relative, dtype=np.float32)

    return df_temp




def add_control_along_trajectories_car(
        df,
        controller_config,
        controller_creator=controller_creator,
        df_modifier=df_modifier,
        controller_output_variable_names=('angular_control_random_mu', 'translational_control_random_mu'),
        integration_method='monte_carlo',
        integration_num_evals=64,
        save_output_only=False,
        **kwargs
):
    environment_attributes_dict = controller_config["environment_attributes_dict"]
    df_temp = df_modifier(df)

    controller_instance = controller_creator(controller_config, initial_environment_attributes={})
    control_outputs = np.zeros((len(df_temp), 2), dtype=np.float32)
    for idx, row in df_temp.iterrows():
        environment_attributes = {key: row[value] for key, value in environment_attributes_dict.items()}
        s = row['state']
        new_controls = controller_instance.step(s, updated_attributes=environment_attributes)
        control_outputs[idx, :] = new_controls

    df[controller_output_variable_names[0]] = control_outputs[:, 0]
    df[controller_output_variable_names[1]] = control_outputs[:, 1]

    return df




def get_waypoints_from_dataset(df):
    # --------------------------------------------------
    # Filter columns that match patterns like "WPT_X_005" or "WPT_Y_12", etc.
    # The regex captures 2 groups:
    #   (1) 'X', 'Y', or 'VX'
    #   (2) the numeric part after the underscore
    # --------------------------------------------------
    pattern = r'^WYPT_(X|Y|VX)_(\d+)$'
    wpt_cols = df.filter(regex=pattern).columns

    # --------------------------------------------------
    # Find the maximum waypoint number 'm' by parsing the numeric part of each column name.
    # This avoids having to guess the largest index.
    # --------------------------------------------------
    max_index = 0
    for col in wpt_cols:
        # Extract the numeric index from the column name
        match = re.match(pattern, col)
        idx_num = int(match.group(2))
        if idx_num > max_index:
            max_index = idx_num

    # --------------------------------------------------
    # Initialize the 3D array of shape (n, m, 10) with zeros.
    # The controller stack expects waypoints shaped like `WaypointUtils.next_waypoints`,
    # i.e. (LOOK_AHEAD_STEPS, 10) with indices defined in `utilities/waypoint_utils.py`.
    # We typically only have X/Y/VX recorded, so other channels stay at 0.
    # --------------------------------------------------
    all_waypoints = np.zeros((len(df), max_index + 1, 10), dtype=np.float32)

    # --------------------------------------------------
    # Prepare a mapping from the axis type ('X','Y','VX') to
    # the position in the last dimension, so we know where to
    # store each WPT column's values.
    # --------------------------------------------------
    channel_map = {
        'X': WP_X_IDX,
        'Y': WP_Y_IDX,
        'VX': WP_VX_IDX,
    }

    # --------------------------------------------------
    # Fill all_waypoints with the values from df for each waypoint variable.
    # Subtract 1 from idx_num because waypoints (1..m) map to array indices (0..m-1).
    # --------------------------------------------------
    for col in wpt_cols:
        axis_type, idx_str = re.match(pattern, col).groups()
        idx_num = int(idx_str)  # zero-based
        all_waypoints[:, idx_num, channel_map[axis_type]] = df[col].values

    return all_waypoints, wpt_cols  # (len(df), num_waypoints, 8) array


def get_lidar_from_dataset(df):

    pattern = r'^LIDAR_(\d+)$'
    LIDAR_cols = df.filter(regex=pattern).columns
    LIDAR_array = np.array(df[LIDAR_cols].values, dtype=np.float32)

    return LIDAR_array, LIDAR_cols  # (len(df), num_waypoints, 8) array

