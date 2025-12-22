import re

import numpy as np
import pandas as pd

from tqdm import trange

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

    def reset(self):
        pass

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):

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
        # Match `CarSystem` behavior: planners read required inputs from their bound utils/state.
        new_controls = self.planner.process_observation()

        return new_controls


def controller_creator(controller_config, initial_environment_attributes):
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

