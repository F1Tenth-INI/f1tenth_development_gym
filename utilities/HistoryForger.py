# HistoryForger.py

import numpy as np

from utilities.waypoint_utils import get_nearest_waypoint
from utilities.InverseDynamics import CarInverseDynamics
from utilities.Settings import Settings

HISTORY_LENGTH = 20  # in 'controller updates'
TIMESTEP_CONTROL = 0.02
TIMESTEP_ENVIRONMENT = 0.01
timesteps_per_controller_update = int(TIMESTEP_CONTROL / TIMESTEP_ENVIRONMENT)
START_AFTER_X_STEPS = 100

class HistoryForger:
    def __init__(self):
        self.previous_control_inputs = []
        self.previous_measured_states = []  # Only for debugging if needed

        self.car_inverse_dynamics = CarInverseDynamics()

        if Settings.FRICTION_FOR_CONTROLLER is not None:
            self.car_inverse_dynamics.change_friction_coefficient(Settings.FRICTION_FOR_CONTROLLER)

        self.forged_history_applied = False
        self.counter = 0

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

    def get_forged_history(self, car_state, waypoint_utils):
        """
        Computes a forged history behind the current `car_state`
        by running inverse dynamics backward. Returns None if not enough data
        or no convergence.
        """
        if self.counter < START_AFTER_X_STEPS:
            return None

        required_length = HISTORY_LENGTH * timesteps_per_controller_update
        if len(self.previous_control_inputs) < required_length:
            self.forged_history_applied = False
            return None

        # Reverse the stored controls in time
        Q_np = np.array(self.previous_control_inputs)[::-1, :, :]  # shape [time_steps, 1, control_dim]
        Q_np = Q_np[:, 0, :]  # shape => [time_steps, control_dim]

        x_next = car_state[np.newaxis, :]  # shape => [1, state_dim]

        # Do the entire backward pass
        states_all, converged_flags = self.car_inverse_dynamics.inverse_entire_trajectory(x_next, Q_np)

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
                planner.process_observation(ranges_, past_car_state)

            # Optionally update rendering or debugging
            render_utils.update(
                past_car_states_alternative=past_car_states,
            )
        else:
            print("Not enough data for forging history.")
