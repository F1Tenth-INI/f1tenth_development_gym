import numpy as np
from copy import deepcopy

from utilities.waypoint_utils import get_nearest_waypoint
from utilities.state_utilities import STATE_VARIABLES, CONTROL_INPUTS
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.computation_library import select_library, TensorType
from SI_Toolkit.General.variable_parameters import VariableParameters
from utilities.InverseDynamics import CarInverseDynamics

HISTORY_LENGTH = 20  # In controller updates
TIMESTEP_CONTROL = 0.02
TIMESTEP_ENVIRONMENT = 0.01
timesteps_per_controller_update = int(TIMESTEP_CONTROL/TIMESTEP_ENVIRONMENT)
config_online_predictor = {
    'dt': -TIMESTEP_ENVIRONMENT,
    'intermediate_steps': 1,
    'computation_library': 'Numpy',
    'predictor_specification': 'HistoryForgerPredictor',
    # 'initial_environment_attributes': {'mu': np.random.uniform(0.1, 1.0, (256,))},
    'initial_environment_attributes': {'mu': 0.75},
    'state_components': STATE_VARIABLES,
    'control_input_components': CONTROL_INPUTS,
}


class HistoryForger:
    def __init__(self):
        self.previous_control_inputs = []

        self.config_online_predictor_for_training  = config_online_predictor
        self.state_components = config_online_predictor['state_components']
        self.control_input_components = config_online_predictor['control_input_components']
        self.lib = select_library(config_online_predictor['computation_library'])

        self.initial_environment_attributes = {key: self.lib.to_variable(value, self.lib.float32) for key, value in
                                               config_online_predictor['initial_environment_attributes'].items()}

        self.variable_parameters = VariableParameters(self.lib)
        self.variable_parameters.set_attributes(self.initial_environment_attributes)

        self.car_inverse_dynamics = CarInverseDynamics()

        self.previous_measured_states = []


    def update_predictor_attributes(self, updated_attributes: "dict[str, TensorType]"):
        for property, new_value in updated_attributes.items():
            attr = getattr(self.variable_parameters, property)
            self.lib.assign(attr, self.lib.to_tensor(new_value, attr.dtype))

    def update_control_history(self, u):
        self.previous_control_inputs.append(u)
        if len(self.previous_control_inputs) > HISTORY_LENGTH * timesteps_per_controller_update:
            self.previous_control_inputs.pop(0)

    def update_state_history(self, s):
        self.previous_measured_states.append(s)
        if len(self.previous_measured_states) > HISTORY_LENGTH * timesteps_per_controller_update:
            self.previous_measured_states.pop(0)

    def get_forged_history(self, car_state, waypoint_utils):

        if len(self.previous_control_inputs) < HISTORY_LENGTH * timesteps_per_controller_update:
            return None

        Q = np.array(self.previous_control_inputs)[::-1, :, :]
        Q = np.transpose(Q, (1, 0, 2))
        s = car_state[np.newaxis, :]

        states = [s]
        for i in range(Q.shape[1]):
            s_previous, converged = self.car_inverse_dynamics.step_core(states[i], Q[:, i, :])
            states.append(s_previous)
            if not converged:
                return None
        states = np.array(states)
        past_states_backwards = states[::timesteps_per_controller_update, 0, :]  # Only keep states at control times and remove the batch dimension
        past_states_backwards = past_states_backwards[1:, :]  # Remove the first state, which is the current state
        past_states = past_states_backwards[::-1, :]  # Reverse the order of the states

        nearest_waypoint_indices = [waypoint_utils.nearest_waypoint_index]
        for i in range(len(past_states_backwards)):
            idx, _  = get_nearest_waypoint(past_states_backwards[i], waypoint_utils.waypoints, nearest_waypoint_indices[-1], -5, 3)
            nearest_waypoint_indices.append(idx)
        nearest_waypoint_indices.pop(0)
        nearest_waypoints_indices = np.array(nearest_waypoint_indices)[::-1]

        next_waypoints_indices_including_ignored = (
                (nearest_waypoints_indices[:, None] + np.arange(waypoint_utils.look_ahead_steps + waypoint_utils.ignore_steps)) % len(waypoint_utils.waypoints)
        )

        next_waypoints_including_ignored = waypoint_utils.waypoints[next_waypoints_indices_including_ignored]

        # Discard the first `ignore_steps` waypoints for each starting index, keeping only the lookahead ones.
        # This slicing along axis 1 gives an array of shape (number_of_start_points, look_ahead_steps, waypoint_dimension).
        nearest_waypoints = next_waypoints_including_ignored[:, waypoint_utils.ignore_steps:]

        # self.car_inverse_dynamics.change_friction_coefficient(1.0-self.car_inverse_dynamics.car_model.car_parameters.mu)

        return past_states, nearest_waypoints


    def feed_planner_forged_history(self, car_state, ranges, waypoint_utils, planner, render_utils, interpolate_local_wp):
        history = self.get_forged_history(car_state, waypoint_utils)
        if history is not None:
            past_car_states, all_past_next_waypoints = history
            # for past_car_state, past_next_waypoints in zip(past_car_states, all_past_next_waypoints):
            #     obstacles = np.array([])  # TODO: Assumed no obstacles for now
            #     ranges = ranges     # TODO: Assumed no lidar for now
            #     past_next_interpolated_waypoints = waypoint_utils.get_interpolated_waypoints(past_next_waypoints, interpolate_local_wp)
            #     planner.pass_data_to_planner(past_next_interpolated_waypoints, past_car_state, obstacles)
            #     angular_control, translational_control = planner.process_observation(ranges, past_car_state)

            render_utils.update(
                past_car_states_alternative=past_car_states,
            )
