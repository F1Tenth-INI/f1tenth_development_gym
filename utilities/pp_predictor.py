'''
A simple predictor for opponents using the Pure Pursuit Planner.

The opponents state is provided by the ego car. Additionally, the predictor computes the next waypoints based on the state.

Given the state and the next way points, we can use the Pure Pursuit Planners "process_observations" function 
to compute agent controls and thus predict the next states.
'''

from collections import deque
from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
from sim.f110_sim.envs.base_classes import normalize_state_yaw
from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_dynamics_pacejka_jax_from_settings
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.waypoint_utils import *
import numpy as np
import jax.numpy as jnp


class PurePursuitPredictor:
    def __init__(self, waypoints_utils, horizon_steps=25):

        self.waypoints_utils = waypoints_utils
        self.horizon_steps = horizon_steps
        self.pp_planner = PurePursuitPlanner()

        self.params = jnp.array(VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE).to_np_array())
        self.dt = Settings.TIMESTEP_SIM


    def create_observation(self, car_state, nearest_waypoint_index):
        '''
        Compute next waypoints. 
        Follow same computation steps as "update_next_waypoints" in waypoint_utils.py without side effects.
        ''' 

        search_threshold_squared = Settings.GLOBAL_WAYPOINTS_SEARCH_THRESHOLD**2
        
        (
            nearest_waypoint_index,
            next_waypoints,
            sector_index,
            sector_scaling,
        ) = jit_update_next_waypoints(
            car_state,
            self.waypoints_utils.waypoints,
            nearest_waypoint_index,
            self.waypoints_utils.ignore_steps,
            self.waypoints_utils.look_ahead_steps,
            self.waypoints_utils.decrease_resolution_factor,
            self.waypoints_utils.sectors,
            search_threshold_squared,  
        )

        observation = {"car_state": car_state, "next_waypoints": next_waypoints}

        return nearest_waypoint_index, observation

    def get_agent_controls(self, state, nearest_waypoint_index):
        '''
        Simply call the Pure Pursuit Planner to get agent controls based on computed waypoints and current state.
        '''

        nearest_waypoint_index, observation = self.create_observation(state, nearest_waypoint_index)
        steering, accel = self.pp_planner.process_observation(observation)
        return nearest_waypoint_index, np.array([steering, accel], dtype=np.float64)

    def predict(self, state, pending_controls, vel_factor, nearest_waypoint_index = None):
        '''
        Predict the opponent's next "horizon_steps" states based on the current state, next waypoints, and pending agent controls.
        return: trajectory, array of shape (horizon_steps + 1, 10)
        '''

        state = np.asarray(state, dtype=np.float64)
        self.pp_planner.waypoint_velocity_factor = vel_factor
        self.pp_planner.correcting_index = 0
        temporary_queue = []
        for control in pending_controls:
            temporary_queue.append(np.asarray(control, dtype=np.float64))

        queue = deque(temporary_queue)
        trajectory = [state.copy()]

        for i in range(self.horizon_steps):
            nearest_waypoint_index, controls = self.get_agent_controls(state, nearest_waypoint_index)
            queue.append(controls)
            executed = queue.popleft()

            intermediate_steps = int(Settings.TIMESTEP_CONTROL / Settings.TIMESTEP_SIM)
            # Run "intermediate_steps" number of physics substeps
            for j in range(intermediate_steps):
                new_state = car_dynamics_pacejka_jax_from_settings(state, executed, self.params, self.dt)
                state = normalize_state_yaw(np.array(new_state, dtype=np.float64))
            trajectory.append(state.copy())

        return np.array(trajectory, dtype=np.float64)


