'''
Helper Class created to keep track and report basic metrics about an experiment (lap times, number of collisions, number of overtaking)
'''

from __future__ import annotations
from utilities.Settings import Settings
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    STATE_VARIABLES,
    STEERING_ANGLE_IDX,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from run.run_simulation import RacingSimulation
import numpy as np
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.car_system import CarSystem
ego_params = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
EGO_WIDTH = float(ego_params.width)
EGO_LENGTH = float(ego_params.length)
OPPONENT_WIDTH = Settings.VIRTUAL_OPPONENT_SIZE[0]
OPPONENT_LENGTH = Settings.VIRTUAL_OPPONENT_SIZE[1]
CLEARANCE = (EGO_LENGTH + OPPONENT_LENGTH)/2


class OvertakeTracker:

    def __init__(self, sim: RacingSimulation):
        self.nr_drivers = sim.number_of_drivers
        self.overtaking_in_progress = []
        self.nr_overtakings = 0

        
        ego_state = sim.world_sim.agents[0].state
        ego_position = np.asarray(ego_state[[POSE_X_IDX, POSE_Y_IDX]])
        drivers: list[CarSystem] = sim.drivers[1:]

        for i, driver in enumerate(drivers):
            opp_state = sim.world_sim.agents[i + 1].state
            opp_position = np.asarray(opp_state[[POSE_X_IDX, POSE_Y_IDX]])
            opp_theta = opp_state[POSE_THETA_IDX]
            cos, sin = np.cos(opp_theta), np.sin(opp_theta)

            difference_vector = ego_position - opp_position
            distance = np.sqrt(np.dot(difference_vector, difference_vector))
            clearance = cos*difference_vector[0] + sin*difference_vector[1]
            is_ahead = clearance > 0

            if not is_ahead and distance < 10*CLEARANCE:
                self.overtaking_in_progress.append(True)
            else:
                self.overtaking_in_progress.append(False)

        self.buffer_backup = self.overtaking_in_progress.copy()



    def update(self, sim):
        ego_state = sim.world_sim.agents[0].state
        ego_position = np.asarray(ego_state[[POSE_X_IDX, POSE_Y_IDX]])
        drivers: list[CarSystem] = sim.drivers[1:]

        for i, driver in enumerate(drivers):
            opp_state = sim.world_sim.agents[i + 1].state
            opp_position = np.asarray(opp_state[[POSE_X_IDX, POSE_Y_IDX]])
            opp_theta = opp_state[POSE_THETA_IDX]
            cos, sin = np.cos(opp_theta), np.sin(opp_theta)

            difference_vector = ego_position - opp_position
            distance = np.sqrt(np.dot(difference_vector, difference_vector))
            clearance = cos*difference_vector[0] + sin*difference_vector[1]
            is_ahead = clearance > 0

            if not is_ahead:
                if distance <10*CLEARANCE:
                    self.overtaking_in_progress[i] = True
                else:
                    self.overtaking_in_progress[i] = False
            else:
                if self.overtaking_in_progress[i] and clearance >= CLEARANCE:
                    #Ego has succesfully overtaken opponent.
                    self.nr_overtakings += 1
                    self.overtaking_in_progress[i] = False

    def reset(self):
        self.overtaking_in_progress = [False] * len(self.overtaking_in_progress)

class MetricTracker:

    def __init__(self, overtake_tracker: OvertakeTracker):
        self.overtake_tracker = overtake_tracker
        self.lap_times = []
        self.completed_laps = 0
        self.car_collisions = 0
        self.wall_collisions = 0
        self.other_resets = 0
        self.last_outcome = None

    def update(self, sim: RacingSimulation):
        ego: CarSystem = sim.drivers[0]
        ego_observations = ego.driver_observation or {}

        if ego._lap_finished:
            self.lap_times.append(float(ego._last_lap_time))
            self.completed_laps += 1

        if ego_observations.get("done"):
            if sim.world_sim.collision_idx[0] > 0:
                self.car_collisions += 1
                self.lap_times.append(np.nan)
                self.last_outcome = "car_collision"
            elif sim.world_sim.agents[0].in_collision:
                self.wall_collisions += 1
                self.lap_times.append(np.nan)
                self.last_outcome = "wall_collision"
            else:
                self.other_resets += 1
                self.lap_times.append(np.nan)
                term = ego_observations.get("episode_termination") or {}
                self.last_outcome = str(term.get("reason", "other")) if isinstance(term, dict) else "other"

            self.overtake_tracker.reset()
            return

        self.overtake_tracker.update(sim)

    def summary(self):
        print(f"Metric Summary: \n - Lap times: {self.lap_times} \n - Average Lap Time: {np.nanmean(self.lap_times)} \n - Number of overtakings: {self.overtake_tracker.nr_overtakings}\n - Completed Laps: {self.completed_laps} \n - Car Collisions: {self.car_collisions} \n - Wall Collisions: {self.wall_collisions} \n - Other resets: {self.other_resets}")



