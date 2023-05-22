from abc import ABC, abstractmethod
import numpy as np

from utilities.lidar_utils import LidarHelper

from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)


class template_planner(ABC):
    def __init__(self):

        self.LIDAR = LidarHelper()
        self.lidar_points = self.LIDAR.points_map_coordinates

        self.waypoints = None
        self.car_state = None

    def set_waypoints(self, waypoints):
        self.waypoints =  np.array(waypoints).astype(np.float32)

    def set_car_state(self, car_state):
        self.car_state = np.array(car_state).astype(np.float32)

    def process_observation(self, ranges=None, ego_odom=None):

        self.LIDAR.load_lidar_measurement(ranges)
        self.lidar_points = self.LIDAR.get_processed_lidar_points_in_map_coordinates(
            self.car_state[POSE_X_IDX], self.car_state[POSE_Y_IDX], self.car_state[POSE_THETA_IDX]
        )

