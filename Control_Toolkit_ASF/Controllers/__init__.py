from abc import ABC, abstractmethod
import numpy as np

from utilities.lidar_utils import LidarHelper

class template_planner(ABC):
    def __init__(self):

        self.translational_control = None
        self.angular_control = None
        self.friction_value = None

        self.LIDAR = LidarHelper()
        self.lidar_points = self.LIDAR.points_map_coordinates

        self.waypoints = None
        self.nearest_waypoint_index = None

        self.car_state = None

        self.simulation_index = 0
        self.time = 0.0



    def set_waypoints(self, waypoints):
        self.waypoints =  np.array(waypoints).astype(np.float32)

    def set_car_state(self, car_state):
        self.car_state = np.array(car_state).astype(np.float32)

    @abstractmethod
    def process_observation(self, ranges=None, ego_odom=None):
        pass

    def pass_data_to_planner(self, next_interpolated_waypoints=None, car_state=None, obstacles=None):
        # Pass data to the planner
        if hasattr(self, 'set_waypoints'):
            self.set_waypoints(next_interpolated_waypoints)
        if hasattr(self, 'set_car_state'):
            self.set_car_state(car_state)
        if hasattr(self, 'set_obstacles'):
            self.set_obstacles(obstacles)



