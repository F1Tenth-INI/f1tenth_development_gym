from abc import ABC, abstractmethod
import numpy as np

from typing import Optional
from utilities.lidar_utils import LidarHelper
from utilities.render_utilities import RenderUtils
from utilities.waypoint_utils import WaypointUtils


class template_planner(ABC):
    def __init__(self):


        self.translational_control = None
        self.angular_control = None


        # Initialized by the driver
        self.car_state = None
        self.waypoint_utils: Optional[WaypointUtils] = None
        self.lidar_utils: Optional[LidarHelper] = None
        self.render_utils: Optional[RenderUtils] = None

        self.simulation_index = 0
        self.time = 0.0

    def reset(self):
        self.simulation_index = 0
        self.time = 0.0
        self.translational_control = 0.0
        self.angular_control = 0.0

    def set_car_state(self, car_state):
        self.car_state = np.array(car_state).astype(np.float32)

    @abstractmethod
    def process_observation(self):
        pass



