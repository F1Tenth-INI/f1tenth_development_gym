from abc import ABC, abstractmethod
import numpy as np

from typing import Any, Dict, Optional
from utilities.lidar_utils import LidarHelper
from utilities.render_utilities import RenderUtils
from utilities.waypoint_utils import WaypointUtils


class template_planner(ABC):
    def __init__(self):


        self.translational_control = None
        self.angular_control = None


        # Initialized by the driver (rendering / utilities only)
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

    @staticmethod
    def get_car_obs(controller_observation: Dict[str, Any]) -> Dict[str, Any]:
        driver_keys = (
            "car_state", "scans", "sensors", "env",
            "collision", "terminated", "interrupted", "done", "info",
        )
        return {k: controller_observation[k] for k in driver_keys if k in controller_observation}

    @staticmethod
    def get_car_state(controller_observation: Dict[str, Any]) -> np.ndarray:
        return np.asarray(controller_observation["car_state"], dtype=np.float32)

    @abstractmethod
    def process_observation(self, controller_observation: Dict[str, Any]):
        pass


