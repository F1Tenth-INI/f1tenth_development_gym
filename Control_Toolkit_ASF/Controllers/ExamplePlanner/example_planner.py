from typing import Optional

from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils
from utilities.lidar_utils import *

from Control_Toolkit_ASF.Controllers import template_planner

# from TrainingLite.mpc_immitator_mu.predict import predict_next_control
from TrainingLite.mpc_immitator_mu.torch_predict import ControlPredictor
# from TrainingLite.slip_prediction.predict import predict_slip_angle
# from TrainingLite.pp_immitator.predict import predict_next_control

class ExamplePlanner(template_planner):

    def __init__(self):

        print('Loading ExamplePlanner')

        super().__init__()


        # Utilities and car_state will be overwritten by the car system. 
        # All relevant data can be accessed from them
        self.car_state = None
        self.lidar_utils = Optional[LidarHelper]
        self.waypoint_utils = Optional[WaypointUtils]
        self.render_utils = Optional[RenderUtils]
        
        self.angular_control = 0
        self.translational_control = 0
        

    # This function is called every control step
    # State and sensor data is already updated in the car system
    def process_observation(self):
        
        
        waypoints = self.waypoint_utils.next_waypoints
        scans = self.lidar_utils.processed_ranges
        print(f'Example planners first waypoint: {waypoints[0]} ')
        print(f'Example planners first scan: {scans[0]} ')
        
        self.angular_control = 0
        self.translational_control = 1
        return self.angular_control, self.translational_control
        

