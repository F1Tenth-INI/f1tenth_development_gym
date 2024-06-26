import sklearn # Don't touch
# sklearn is needed later, need to import it here for nni planner to work on nvidia jetson:
# https://forums.developer.nvidia.com/t/sklearn-skimage-cannot-allocate-memory-in-static-tls-block/236960

from utilities.waypoint_utils import *

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit.Controllers.controller_neural_imitator import controller_neural_imitator

from TrainingLite.mpc_immitator.predict import predict_next_control

class NNLitePlanner(template_planner):

    def __init__(self, speed_fraction=1, batch_size=1):

        print('Loading NNLitePlanner')

        super().__init__()

        self.simulation_index = 0
        self.car_state = None
        self.waypoints = None
        
        self.waypoint_utils = None  # Will be overwritten with a WaypointUtils instance from car_system
        
        self.angular_control = 0
        self.translational_control = 0


    

    def process_observation(self, ranges=None, ego_odom=None):
        
        # Waypoint dict
        waypoints_relative = WaypointUtils.get_relative_positions(self.waypoints, self.car_state)
        waypoints_relative_x = waypoints_relative[:, 0]
        waypoints_relative_y = waypoints_relative[:, 1]
        next_waypoint_vx = self.waypoints[:, WP_VX_IDX]
        
        controls = predict_next_control(self.car_state, waypoints_relative, self.waypoints)
        control = controls[0]
        
        # print("control", control)
        
        self.angular_control = control[0]
        self.translational_control = control[1]

        # Accelerate at the beginning "Schupf" (St model explodes for small velocity) -> must come after loading of waypoints otherwise they aren't saved
        if self.simulation_index < Settings.ACCELERATION_TIME:
            self.simulation_index += 1
            self.translational_control = Settings.ACCELERATION_AMPLITUDE
            self.angular_control = 0
            return self.angular_control, self.translational_control,


        return self.angular_control, self.translational_control
        

