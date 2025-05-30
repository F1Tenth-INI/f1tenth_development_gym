from typing import Optional

from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils
from utilities.lidar_utils import *
from utilities.controller_utilities import ControllerUtilities

from Control_Toolkit_ASF.Controllers import template_planner

# from TrainingLite.mpc_immitator_mu.predict import predict_next_control
from TrainingLite.mpc_immitator_mu.torch_predict import ControlPredictor
# from TrainingLite.slip_prediction.predict import predict_slip_angle
# from TrainingLite.pp_immitator.predict import predict_next_control

class SysIdPlanner(template_planner):

    def __init__(self):

        print('Loading SysIdPlanner')

        super().__init__()

        self.car_state = None
        self.LIDAR = Optional[LidarHelper]
        self.car_state_history = []
        self.render_utils = RenderUtils()

        self.controller_utils = ControllerUtilities()
        
        Settings.RECORDING_FOLDER = 'ExperimentRecordings/Sysid/'
        Settings.RECORDING_NAME = 'TestRecording'
        
        
        

        
        self.turning = False
        
        self.waypoint_utils = None  # Will be overwritten with a WaypointUtils instance from car_system
        
        self.angular_control = 0
        self.translational_control = 0
        
        self.control_index = 0
        

    def process_observation(self, ranges=None, ego_odom=None):
        
        
        self.angular_control = 0.
        desired_speed = 6.        
        desired_angle = 0.4 
        current_speed = self.car_state[LINEAR_VEL_X_IDX]
        # print('Speed:', current_speed)
        
        if(current_speed > desired_speed - 0.5):
            if self.turning == False:
                print(f"Initiating turn with angle: {desired_angle} at speed: {current_speed} / {desired_speed}")
            self.turning = True
        if(current_speed < 0.5):
            self.turning = False
            
            
        if(self.turning == 1):
            self.angular_control = desired_angle
            
            
        acceleration = self.controller_utils.motor_pid(desired_speed, current_speed)
        self.translational_control = acceleration
        
        self.control_index += 1
        return self.angular_control, self.translational_control
        

