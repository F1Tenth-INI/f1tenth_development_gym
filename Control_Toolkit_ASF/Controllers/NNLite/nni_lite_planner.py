import sklearn # Don't touch
# sklearn is needed later, need to import it here for nni planner to work on nvidia jetson:
# https://forums.developer.nvidia.com/t/sklearn-skimage-cannot-allocate-memory-in-static-tls-block/236960

from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit.Controllers.controller_neural_imitator import controller_neural_imitator

# from TrainingLite.mpc_immitator_mu.predict import predict_next_control
from TrainingLite.mpc_immitator_mu.torch_predict import predict_next_control
# from TrainingLite.slip_prediction.predict import predict_slip_angle
# from TrainingLite.pp_immitator.predict import predict_next_control

class NNLitePlanner(template_planner):

    def __init__(self, speed_fraction=1, batch_size=1):

        print('Loading NNLitePlanner')

        super().__init__()

        self.simulation_index = 0
        self.car_state = None
        self.waypoints = None
        self.imu_data = None
        
        self.car_state_history = []
        
        self.render_utils = RenderUtils()

        
        self.waypoint_utils = None  # Will be overwritten with a WaypointUtils instance from car_system
        
        self.angular_control = 0
        self.translational_control = 0
        
        self.control_index = 0
        
        self.predicted_frictions = []


    

    def process_observation(self, ranges=None, ego_odom=None):
        
        
        self.car_state_history.append(self.car_state)
        if len(self.predicted_frictions) > 100:
            self.car_state_history.pop(0)
        
        # Waypoint dict
        waypoints_relative = WaypointUtils.get_relative_positions(self.waypoints, self.car_state)
        waypoints_relative_x = waypoints_relative[:, 0]
        waypoints_relative_y = waypoints_relative[:, 1]
        next_waypoint_vx = self.waypoints[:, WP_VX_IDX]
        
        # slip_angle_predicted = predict_slip_angle(np.array(self.car_state), np.array(self.imu_data))[0]
        # print("slip_angle", slip_angle, self.car_state[SLIP_ANGLE_IDX])
        
        # Overwrite slip angle with predicted value
        # self.car_state[SLIP_ANGLE_IDX] = slip_angle_predicted
        # self.car_state[SLIP_ANGLE_IDX] = 0 
        
        
        controls = predict_next_control(self.car_state, waypoints_relative, self.waypoints)
        control = controls[0]
        
        # print("control", control)
        
        if len(control) == 3:

            predicted_friction = control[2]
            self.predicted_frictions.append(predicted_friction)
            
            if len(self.predicted_frictions) > Settings.AVERAGE_WINDOW:
                self.predicted_frictions.pop(0)
            average_friction = sum(self.predicted_frictions) / len(self.predicted_frictions)
            self.render_utils.set_label_dict({
                # '3: slip_angle_predicted': slip_angle_predicted,
                '5: predicted_friction': average_friction,
            })

        
        self.angular_control = control[0]
        self.translational_control = control[1]

        if self.control_index < 20:
            self.angular_control = 0
            self.translational_control = 2

    
        self.control_index += 1
        return self.angular_control, self.translational_control
        

