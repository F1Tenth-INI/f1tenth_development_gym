import sklearn # Don't touch
# sklearn is needed later, need to import it here for nni planner to work on nvidia jetson:
# https://forums.developer.nvidia.com/t/sklearn-skimage-cannot-allocate-memory-in-static-tls-block/236960
import numpy as np
import tensorflow as tf
import yaml
import os
from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import *
import time

from types import SimpleNamespace

from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from Control_Toolkit_ASF.Controllers import template_planner

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.TF.Compile import CompileTF

from SI_Toolkit.computation_library import TensorFlowLibrary


NET_NAME = Settings.NET_NAME
PATH_TO_MODELS = Settings.PATH_TO_MODELS

class NeuralNetImitatorPlanner(template_planner):

    def __init__(self, speed_fraction=1, batch_size=1):

        super().__init__()

        self.lib = TensorFlowLibrary

        print('Loading NeuralNetImitatorPlanner')

        self.translational_control = None
        self.angular_control = None

        self.simulation_index = 0

        self.waypoint_utils = None  # Will be overwritten with a WaypointUtils instance from car_system

        # self.waypoint_utils = WaypointUtils()  #Necessary for the recording of Waypoints in the the CSV file
        #                                        # !!Attention!! same number of waypoints to ignore as in config.yml is used -> set config to what was used during data collection


        a = SimpleNamespace()
        self.batch_size = batch_size  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented

        a.path_to_models = PATH_TO_MODELS
        a.net_name = NET_NAME

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.normalize_inputs = get_normalization_function(self.normalization_info, self.net_info.inputs, self.lib)
        self.denormalize_outputs = get_denormalization_function(self.normalization_info, self.net_info.outputs, self.lib)

        self.net_input = None
        self.net_input_normed = tf.Variable(
            tf.zeros([len(self.net_info.inputs),], dtype=tf.float32))
    
        self.config_training_NN = yaml.load(open(os.path.join(PATH_TO_MODELS, NET_NAME, "config_training.yml")), Loader=yaml.FullLoader)
        

    def process_observation(self, ranges=None, ego_odom=None):

        # FIXME: This stays as a reminder that you need to test LidarHelper for it and enforcing using this instead.
        # for index, range in enumerate(ranges):
        #     if(range < 0.1): range = 10
        ranges[0] = ranges[2]
        ranges[1] = ranges[2]
        ranges[-1] = ranges[-3]
        ranges[-2] = ranges[-3]

        self.LIDAR.load_lidar_measurement(ranges)

        if Settings.LIDAR_CORRUPT:
            self.LIDAR.corrupt_lidar_set_indices()
            self.LIDAR.corrupt_scans()
            self.LIDAR.corrupted_scans_high2zero()

        self.LIDAR.plot_lidar_data()
       
        #finding number of next waypoints divided in WYPT_X and WYPT_Y as defined in config_training of Model.
        config_inputs = np.append(self.config_training_NN["training_default"]["state_inputs"], self.config_training_NN["training_default"]["control_inputs"])
        number_of_next_waypoints = 0
        for element in config_inputs:
            if "WYPT_REL_X" in element:
                number_of_next_waypoints += 1


        #Loading next n wypts using waypoint_utils.py
        self.waypoint_utils.look_ahead_steps = number_of_next_waypoints # Do at init? 
        
        # The NNI planner needs relativa waypoints in any case
        next_waypoints = WaypointUtils.get_relative_positions(self.waypoints, self.car_state)

        #Split up Waypoint Tuples into WYPT_X and WYPT_Y because Network used this format in training from CSV
        next_waypoints_x = next_waypoints[:,0]
        next_waypoints_y = next_waypoints[:,1]

        #Load Waypoint Velocities for next n (defined in Settings) waypoints
        next_waypoint_vx = self.waypoints[:,WP_VX_IDX] 

        
        
        #In training all inputs are ordered alphabetically according to their index -> first LIDAR, then WYPTS, then States (because not capital letters)
        #Example of all possible inputs in correct order:
        # Order has to stay the same: SAME AS IN Config_training
        # If we want to change, look at recording
        #input_data = np.concatenate((ranges, next_waypoints_x, next_waypoints_y,
        #                              [self.car_state[ANGULAR_VEL_Z_IDX], self.car_state[LINEAR_VEL_X_IDX],
        #                              self.car_state[POSE_THETA_COS_IDX], self.car_state[POSE_THETA_SIN_IDX],
        #                              self.car_state[POSE_X_IDX], self.car_state[POSE_Y_IDX]]), axis=0)
        
        #Current Input:
        input_data = np.concatenate((self.LIDAR.processed_scans, next_waypoints_x, next_waypoints_y, next_waypoint_vx,
                                      [self.car_state[ANGULAR_VEL_Z_IDX], self.car_state[LINEAR_VEL_X_IDX], self.car_state[SLIP_ANGLE_IDX], self.car_state[STEERING_ANGLE_IDX]]), axis=0)

        net_input = tf.convert_to_tensor(input_data, tf.float32)

        net_output = self.process_tf(net_input)

        net_output = net_output.numpy()

        angular_control = float(net_output[0])
        translational_control = float(net_output[1])


        # Accelerate at the beginning "Schupf" (St model explodes for small velocity) -> must come after loading of waypoints otherwise they aren't saved
        if self.simulation_index < Settings.ACCELERATION_TIME:
            self.simulation_index += 1
            self.translational_control = Settings.ACCELERATION_AMPLITUDE
            self.angular_control = 0
            return self.angular_control, self.translational_control,

        self.translational_control = 1.0 * translational_control
        self.angular_control = angular_control


        return angular_control, translational_control 

    @CompileTF 
    def process_tf(self, net_input):

        self.net_input_normed.assign(self.normalize_inputs(
            net_input
        ))

        net_input = (tf.reshape(self.net_input_normed, [-1, 1, len(self.net_info.inputs)]))

        net_output = self.net(net_input)

        net_output = self.denormalize_outputs(net_output)

        return tf.squeeze(net_output)
        

