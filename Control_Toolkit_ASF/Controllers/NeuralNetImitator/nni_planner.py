import numpy as np
import tensorflow as tf
import yaml
import os
from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import WaypointUtils


from types import SimpleNamespace

from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.TF.Compile import CompileTF

from SI_Toolkit.computation_library import TensorFlowLibrary

#NET_NAME = 'GRU-104IN-32H1-32H2-2OUT-1'
NET_NAME = 'GRU-74IN-32H1-32H2-2OUT-1'
PATH_TO_MODELS = 'SI_Toolkit_ASF/Experiments/Experiment-MPPI-Imitator/Models/'

class NeuralNetImitatorPlanner:

    def __init__(self, speed_fraction=1, batch_size=1):

        self.lib = TensorFlowLibrary

        print('Loading NeuralNetImitatorPlanner')

        self.translational_control = None
        self.angular_control = None

        self.simulation_index = 0

        self.car_state = None
        self.waypoint_utils = WaypointUtils()  #Necessary for the recording of Waypoints in the the CSV file


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

    def render(self, e):
        return



    def process_observation(self, ranges=None, ego_odom=None):

        # Accelerate at the beginning (St model explodes for small velocity)
        if self.simulation_index < 20:
            self.simulation_index += 1
            self.translational_control = 10
            self.angular_control = 0
            return self.translational_control, self.angular_control

        if Settings.ONLY_ODOMETRY_AVAILABLE:
            s = odometry_dict_to_state(ego_odom)
        else:
            s = self.car_state


        #code for Lidar bounds and Lidar data reduction
        ranges = ranges[200:880]
        ranges = ranges[::10]

        #finding number of next waypoints divided in WYPT_X and WYPT_Y as defined in config_training of Model.
        config_training_NN = yaml.load(open(os.path.join(PATH_TO_MODELS, NET_NAME, "config_training.yml")), Loader=yaml.FullLoader)
        state_inputs = config_training_NN["training_default"]["state_inputs"]
        number_of_next_waypoints = 0
        for element in state_inputs:
            if "WYPT_X" in element:
                number_of_next_waypoints += 1


        #Loading next n wypts using waypoint_utils.py
        self.waypoint_utils.look_ahead_steps = number_of_next_waypoints
        car_position = [s[POSE_X_IDX], s[POSE_Y_IDX]]
        self.waypoint_utils.update_next_waypoints(car_position)
        next_waypoints = self.waypoint_utils.next_waypoint_positions

        #Split up Waypoint Tuples into WYPT_X and WYPT_Y because Network used this format in training from CSV
        next_waypoints_x = next_waypoints[:,0]
        next_waypoints_y = next_waypoints[:,1]


        #input_data = car_states + Lidar + next waypoints #ToDo append exactly what was listed as state inputs in config training and or CSV file of Model automatically instead of appending it manually
        input_data = np.concatenate(([self.car_state[POSE_THETA_COS_IDX], self.car_state[POSE_THETA_SIN_IDX], self.car_state[POSE_X_IDX], self.car_state[POSE_Y_IDX], self.car_state[LINEAR_VEL_X_IDX], self.car_state[ANGULAR_VEL_Z_IDX]], ranges, next_waypoints_x,next_waypoints_y), axis=0)
        net_input = tf.convert_to_tensor(input_data, tf.float32)

        net_output = self.process_tf(net_input)
        net_output = net_output.numpy()

        speed = float(net_output[0])
        steering_angle = float(net_output[1])

        self.translational_control = speed
        self.angular_control = steering_angle


        return speed, steering_angle

    @CompileTF
    def process_tf(self, net_input):

        self.net_input_normed.assign(self.normalize_inputs(
            net_input
        ))

        net_input = (tf.reshape(net_input, [-1, 1, len(self.net_info.inputs)]))

        net_output = self.net(net_input)

        net_output = self.denormalize_outputs(net_output)

        return tf.squeeze(net_output)
        

