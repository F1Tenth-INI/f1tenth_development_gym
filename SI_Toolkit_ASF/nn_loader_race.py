import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib

from types import SimpleNamespace

from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function

try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.Functions.TF.Compile import CompileTF

from SI_Toolkit.computation_library import TensorFlowLibrary

from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    odometry_dict_to_state
)


PATH_TO_MODELS = 'SI_Toolkit_ASF/Experiments/Ultra_multimap/Models/' ###default model length

LEN_INPUTS_POSE = 12 ##currently not used for slip steer only some pose predictors


class NeuralNetImitatorPlannerNV:

    def __init__(self, name, PATH_TO_MODELS=PATH_TO_MODELS, speed_fraction=1, batch_size=1):

        self.lib = TensorFlowLibrary()

        print('Loading NeuralNetImitatorPlanner')

        self.translational_control = None
        self.angular_control = None

        self.simulation_index = 0

        self.car_state = None

        a = SimpleNamespace()
        self.batch_size = batch_size  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented

        a.path_to_models = PATH_TO_MODELS

        a.net_name = name
        a.wash_out_len = 100
        a.post_wash_out_len = 50
        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.net_input_length = len(self.net_info.inputs)
        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.normalize_inputs = get_normalization_function(self.normalization_info, self.net_info.inputs, self.lib)
        self.denormalize_outputs = get_denormalization_function(self.normalization_info, self.net_info.outputs, self.lib)

        self.net_input = None
        self.net_input_normed = tf.Variable(
            tf.zeros([len(self.net_info.inputs),], dtype=tf.float32))

        self.input_buffer = []

        if self.net_input_length == 14:
            self.input_list = ['angular_vel_z', 'linear_vel_x', 'pose_theta_cos', 'pose_theta_sin']
        else:
            self.input_list = ['angular_vel_z', 'linear_vel_x', 'pose_theta_cos', 'pose_theta_sin', 'pose_x', 'pose_y']


    def render(self, e):
        return


    def process_observation2(self, ranges=None, ego_odom=None):

        net_input = tf.convert_to_tensor(ranges, tf.float32)

        net_output = self.process_tf(net_input)

        net_output = net_output.numpy()

        out_1 = float(net_output[0])
        out_2 = float(net_output[1])

        return out_1, out_2


    def process_observation1(self, ranges=None, ego_odom=None):

        net_input = tf.convert_to_tensor(ranges, tf.float32)

        net_output = self.process_tf(net_input)

        net_output = net_output.numpy()

        out = float(net_output)

        return out


    def get_slip_steer_car_state(self, slip_estimator, odom, t_control, a_control):

        self.input_buffer.extend([odom[x] for x in self.input_list])
        odometry = odometry_dict_to_state(odom)

        if len(self.input_buffer) == (self.net_input_length - 2):
            self.input_buffer.extend([t_control, a_control])

            steer_estimation = self.process_observation1(self.input_buffer) ###there is maybe a better way of doing this
            self.input_buffer.append(steer_estimation)
            slip_estimation = slip_estimator.process_observation1(self.input_buffer)

            ###clean up input vectors for next loop -> put this in a seperate update input buffer list function!
            del self.input_buffer[:len(self.input_list)]
            del self.input_buffer[-3:]

            odometry = np.append(odometry, [slip_estimation, steer_estimation])
            return odometry
        else:
            odometry = np.append(odometry, [0.0, 0.0])
            return odometry


    def downsample_lidar(self, lidar):
        #ranges_to_save = ranges
        lidar_downsampled = lidar[200:880:10]
            # ranges_to_save = ranges_to_save[::10]
        return lidar_downsampled



    def show_slip_steer_results(self, x, real_slip_vec, est_slip_vec, real_steer_vec, est_steer_vec):
        # fig.suptitle(NET_NAME + ' Oschersleben')
        fig, axs = plt.subplots(2)
        print(real_slip_vec)
        print(est_slip_vec)
        axs[0].plot(x, real_slip_vec, linewidth=1.25)
        axs[0].plot(x, est_slip_vec, linewidth=0.6)
        # axs[0].set_ylabel(OUTPUTS_LIST[0] + ' [m]')
        axs[0].set_xlabel('Timesteps (0.03s)')
        axs[0].legend(['ground truth', 'estimation'])
        # axs[0].set_ylim((None, 1.75))

        axs[1].plot(x, real_steer_vec, linewidth=1.25)
        axs[1].plot(x, est_steer_vec, linewidth=0.6)
        # axs[1].set_ylabel(OUTPUTS_LIST[1] + ' [m]')
        axs[1].set_xlabel('Timesteps (0.03s)')
        axs[1].legend(['ground truth', 'estimation'])
        # axs[1].set_ylim((-0.25, 0.25))
        plt.tight_layout()
        plt.show()


    @CompileTF
    def process_tf(self, net_input):
        # self.net_input_normed.assign(self.normalize_inputs(
        #     net_input
        # ))

        norm_inp = self.normalize_inputs(net_input)
        net_input = (tf.reshape(norm_inp, [-1, 1, len(self.net_info.inputs)]))

        net_output = self.net(net_input)

        net_output = self.denormalize_outputs(net_output)

        return tf.squeeze(net_output)


    @staticmethod
    def not_fun_sort(o):
        sorted = [o[19], o[12], o[6], o[0], o[13], o[7], o[1], o[14], o[8], o[2], o[15],
                  o[9], o[3], o[16], o[10], o[4], o[17], o[11], o[5], o[18]]
        return sorted


    @staticmethod
    def lidar_sort(o):
        sorted = [o[0], o[1], o[10], o[11], o[12], o[13], o[14], o[15], o[16],
                  o[17], o[18], o[19], o[2], o[20], o[21], o[22], o[23], o[24], o[25],
                  o[26], o[27], o[28], o[29], o[3], o[30], o[31], o[32], o[33],
                  o[34], o[35], o[36], o[37], o[38], o[39], o[4], o[40], o[41], o[42],
                  o[43], o[44], o[45], o[46], o[47], o[48], o[49], o[5], o[50],
                  o[51], o[52], o[53], o[54], o[55], o[56], o[57], o[58],
                  o[59], o[6], o[60], o[61], o[62], o[63], o[64], o[65], o[66],
                  o[67], o[7], o[8], o[9]]
        return sorted




