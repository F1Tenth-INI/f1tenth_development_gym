import tensorflow as tf
from types import SimpleNamespace
import copy

from SI_Toolkit.computation_library import TensorFlowLibrary

from utilities.Settings import Settings
from SI_Toolkit.Functions.TF.Network import compose_net_from_net_name, compose_net_from_module
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net


class TIVModel(tf.keras.Model):
    def __init__(self, time_series_length, batch_size, net_info=None, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dt = Settings.TIMESTEP_PLANNER
        self.batch_size = batch_size
        self.lib = TensorFlowLibrary
        self.net_info = net_info
        self.time_series_length = time_series_length
        
        self.normalization_info = get_norm_info_for_net(net_info, copy_files=False)
        self.denormalize_pose_theta = get_denormalization_function(self.normalization_info, ['pose_theta'], self.lib)
        self.denormalize_car_pose_x_y = get_denormalization_function(self.normalization_info, ['D_car_pose_x', 'D_car_pose_y'], self.lib)
        self.normalize_pose_x_y = get_normalization_function(self.normalization_info, ['D_pose_x', 'D_pose_y'], self.lib)
        
    def setup_net(self, net_specification):
        nn_net_info = copy.deepcopy(self.net_info)
        nn_net_info.net_name = net_specification
        nn_net_info.inputs = [s for s in self.net_info.inputs if 'pose' not in s]
        nn_net_info.outputs = self.net_info.outputs

        self.net, nn_net_info = compose_net_from_net_name(
            nn_net_info,
            self.time_series_length
        )

    def remove_pose_information(self, state):
        ''' We want to remove all pose information (x, y, theta)'''
        indices = [i for i, s in enumerate(self.net_info.inputs) if 'pose' not in s]
        output = tf.gather(state, indices, axis=2)
        return output

    def rotate_tensor(self, rotation_tuple):
        (rotation_matrix, tensor) = rotation_tuple
        out = tf.tensordot(rotation_matrix, tensor, 1)  # This seems to work
        return out

    def convert_tiv_delta_to_full_state_delta(self, out_tiv, pose_theta):
        ''' We want to convert the delta values in the car coordinate frame to a global coordinate frame.
        This is a rotation of D_pose_x and y for an angle of theta.

        @param output_tiv: Output of net, state (D_angular_vel_z, D_linear_vel_x, D_pose_theta_cos, D_pose_theta_sin, pose_x, pose_y, slip_angle, steering_angle)
            where the pose values are in the car coordinate frame
        '''
        theta_cos = tf.math.cos(pose_theta)
        theta_sin = tf.math.sin(pose_theta)
        out_tiv = out_tiv[:, -1, :]

        rotation_matrices = tf.stack((tf.stack((theta_cos, theta_sin), axis=1), tf.stack((-theta_sin, theta_cos), axis=1)), axis=2)

        out_tiv_split = tf.split(out_tiv, [3, 2, 2], axis=1)  # TODO: Split this according to the net_input.inputs
        pose_x_y = self.denormalize_car_pose_x_y(out_tiv_split[1])
        # pose_x_y = out_tiv_split[1]
        pose_global = tf.map_fn(self.rotate_tensor, (rotation_matrices, pose_x_y), fn_output_signature=tf.float32)
        out_tiv_split[1] = self.normalize_pose_x_y(pose_global)
        # out_tiv_split[1] = pose_global
        output = tf.concat(out_tiv_split, axis=1)
        output = tf.expand_dims(output, 1)

        # tf.print()

        return output

    def call(self, x, training=None, mask=None):
        '''
        x: Full state [[['angular_control', 'translational_control', 'angular_vel_z', 'linear_vel_x', 'pose_theta', 'slip_angle', 'steering_angle']]]
        '''
        state_tiv = self.remove_pose_information(x)
        out_tiv = self.net(state_tiv)  # [batch_size, 1, states]

        pose_theta_idx = self.net_info.inputs.index('pose_theta')
        pose_theta = self.denormalize_pose_theta(x[:, -1, pose_theta_idx])
        out_tiv = self.convert_tiv_delta_to_full_state_delta(out_tiv, pose_theta)
        return out_tiv


class TIVModelDNN(TIVModel):
    def __init__(self, time_series_length, batch_size, net_info=None, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info, name, **kwargs)
        self.setup_net('Dense-64H1-128H2-64H3')


class TIVModelLSTM(TIVModel):
    def __init__(self, time_series_length, batch_size, net_info=None, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info, name, **kwargs)
        self.setup_net('LSTM-32H1-64H2-32H3')





class TIVModelRealCar(tf.keras.Model):
    def __init__(self, time_series_length, batch_size, net_info=None, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dt = Settings.TIMESTEP_PLANNER
        self.batch_size = batch_size
        self.lib = TensorFlowLibrary
        self.net_info = net_info
        self.time_series_length = time_series_length
        
        self.normalization_info = get_norm_info_for_net(net_info, copy_files=False)
        self.denormalize_pose_theta = get_denormalization_function(self.normalization_info, ['pose_theta'], self.lib)
        self.denormalize_car_pose_x_y = get_denormalization_function(self.normalization_info, ['D_car_pose_x', 'D_car_pose_y'], self.lib)
        self.normalize_pose_x_y = get_normalization_function(self.normalization_info, ['D_pose_x', 'D_pose_y'], self.lib)
        
    def setup_net(self, net_specification):
        nn_net_info = copy.deepcopy(self.net_info)
        nn_net_info.net_name = net_specification
        nn_net_info.inputs = ['angular_control', 'translational_control', 'angular_vel_z', 'linear_vel_x', 'steering_angle']
        nn_net_info.outputs = ['D_angular_vel_z', 'D_linear_vel_x', 'D_pose_theta', 'D_pose_x', 'D_pose_y', 'D_steering_angle']

        self.net, nn_net_info = compose_net_from_net_name(
            nn_net_info,
            self.time_series_length
        )

    def remove_pose_information(self, state):
        ''' We want to remove all pose information (x, y, theta)'''
        output = tf.gather(state, [0, 1, 2, 3, 5], axis=2)
        return output

    def rotate_tensor(self, rotation_tuple):
        (rotation_matrix, tensor) = rotation_tuple
        out = tf.tensordot(rotation_matrix, tensor, 1)  # This seems to work
        return out

    def convert_tiv_delta_to_full_state_delta(self, out_tiv, pose_theta):
        ''' We want to convert the delta values in the car coordinate frame to a global coordinate frame.
        This is a rotation of D_pose_x and y for an angle of theta.

        @param output_tiv: Output of net, state (D_angular_vel_z, D_linear_vel_x, D_pose_theta_cos, D_pose_theta_sin, pose_x, pose_y, slip_angle, steering_angle)
            where the pose values are in the car coordinate frame
        '''
        theta_cos = tf.math.cos(pose_theta)
        theta_sin = tf.math.sin(pose_theta)
        out_tiv = out_tiv[:, -1, :]

        rotation_matrices = tf.stack((tf.stack((theta_cos, theta_sin), axis=1), tf.stack((-theta_sin, theta_cos), axis=1)), axis=2)

        out_tiv_split = tf.split(out_tiv, [3, 2, 1], axis=1)
        # tf.print(out_tiv_split[1], summarize=-1)
        pose_x_y = self.denormalize_car_pose_x_y(out_tiv_split[1])
        # tf.print(pose_x_y, summarize=-1)
        # tf.print(pose_theta, summarize=-1)
        # pose_x_y = out_tiv_split[1]
        pose_global = tf.map_fn(self.rotate_tensor, (rotation_matrices, pose_x_y), fn_output_signature=tf.float32)
        # tf.print(pose_global, summarize=-1)
        out_tiv_split[1] = self.normalize_pose_x_y(pose_global)
        # tf.print(out_tiv_split[1], summarize=-1)
        # out_tiv_split[1] = pose_global
        output = tf.concat(out_tiv_split, axis=1)
        output = tf.expand_dims(output, 1)

        # tf.print()

        return output

    def call(self, x, training=None, mask=None):
        '''
        x: Full state [[['angular_control', 'translational_control', 'angular_vel_z', 'linear_vel_x', 'pose_theta', 'slip_angle', 'steering_angle']]]
        '''
        state_tiv = self.remove_pose_information(x)
        out_tiv = self.net(state_tiv)  # [batch_size, 1, states]

        pose_theta = self.denormalize_pose_theta(x[:, -1, 4])
        out_tiv = self.convert_tiv_delta_to_full_state_delta(out_tiv, pose_theta)
        return out_tiv


class TIVModelDNNRealCar(TIVModelRealCar):
    def __init__(self, time_series_length, batch_size, net_info=None, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info, name, **kwargs)
        self.setup_net('Dense-64H1-128H2-64H3')


class TIVModelLSTMRealCar(TIVModelRealCar):
    def __init__(self, time_series_length, batch_size, net_info=None, name=None, **kwargs):
        super().__init__(time_series_length, batch_size, net_info, name, **kwargs)
        self.setup_net('LSTM-32H1-64H2-32H3')



if __name__ == '__main__':
    model = TIVModel(1, 1)
    # ['angular_control', 'translational_control', 'angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
    state_tiv = tf.constant([[[0.0, 4.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 4.0, 0.0, 4.0, 1.57, 1.0, 0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 4.0, 0.0, 4.0, 3.14, 1.0, 0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 4.0, 0.0, 4.0, 4.71, 1.0, 0.0, 0.0, 0.0, 0.0]]])
    # ['D_angular_vel_z', 'D_linear_vel_x', 'D_pose_theta_cos', 'D_pose_theta_sin', 'D_pose_x', 'D_pose_y', 'D_slip_angle', 'D_steering_angle'],
    out_tiv = tf.constant([[[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                           [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                           [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                            [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    print('Output is:', model.convert_tiv_delta_to_full_state_delta(out_tiv, state_tiv))
    print()
