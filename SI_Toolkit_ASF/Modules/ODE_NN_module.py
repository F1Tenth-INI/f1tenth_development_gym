import tensorflow as tf
from types import SimpleNamespace

from utilities.Settings import Settings
from SI_Toolkit_ASF.Modules.ODE_module import KSModel
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.TF.Network import compose_net_from_net_name
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net


class ODE_DNN(tf.keras.Model):
    def __init__(self, time_series_length, batch_size, net_info=None, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dt = Settings.TIMESTEP_PLANNER
        self.intermediate_steps = 4
        self.batch_size = batch_size
        self.time_series_length = time_series_length
        self.lib = TensorFlowLibrary
        self.car_model = KSModel(time_series_length, batch_size, name, **kwargs)
        self.setup_net()

        self.normalization_info = get_norm_info_for_net(net_info, copy_files=False)
        states = ['D_angular_vel_z', 'D_linear_vel_x', 'D_pose_theta_cos', 'D_pose_theta_sin', 'D_pose_x', 'D_pose_y', 'D_slip_angle', 'D_steering_angle']
        self.denormalize = get_denormalization_function(self.normalization_info, states, self.lib)
        self.normalize = get_normalization_function(self.normalization_info, states, self.lib)

    def setup_net(self):
        net_info = SimpleNamespace()
        net_info.net_name = 'Dense-64H1-128H2-64H3'
        net_info.inputs = ['D_angular_vel_z', 'D_linear_vel_x', 'D_pose_theta_cos', 'D_pose_theta_sin', 'D_pose_x', 'D_pose_y', 'D_slip_angle', 'D_steering_angle']
        net_info.outputs = ['D_angular_vel_z', 'D_linear_vel_x', 'D_pose_theta_cos', 'D_pose_theta_sin', 'D_pose_x', 'D_pose_y', 'D_slip_angle', 'D_steering_angle']

        self.net, nn_net_info = compose_net_from_net_name(
            net_info,
            self.time_series_length
        )

    def setup_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(batch_size=self.batch_size, shape=(self.time_series_length, len(self.net_info.inputs))))
        self.model.add(self.car_model)
        self.model.add(self.net)
        

    def remove_pose_theta(self, state):
        return tf.gather(state, [0, 1, 3, 4, 5, 6, 7, 8], axis=2)

    def calculate_derivative(self, x_old, x_new, dt):
        return (x_new - x_old) / dt

    def call(self, x, training=None, mask=None):
        s = x[:, :, 2:]

        car_output = self.car_model(x)  # Third argument params not implemented, returns ['D_angular_vel_z', 'D_linear_vel_x', 'D_pose_theta', 'D_pose_theta_cos', 'D_pose_theta_sin', 'D_pose_x', 'D_pose_y', 'D_slip_angle', 'D_steering_angle']
        delta_output = self.calculate_derivative(s, car_output, self.dt)
        output_short = self.remove_pose_theta(delta_output)
        output_norm = self.normalize(output_short)
        output = self.net(output_norm)
        output_denorm = self.denormalize(output)
        return output_denorm
