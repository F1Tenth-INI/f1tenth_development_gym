import yaml
import tensorflow as tf

from SI_Toolkit.Functions.TF.Compile import CompileTF

from utilities.state_utilities import *

from utilities.Settings import Settings


class next_state_predictor_ODE_tf():

    def __init__(self, dt, intermediate_steps, batch_size=1, disable_individual_compilation=False, planning_environment=None):
        self.s = tf.convert_to_tensor(create_car_state())

        self.intermediate_steps = intermediate_steps
        self.t_step = dt / float(self.intermediate_steps)

        config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
        if Settings.SYSTEM == 'car':
            from SI_Toolkit_ASF.f1t_model import f1t_model
            self.env = f1t_model(dt=dt, intermediate_steps=intermediate_steps, **{**config['f1t_car_model'], **{
                "num_control_inputs": config["num_control_inputs"]}})  # Environment model, keeping car ODEs
        else:
            raise NotImplementedError('{} not yet implemented in next_state_predictor_ODE_tf'.format(Settings.SYSTEM))

        if disable_individual_compilation:
            self.step = self._step
        else:
            self.step = CompileTF(self._step)

    def _step(self, s, Q, params):

        s_next = self.env.step_dynamics(s, Q, params)
        return s_next



class predictor_output_augmentation_tf:
    def __init__(self, net_info, disable_individual_compilation=False, differential_network=False):
        self.net_output_indices = {key: value for value, key in enumerate(net_info.outputs)}
        indices_augmentation = []
        features_augmentation = []
        if 'angular_vel_z' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['angular_vel_z'])
            features_augmentation.append('angular_vel_z')
        if 'linear_vel_x' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['linear_vel_x'])
            features_augmentation.append('linear_vel_x')
        if 'linear_vel_y' not in net_info.outputs and 'linear_vel_y' in STATE_INDICES.keys():  # Quadruped only
            indices_augmentation.append(STATE_INDICES['linear_vel_y'])
            features_augmentation.append('linear_vel_y')

        if 'pose_theta' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['pose_theta'])
            features_augmentation.append('pose_theta')
        if 'pose_theta_cos' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['pose_theta_cos'])
            features_augmentation.append('pose_theta_cos')
        if 'pose_theta_sin' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['pose_theta_sin'])
            features_augmentation.append('pose_theta_sin')

        if 'slip_angle' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['slip_angle'])
            features_augmentation.append('slip_angle')
        if 'steering_angle' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['steering_angle'])
            features_augmentation.append('steering_angle')

        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

        if 'pose_theta' in net_info.outputs:
            self.index_pose_theta = tf.convert_to_tensor(self.net_output_indices['pose_theta'])
        if 'pose_theta_sin' in net_info.outputs:
            self.index_pose_theta_sin = tf.convert_to_tensor(self.net_output_indices['pose_theta_sin'])
        if 'pose_theta_cos' in net_info.outputs:
            self.index_pose_theta_cos = tf.convert_to_tensor(self.net_output_indices['pose_theta_cos'])

        if disable_individual_compilation:
            self.augment = self._augment
        else:
            self.augment = CompileTF(self._augment)

    def get_indices_augmentation(self):
        return self.indices_augmentation

    def get_features_augmentation(self):
        return self.features_augmentation

    def _augment(self, net_output):

        output = net_output
        if 'angular_vel_z' in self.features_augmentation:
            angular_vel_z = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, angular_vel_z], axis=-1)
        if 'linear_vel_x' in self.features_augmentation:
            linear_vel_x = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, linear_vel_x], axis=-1)
        if 'linear_vel_y' in self.features_augmentation:
            linear_vel_y = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, linear_vel_y], axis=-1)

        if 'pose_theta' in self.features_augmentation:
            pose_theta = tf.math.atan2(
                net_output[..., self.index_pose_theta_sin],
                net_output[..., self.index_pose_theta_cos])[:, :,
                         tf.newaxis]  # tf.math.atan2 removes the features (last) dimension, so it is added back with [:, :, tf.newaxis]
            output = tf.concat([output, pose_theta], axis=-1)

        if 'angle_sin' in self.features_augmentation:
            pose_theta_sin = \
                tf.sin(net_output[..., self.index_pose_theta])[:, :, tf.newaxis]
            output = tf.concat([output, pose_theta_sin], axis=-1)

        if 'angle_cos' in self.features_augmentation:
            pose_theta_cos = \
                tf.cos(net_output[..., self.index_pose_theta])[:, :, tf.newaxis]
            output = tf.concat([output, pose_theta_cos], axis=-1)

        if 'slip_angle' in self.features_augmentation:
            slip_angle = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, slip_angle], axis=-1)

        if 'steering_angle' in self.features_augmentation:
            steering_angle = tf.zeros_like(net_output[:, :, -1:])
            output = tf.concat([output, steering_angle], axis=-1)


        return output
