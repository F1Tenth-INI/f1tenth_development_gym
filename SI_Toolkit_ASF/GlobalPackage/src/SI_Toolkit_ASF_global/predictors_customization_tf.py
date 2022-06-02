import tensorflow as tf

from SI_Toolkit.TF.TF_Functions.Compile import Compile

from SI_Toolkit_ASF_global.predictors_customization import POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, SPEED_IDX, STEERING_IDX
from SI_Toolkit_ASF_global.predictors_customization import STATE_INDICES

class next_state_predictor_ODE_tf():

    def __init__(self, dt, intermediate_steps, disable_individual_compilation=False):
        self.s = None

        self.intermediate_steps = tf.convert_to_tensor(intermediate_steps, dtype=tf.int32)
        self.intermediate_steps_float = tf.convert_to_tensor(intermediate_steps, dtype=tf.float32)
        self.t_step = tf.convert_to_tensor(dt / float(self.intermediate_steps), dtype=tf.float32)

        if disable_individual_compilation:
            self.step = self._step
        else:
            self.step = Compile(self._step)

    @Compile
    def _step(self, s, Q, params):

        pose_theta = s[:, POSE_THETA_IDX]
        pose_x = s[:, POSE_X_IDX]
        pose_y = s[:, POSE_Y_IDX]

        speed = Q[:, SPEED_IDX]
        steering = Q[:, STEERING_IDX]
        linear_vel_x = s[:, LINEAR_VEL_X]

        for _ in tf.range(self.intermediate_steps):
            pose_theta = pose_theta + 0.5*(steering/self.intermediate_steps_float)
            pose_x = pose_x + self.t_step * speed * tf.math.cos(pose_theta)
            pose_y = pose_y + self.t_step * speed * tf.math.sin(pose_theta)

        pose_theta_cos = tf.math.cos(pose_theta)
        pose_theta_sin = tf.math.sin(pose_theta)

        s_next = tf.stack([tf.zeros_like(pose_x), linear_vel_x, tf.zeros_like(pose_x),
                           pose_theta, pose_theta_cos, pose_theta_sin,
                           pose_x, pose_y], axis=1)

        return s_next


class predictor_output_augmentation_tf:
    def __init__(self, net_info, disable_individual_compilation=False):
        self.net_output_indices = {key: value for value, key in enumerate(net_info.outputs)}
        indices_augmentation = []
        features_augmentation = []
        if 'angular_vel_z' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['angular_vel_z'])
            features_augmentation.append('angular_vel_z')
        if 'linear_vel_x' not in net_info.outputs:
            indices_augmentation.append(STATE_INDICES['linear_vel_x'])
            features_augmentation.append('linear_vel_x')
        if 'linear_vel_y' not in net_info.outputs:
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
            self.augment = Compile(self._augment)

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
                    net_output[..., self.index_pose_theta_cos])[:, :, tf.newaxis]  # tf.math.atan2 removes the features (last) dimension, so it is added back with [:, :, tf.newaxis]
            output = tf.concat([output, pose_theta], axis=-1)

        if 'angle_sin' in self.features_augmentation:
            pose_theta_sin = \
                tf.sin(net_output[..., self.index_pose_theta])[:, :, tf.newaxis]
            output = tf.concat([output, pose_theta_sin], axis=-1)

        if 'angle_cos' in self.features_augmentation:
            pose_theta_cos = \
                tf.cos(net_output[..., self.index_pose_theta])[:, :, tf.newaxis]
            output = tf.concat([output, pose_theta_cos], axis=-1)

        return output
