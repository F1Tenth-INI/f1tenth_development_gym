import tensorflow as tf

from SI_Toolkit.TF.TF_Functions.Compile import Compile
import numpy as np


STATE_INDICES = {} # This could be imported


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

        pose_theta = s[:, 0]
        pose_x = s[:, 1]
        pose_y = s[:, 2]

        speed = Q[:, 0]
        steering = Q[:, 1]

        for _ in tf.range(self.intermediate_steps):
            pose_theta = pose_theta + 0.5*(steering/self.intermediate_steps_float)
            pose_x = pose_x + self.t_step * speed * tf.math.cos(pose_theta)
            pose_y = pose_y + self.t_step * speed * tf.math.sin(pose_theta)

        s_next = tf.stack([pose_theta, pose_x, pose_y], axis=1)

        return s_next


class predictor_output_augmentation_tf:
    def __init__(self, net_info):
        self.net_output_indices = {key: value for value, key in enumerate(net_info.outputs)}
        indices_augmentation = []
        features_augmentation = []
        # if 'sin(x)' not in net_info.outputs:
        #     indices_augmentation.append(STATE_INDICES['sin(x)'])
        #     features_augmentation.append('sin(x)')
        #
        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

        if 'x' in net_info.outputs:
            self.index_x = tf.convert_to_tensor(self.net_output_indices['x'])

    def get_indices_augmentation(self):
        return self.indices_augmentation

    def get_features_augmentation(self):
        return self.features_augmentation

    @Compile
    def augment(self, net_output):

        output = net_output
        # if 'sin(x)' in self.features_augmentation:
        #     sin_x = tf.math.sin(net_output[..., self.index_x])[:, :, tf.newaxis]
        #     output = tf.concat([output, sin_x], axis=-1)

        return output
