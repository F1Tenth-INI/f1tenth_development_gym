import numpy as np

from utilities.Settings import Settings

from SI_Toolkit.computation_library import NumpyLibrary

environment_name = Settings.ENVIRONMENT_NAME
model_of_car_dynamics = Settings.ODE_MODEL_OF_CAR_DYNAMICS
car_parameter_file = Settings.CONTROLLER_CAR_PARAMETER_FILE

STATE_VARIABLES_FOR_PREDICTOR = np.sort([
    'angular_vel_z',  # x5: yaw rate
    'linear_vel_x',  # x3: velocity in x direction
    'linear_vel_y',  # x6: velocity in y direction
    'pose_theta',  # x4: yaw angle
    'pose_theta_cos',
    'pose_theta_sin',
    'pose_x',  # x0: x position in global coordinates
    'pose_y',  # x1: y position in global coordinates
    'slip_angle',  # [DEPRECATED] x6: slip angle at vehicle center
    'steering_angle'  # x2: steering angle of front wheels
])

STATE_INDICES = {x: np.where(STATE_VARIABLES_FOR_PREDICTOR == x)[0][0] for x in STATE_VARIABLES_FOR_PREDICTOR}
STATE_VARIABLES = STATE_VARIABLES_FOR_PREDICTOR

CONTROL_INPUTS_FOR_PREDICTOR = np.sort([
    'angular_control_calculated_1',
    'translational_control_calculated_1',
    # 'mu'  # Include it for brunton plot
])
CONTROL_INPUTS = CONTROL_INPUTS_FOR_PREDICTOR
CONTROL_INPUTS_LEN = len(CONTROL_INPUTS_FOR_PREDICTOR)

class next_state_predictor_ODE():

    def __init__(self,
                 dt,
                 intermediate_steps,
                 lib,
                 batch_size=1,
                 variable_parameters=None,
                 disable_individual_compilation=False,
                 **kwargs,
                 ):
        self.lib = lib
        self.intermediate_steps = int(intermediate_steps)
        self.t_step = float(dt / float(self.intermediate_steps))
        self.variable_parameters = variable_parameters

        if "core_dynamics_only" in kwargs and kwargs["core_dynamics_only"] is True:
            self.core_dynamics_only = True
        else:
            self.core_dynamics_only = False

        if environment_name == 'Car':
            from SI_Toolkit_ASF.car_model import car_model
            self.env = car_model(
                model_of_car_dynamics=model_of_car_dynamics,
                batch_size=batch_size,
                car_parameter_file=car_parameter_file,
                dt=dt,
                computation_lib=lib,
                intermediate_steps=intermediate_steps,
                                 )  # Environment model, keeping car ODEs
        else:
            raise NotImplementedError('{} not yet implemented in next_state_predictor_ODE_tf'.format(Settings.ENVIRONMENT_NAME))

        self.params = self.env.car_parameters

        if disable_individual_compilation:
            self.step = self._step
        else:
            from SI_Toolkit.Compile import CompileAdaptive  # Lazy import
            self.step = CompileAdaptive(self._step)

    def _step(self, s, Q):

        if self.variable_parameters is not None and hasattr(self.variable_parameters, 'mu'):
            # mu = self.lib.to_numpy(self.variable_parameters.mu, self.lib.float32)
            self.params.mu = self.variable_parameters.mu

        if self.core_dynamics_only:
            s_next = self.env.step_dynamics_core(s, Q)
        else:
            s_next = self.env.step_dynamics(s, Q)

        return s_next



class predictor_output_augmentation:
    def __init__(self, net_info, lib=NumpyLibrary(), disable_individual_compilation=False, differential_network=False):

        self.lib = lib

        outputs_after_integration = [(x[2:] if x[:2] == 'D_' else x) for x in net_info.outputs]
        self.outputs_after_integration_indices = {key: value for value, key in enumerate(outputs_after_integration)}
        indices_augmentation = []
        features_augmentation = []
        if 'angular_vel_z' not in outputs_after_integration:
            indices_augmentation.append(STATE_INDICES['angular_vel_z'])
            features_augmentation.append('angular_vel_z')
        if 'linear_vel_x' not in outputs_after_integration:
            indices_augmentation.append(STATE_INDICES['linear_vel_x'])
            features_augmentation.append('linear_vel_x')
        if 'linear_vel_y' not in outputs_after_integration and 'linear_vel_y' in STATE_INDICES.keys():  # Quadruped only
            indices_augmentation.append(STATE_INDICES['linear_vel_y'])
            features_augmentation.append('linear_vel_y')

        if 'pose_theta' not in outputs_after_integration and 'pose_theta_sin' in outputs_after_integration and 'pose_theta_cos' in outputs_after_integration:
            indices_augmentation.append(STATE_INDICES['pose_theta'])
            features_augmentation.append('pose_theta')
        if 'pose_theta_sin' not in outputs_after_integration and 'pose_theta' in outputs_after_integration:
            indices_augmentation.append(STATE_INDICES['pose_theta_sin'])
            features_augmentation.append('pose_theta_sin')
        if 'pose_theta_cos' not in outputs_after_integration and 'pose_theta' in outputs_after_integration:
            indices_augmentation.append(STATE_INDICES['pose_theta_cos'])
            features_augmentation.append('pose_theta_cos')

        if 'slip_angle' not in outputs_after_integration:
            indices_augmentation.append(STATE_INDICES['slip_angle'])
            features_augmentation.append('slip_angle')
        if 'steering_angle' not in outputs_after_integration:
            indices_augmentation.append(STATE_INDICES['steering_angle'])
            features_augmentation.append('steering_angle')

        self.indices_augmentation = indices_augmentation
        self.features_augmentation = features_augmentation
        self.augmentation_len = len(self.indices_augmentation)

        if 'pose_theta' in outputs_after_integration:
            self.index_pose_theta = self.lib.to_tensor(self.outputs_after_integration_indices['pose_theta'], self.lib.int32)
        if 'pose_theta_sin' in outputs_after_integration:
            self.index_pose_theta_sin = self.lib.to_tensor(self.outputs_after_integration_indices['pose_theta_sin'], self.lib.int32)
        if 'pose_theta_cos' in outputs_after_integration:
            self.index_pose_theta_cos = self.lib.to_tensor(self.outputs_after_integration_indices['pose_theta_cos'], self.lib.int32)

        if disable_individual_compilation:
            self.augment = self._augment
        else:
            from SI_Toolkit.Compile import CompileTF # Lazy import
            self.augment = CompileTF(self._augment)

    def get_indices_augmentation(self):
        return self.indices_augmentation

    def get_features_augmentation(self):
        return self.features_augmentation

    def _augment(self, net_output):

        output = net_output
        if 'angular_vel_z' in self.features_augmentation:
            angular_vel_z = self.lib.zeros_like(net_output[:, :, -1:])
            output = self.lib.concat([output, angular_vel_z], axis=-1)
        if 'linear_vel_x' in self.features_augmentation:
            linear_vel_x = self.lib.zeros_like(net_output[:, :, -1:])
            output = self.lib.concat([output, linear_vel_x], axis=-1)
        if 'linear_vel_y' in self.features_augmentation:
            linear_vel_y = self.lib.zeros_like(net_output[:, :, -1:])
            output = self.lib.concat([output, linear_vel_y], axis=-1)

        if 'pose_theta' in self.features_augmentation:
            pose_theta = self.lib.atan2(
                net_output[..., self.index_pose_theta_sin],
                net_output[..., self.index_pose_theta_cos])[:, :,
                         self.lib.newaxis]  # tf.math.atan2 removes the features (last) dimension, so it is added back with [:, :, tf.newaxis]
            output = self.lib.concat([output, pose_theta], axis=-1)

        if 'pose_theta_sin' in self.features_augmentation:
            pose_theta_sin = \
                self.lib.sin(net_output[..., self.index_pose_theta])[:, :, self.lib.newaxis]
            output = self.lib.concat([output, pose_theta_sin], axis=-1)

        if 'pose_theta_cos' in self.features_augmentation:
            pose_theta_cos = \
                self.lib.cos(net_output[..., self.index_pose_theta])[:, :, self.lib.newaxis]
            output = self.lib.concat([output, pose_theta_cos], axis=-1)

        if 'slip_angle' in self.features_augmentation:
            slip_angle = self.lib.zeros_like(net_output[:, :, -1:])
            output = self.lib.concat([output, slip_angle], axis=-1)

        if 'steering_angle' in self.features_augmentation:
            steering_angle = self.lib.zeros_like(net_output[:, :, -1:])
            output = self.lib.concat([output, steering_angle], axis=-1)


        return output

