
from types import SimpleNamespace
from utilities.state_utilities import *
from utilities.Settings import Settings
import os
from SI_Toolkit.Predictors.neural_network_evaluator import neural_network_evaluator
from SI_Toolkit.Functions.General.Initialization import get_net, get_norm_info_for_net
import tensorflow as tf

from SI_Toolkit.computation_library import NumpyLibrary

environment_name = Settings.ENVIRONMENT_NAME
model_of_car_dynamics = Settings.ODE_MODEL_OF_CAR_DYNAMICS
car_parameter_file = Settings.CONTROLLER_CAR_PARAMETER_FILE

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
        self.s = self.lib.to_tensor(create_car_state(), dtype=self.lib.float32)

        self.params = None

        self.intermediate_steps = int(intermediate_steps)
        self.t_step = float(dt / float(self.intermediate_steps))

        #TODO: Probably not best to hard code these things
        # self.nn_path = "SI_Toolkit_ASF/Experiments/DNN_history/Models/Dense-72IN-32H1-64H2-128H3-9OUT-0"
        self.nn_path = "SI_Toolkit_ASF/Experiments/DNN_history/Models/Dense-72IN-32H1-64H2-128H3-256H4-64H5-9OUT-0" # Doing the best
        # self.nn_path = "SI_Toolkit_ASF/Experiments/DNN_history/Models/Dense-72IN-32H1-64H2-128H3-256H4-128H5-9OUT-0" #Doing the worst 
        # self.nn_path= "SI_Toolkit_ASF/Experiments/DNN_nohist/Models/Dense-9IN-32H1-64H2-128H3-256H4-128H5-9OUT-0" #No history 
        # self.nn_path = "SI_Toolkit_ASF/Experiments/DNN_history/Models/Dense-72IN-32H1-64H2-128H3-256H4-64H5-9OUT-2" #No Normalization
       
        self.history=7
        
        self.state_residual_NN = neural_network_evaluator(
            net_name=os.path.basename(self.nn_path),
            path_to_models=os.path.dirname(self.nn_path),
            batch_size=batch_size
            )

        # a = SimpleNamespace()
        # a.path_to_models = os.path.dirname(self.nn_path)
        # a.net_name = os.path.basename(self.nn_path)
        # self.net, self.net_info = \
        #     get_net(a, time_series_length=1,
        #             batch_size=batch_size, stateful=True)
        print("Yay, loaded the pretrained NN!")
        self.past_states=[]

        self.hist_count=0
        for i, w in enumerate(self.state_residual_NN.net.weights):
            if tf.reduce_any(tf.math.is_nan(w)).numpy():
                print(f"NaN found in weight {i}: {w.name}")
            else:
                print(f"Weight {i} is fine: {w.name}")



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
                variable_parameters=variable_parameters,
                                 )  # Environment model, keeping car ODEs
        else:
            raise NotImplementedError('{} not yet implemented in next_state_predictor_ODE_tf'.format(Settings.ENVIRONMENT_NAME))

        self.variable_parameters = variable_parameters

        if disable_individual_compilation:
            self.step = self._step
        else:
            from SI_Toolkit.Compile import CompileTF # Lazy import
            self.step = CompileTF(self._step)

    def _step(self, s, Q):

        if self.core_dynamics_only:
            s_next = self.env.step_dynamics_core(s, Q)
            print("IN CORE DYNAMICS ONLY")
            return s_next
        else:
            s_next = self.env.step_dynamics(s, Q)

            #alter the output to remove the pos_x,pos_y (index 7,8)
            if self.hist_count < self.history:
                print("There is not enough History, appending!")

                s_del = tf.concat([s[:, :6], s[:, 9:]], axis=1)
          
                sq=tf.concat([s_del,Q], axis=1)


                self.past_states.append(sq)
                self.hist_count+=1
                # print("current history count:", self.hist_count)
                return s_next

            else:
                # print("Have full history, trying to calculate")
                return s_next
                s_del = tf.concat([s[:, :6], s[:, 9:]], axis=1)
                sq=tf.concat([s_del,Q], axis=1)
                #sq now looks like this (batch, 9 states):
                #[angular_vel_z, linear_vel_x, linear_vel_y, pose_theta, pose_theta_cos, pose_theta_sin, steering_angle, angular control, translational_control
                self.past_states.append(sq)

                #Now re-order the states into alphabetical order:
                #Angular control, angular_vel_z, linear_vel_x, linear_vel_y, pose_theta, pose_theta_cos, pose_theta_sin, steering_angle,translational_control
                nn_index_map  = [7,0,1,2,3,4,5,6,8]
                NN_input = tf.concat(self.past_states, axis=1)
                NN_input_reorder = []

     
                #add the controls first
                NN_input_reorder.append(NN_input[:, (self.history +1) * 9 -2 : (self.history+1) * 9]) #angular control, translational control

                for i in nn_index_map:
                    for j in range(self.history+1):
                        if i == 7 or i == 8: #Search for controls
                            if j == self.history: #if it is the last one, we don't want to add it again
                                continue
                            else:
                                NN_input_reorder.append(NN_input[:, i + 9 * j : i + 9 * j + 1])
                        else:
                            NN_input_reorder.append(NN_input[:, i + 9 * j : i + 9 * j + 1])
    

                #Now remove the duplicate control at the end
                NN_input_final = tf.concat(NN_input_reorder, axis=1)

                #Output of NN is in the following order:
                #ang_vel_z, lin_vel_x, lin_vel_y, pose_theta, pose_theta_cos, pose_theta_sin, steering angle, err_x, err_y
                NN_residual = self.state_residual_NN.step(NN_input_final)

                self.past_states.pop(0)


                #Mapping the output of NN to correct index of state
                nn_index_map  = [0,1,2,3,4,5,7,8,6]
                NN_residual = tf.squeeze(NN_residual, axis=1) #removes unneccesary dim 1
                NN_residual_reordered = tf.gather(NN_residual, nn_index_map, axis=1)  # shape: (batch_size, 9)

                residual_padded = tf.zeros_like(s_next)  # shape (batch_size, 10)
                zero_col = tf.zeros((tf.shape(NN_residual_reordered)[0], 1))  # shape (batch_size, 1)
                
                #ang_vel_z, lin_vel_x, lin_vel_y, pose_theta, pose_theta_cos, pose_theta_sin,  err_x, err_y, steering angle
                residual_weights = tf.constant([1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1], dtype=tf.float32)
                weighted_residual = NN_residual_reordered * residual_weights  # shape: (batch_size, 9)

                # Split and insert at index 9
                residual_padded = tf.concat([
                    # NN_residual_reordered[:, :8],  # ang_vel_z, lin_vel_x, lin_vel_y, pose_theta, pose_theta_cos, pose_theta_sin, err_x, err_y
                    weighted_residual[:, :8],  # NN_residual_reordered Or change to weighted_residual[:, :8]
                    zero_col,            # Insert zero column at index 9
                    weighted_residual[:, 8:]  
                    # NN_residual_reordered[:, 8:]  # steering angle
                ], axis=1)
                
                s_next_updated = s_next + residual_padded
                

                return s_next_updated



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
            self.index_pose_theta = self.lib.to_tensor(self.outputs_after_integration_indices['pose_theta'])
        if 'pose_theta_sin' in outputs_after_integration:
            self.index_pose_theta_sin = self.lib.to_tensor(self.outputs_after_integration_indices['pose_theta_sin'])
        if 'pose_theta_cos' in outputs_after_integration:
            self.index_pose_theta_cos = self.lib.to_tensor(self.outputs_after_integration_indices['pose_theta_cos'])

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
