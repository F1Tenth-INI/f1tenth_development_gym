import tensorflow as tf
import numpy as np

from SI_Toolkit_ASF.car_model import car_model
from utilities.Settings import Settings
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function
from SI_Toolkit.Functions.General.Initialization import get_norm_info_for_net
from SI_Toolkit.computation_library import TensorFlowLibrary

# This is a custom keras model for the ODE model of the car dynamics
# It is not a real network, but it can calculate the ODE with neuw paramteres without recompiling the TensorFlow Graph


class ODEModel(tf.keras.Model):
    
    # 1st important function for a "fake network" keras model
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dt = Settings.TIMESTEP_PLANNER
        self.lib = TensorFlowLibrary()
        self.intermediate_steps = 4
        self.batch_size = batch_size
        self.car_parameter_file = 'utilities/car_files/custom_car_parameters.yml'
        self.setup_car_model(self.car_parameter_file, Settings.ODE_MODEL_OF_CAR_DYNAMICS)

        self.normalization_info = get_norm_info_for_net(net_info, copy_files=False)
        self.denormalize_inputs = get_denormalization_function(self.normalization_info, net_info.inputs, self.lib)
        self.normalize_outputs = get_normalization_function(self.normalization_info, net_info.outputs, self.lib)

    def setup_car_model(self, car_parameter_file, model_of_car_dynamics, trainable_params='all'):
        self.car_model = car_model(
            model_of_car_dynamics=model_of_car_dynamics,
            batch_size=self.batch_size,
            car_parameter_file=car_parameter_file,
            dt=self.dt,
            intermediate_steps=self.intermediate_steps,
            computation_lib=self.lib
        )

        self.car_parameters_tf = {}
        for name, param in self.car_model.car_parameters.items():
            if trainable_params == 'all':
                trainable = True
            else:
                trainable = name in trainable_params
            self.car_parameters_tf[name] = tf.Variable(param, name=name, trainable=trainable, dtype=tf.float32)
        self.car_model.car_parameters = self.car_parameters_tf

    def calculate_derivative(self, x_old, x_new, dt):
        return (x_new - x_old) / dt


    # 2nt important function for a "fake network" keras model
    # Will be called in every training/predition step
    def call(self, x, training=None, mask=None):
        x = self.denormalize_inputs(x)
        Q = x[:, 0, 0:2]
        s = x[:, 0, 2:]

        output = self.car_model.step_dynamics(s, Q, None)  # Third argument params not implemented
        # output = self.calculate_derivative(s, output, self.dt)
        output = tf.expand_dims(output, 1)
        output = self.normalize_outputs(output)
        return output


class STModel(ODEModel):
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, net_info, name=None, **kwargs)
        self.setup_car_model(self.car_parameter_file, 'ODE:st', trainable_params=['mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'I'])


class STModelMu(ODEModel):
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, net_info, name=None, **kwargs)
        self.setup_car_model(self.car_parameter_file, 'ODE:st', trainable_params=['mu'])


class STModelLowMu(ODEModel):
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, net_info, name=None, **kwargs)
        self.setup_car_model('utilities/car_files/custom_car_parameters_lower_mu.yml', 'ODE:st', trainable_params=['mu'])


class KSModel(ODEModel):
    def __init__(self, horizon, batch_size, net_info, name=None, **kwargs):
        super().__init__(horizon, batch_size, net_info, name=None, **kwargs)
        self.setup_car_model(self.car_parameter_file, 'ODE:st', trainable_params=['mu', 'C_Sf', 'C_Sr', 'lf', 'lr', 'h', 'I'])