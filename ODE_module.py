import tensorflow as tf
import numpy as np

from SI_Toolkit_ASF.car_model import car_model
from utilities import Settings


class STModel(tf.keras.Model):
    def __init__(self, horizon, batch_size, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dt = Settings.TIMESTEP_PLANNER
        self.intermediate_steps = 1
        self.batch_size = batch_size

    def setup_car_model(self, car_parameter_file):
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            with_pid=True,
            batch_size=self.batch_size,
            car_parameter_file=car_parameter_file,
            dt=self.dt,
            intermediate_steps=self.intermediate_steps
        )

        self.car_parameters_tf = {}
        for name, param in self.car_model.car_parameters.items():
            self.car_parameters_tf[name] = tf.Variable(param, name=name, trainable=name in ['mu'], dtype=tf.float32)
        self.car_model.car_parameters = self.car_parameters_tf

    def calculate_derivative(self, x_old, x_new, dt):
        # It needs to be s - s_previous!
        return (x_new - x_old) / dt

    def call(self, x, training=None, mask=None):
        Q = x[:, 0, 0:2]
        s = x[:, 0, 2:]

        output = self.car_model.step_dynamics(s, Q, None)  # Third argument params not implemented
        # output = self.calculate_derivative(s, output, self.dt)
        output = tf.expand_dims(output, 1)
        return output


class STModel_high_mu(STModel):
    def __init__(self, horizon, batch_size, name=None, **kwargs):
        super().__init__(horizon, batch_size, name, **kwargs)
        self.setup_car_model('utilities/car_files/custom_car_parameters.yml')


class STModel_low_mu(STModel):
    def __init__(self, horizon, batch_size, name=None, **kwargs):
        super().__init__(horizon, batch_size, name, **kwargs)
        self.setup_car_model('utilities/car_files/custom_car_parameters_lower_mu.yml')


if __name__ == '__main__':
    model = STModel(1, 1)

    # Random input
    # test_input = tf.constant([[[-0.595, 4.752, -1.17, 5.36, 2.241, -0.621, 0.784, -11.343, -75.338, 0.121, -0.192]]])
    # test_output = tf.constant([[-3.884, 5.039, 2.15, -0.547, 0.837, -11.487, -75.187, 0.127, -0.32]])

    # From training file
    # test_input = tf.constant([[[-0.174, 8.964, -0.005, 0.167, -0.0, 1.0, -0.0, 500.001, 500.0, 0.0, -0.02]]])
    # test_output = tf.constant([[-0.019, 0.257, -0.0, 1.0, -0.0, 500.002, 500.0, 0.0, -0.035]])

    # From training file with higher precision
    # test_input = tf.constant([[[0.734321515676981, 10.523349338948194, 0.111248672718177, 8.400835812164585, 0.894186934039111, 0.894186934039111, 0.779700235280048, 512.725097207171, 506.07671141243696, -0.070781659888563, -0.107033997920816]]])
    # test_output = tf.constant([[-0.275381585802276, 8.479332989676983, 0.895299420766293, 0.625285177553879, 0.780396339516924, 512.7821999988048, 506.13832858067695, -0.069763924785444, -0.075033997920816]])

    # Test out delta
    test_input = tf.constant([[[ -0.55795187,  14.047229  ,   0.5890474 ,   1.0044736 ,
          0.0147171 ,   0.9998917 ,   0.01471657, 500.05103   ,
        500.00018   ,   0.        ,   0.20852426]]])
    test_output = tf.constant([[-26.41,   9.01,   0.59,  -0.01,   0.59,   1.,     0.01,   4.25,  -3.2 ]])

    np.set_printoptions(suppress=True, precision=5, linewidth=100)
    print(test_input.numpy())
    output = model(test_input)
    print(output.numpy())
    print(test_output.numpy())
    print()
