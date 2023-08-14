import tensorflow as tf

from SI_Toolkit_ASF.car_model import car_model


class STModel(tf.keras.Model):
    def __init__(self, horizon, batch_size, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dt = 0.04

        self.car_model = car_model(
            model_of_car_dynamics='ODE:st',
            with_pid=True,
            batch_size=batch_size,
            car_parameter_file='utilities/car_files/custom_car_parameters.yml',
            dt=self.dt,
            intermediate_steps=1
        )

        self.car_parameters_tf = {}
        for name, param in self.car_model.car_parameters.items():
            self.car_parameters_tf[name] = tf.Variable(param, name=name, trainable=name in ['mu'], dtype=tf.float32)
        self.car_model.car_parameters = self.car_parameters_tf

    def calculate_derivative(self, x_old, x_new, dt):
        # It needs to be s - s_previous!
        return (x_new - x_old) / dt

    def call(self, x, training=None, mask=None):
        # print(f'x: {x}')
        Q = x[:, 0, 0:2]
        s = x[:, 0, 2:]
        # print(f's: {s}')
        # print(f'Q: {Q}')

        # output = self.predictor.predict_tf(s, Q)
        # output = self.next_step_predictor.step(s, Q)
        output = self.car_model.step_dynamics(s, Q, None)  # Third argument params not implemented

        # output = self.calculate_derivative(s, output, self.dt)
        output = tf.expand_dims(output, 1)

        return output


if __name__ == '__main__':
    model = STModel(1, 1)

    test_input = tf.constant([[[-0.595, 4.752, -1.17, 5.36, 2.241, -0.621, 0.784, -11.343, -75.338, 0.121, -0.192]]])
    test_output = tf.constant([[-3.884, 5.039, 2.15, -0.547, 0.837, -11.487, -75.187, 0.127, -0.32]])
    print(test_input.numpy().round(2))
    output = model(test_input)
    print(output.numpy().round(2))
    print(test_output.numpy().round(2))
    print()
