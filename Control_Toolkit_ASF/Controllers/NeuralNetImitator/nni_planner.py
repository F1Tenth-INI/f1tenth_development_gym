import tensorflow as tf

from types import SimpleNamespace

from SI_Toolkit.TF.TF_Functions.Normalising import get_normalization_function_tf, get_denormalization_function_tf

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_VARIABLES, STATE_INDICES, \
        CONTROL_INPUTS, augment_predictor_output
except ModuleNotFoundError:
    print('SI_Toolkit_ASF not yet created')

from SI_Toolkit.TF.TF_Functions.Initialization import get_net, get_norm_info_for_net
from SI_Toolkit.TF.TF_Functions.Compile import Compile

NET_NAME = 'Dense-68IN-128H1-128H2-2OUT-0'
PATH_TO_MODELS = 'SI_Toolkit_ASF/Experiments/Experiment-MPPI-Imitator/Models/'

class NeuralNetImitatorPlanner:

    def __init__(self, speed_fraction=1, batch_size=1):

        print('Loading NeuralNetImitatorPlanner')

        self.translational_control = None
        self.angular_control = None

        self.simulation_index = 0

        self.car_state = None

        a = SimpleNamespace()
        self.batch_size = batch_size  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented

        a.path_to_models = PATH_TO_MODELS

        a.net_name = NET_NAME

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.normalization_info = get_norm_info_for_net(self.net_info)

        self.normalize_inputs = get_normalization_function_tf(self.normalization_info, self.net_info.inputs)
        self.denormalize_outputs = get_denormalization_function_tf(self.normalization_info, self.net_info.outputs)

        self.net_input = None
        self.net_input_normed = tf.Variable(
            tf.zeros([len(self.net_info.inputs),], dtype=tf.float32))

    def render(self, e):
        return



    def process_observation(self, ranges=None, ego_odom=None):

        # Accelerate at the beginning (St model expoldes for small velocity)
        if self.simulation_index < 20:
            self.simulation_index += 1
            self.translational_control = 10
            self.angular_control = 0
            return self.translational_control, self.angular_control

        ranges = ranges[200:880]
        ranges = ranges[::10]

        net_input = tf.convert_to_tensor(ranges, tf.float32)

        net_output = self.process_tf(net_input)

        net_output = net_output.numpy()

        speed = float(net_output[0])
        steering_angle = float(net_output[1])

        self.translational_control = speed
        self.angular_control = steering_angle


        return speed, steering_angle

    @Compile
    def process_tf(self, net_input):

        self.net_input_normed.assign(self.normalize_inputs(
            net_input
        ))

        net_input = (tf.reshape(net_input, [-1, 1, len(self.net_info.inputs)]))

        net_output = self.net(net_input)

        net_output = self.denormalize_outputs(net_output)

        return tf.squeeze(net_output)
        

