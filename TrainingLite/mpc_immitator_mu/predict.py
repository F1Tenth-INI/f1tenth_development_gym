
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from joblib import load
import numpy as np
import os
import math
import tensorflow as tf
import yaml

from utilities.state_utilities import *
from utilities.waypoint_utils import *

from tensorflow.keras.layers import LSTM, GRU, Dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

experiment_path =os.path.dirname(os.path.realpath(__file__))

# model_name = "GRU-Example2"
# model_name = "LSTM7_s1w0_large"
model_name = "tGRU_6_stateful4"


# Load the model

# When loading, provide the custom loss function in the `custom_objects` argument
# custom_objects = {'custom_mse_loss_with_washout': custom_mse_loss_with_washout}
custom_objects = {}

# Load the model and include the custom objects
model = load_model(experiment_path + '/models/'+model_name+'/my_model.keras', custom_objects=custom_objects)

input_scaler = load(experiment_path + '/models/'+model_name+'/input_scaler.joblib')
output_scaler = load(experiment_path + '/models/'+model_name+'/output_scaler.joblib')
with open(experiment_path + '/models/'+model_name+'/network.yaml', 'r') as file:
    network_yaml = yaml.safe_load(file)
    
    
sequence_length = network_yaml["sequence_length"]

mu_history = []
input_history = []


# seq_length = 1
sequence_length = 1
features = len(network_yaml["input_cols"])
outputs = len(network_yaml["output_cols"])
deploy_model = Sequential()
deploy_model.add(LSTM(64, batch_input_shape=(1, 1, features), return_sequences=True, stateful=False))
deploy_model.add(LSTM(64, return_sequences=False, stateful=False))
deploy_model.add(Dense(outputs))

deploy_model.set_weights(model.get_weights())
# deploy_model = model



def predict_next_control(s, waypoints_relative, waypoints):
    
    # state = [s[ANGULAR_VEL_Z_IDX], s[LINEAR_VEL_X_IDX],s[POSE_THETA_IDX],s[STEERING_ANGLE_IDX]]
    state = [s[ANGULAR_VEL_Z_IDX], s[LINEAR_VEL_X_IDX],s[POSE_THETA_IDX],s[STEERING_ANGLE_IDX], s[SLIP_ANGLE_IDX]]
    waypoints_x = waypoints_relative[:,0]
    waypoints_y = waypoints_relative[:,1]
    waypoints_vx = waypoints[:,WP_VX_IDX]
    
    input = np.concatenate((state, waypoints_x, waypoints_y, waypoints_vx))
    

    output = predict(input)
    
    control = [output[0]]
    # print("mu: ", output[0,2])
    # mu_history.append(output[0,2])
    
    window = 250
    # if len(mu_history) >= window:
        # last_avg = sum(mu_history[-window:]) / window
        # print(f"mu: {last_avg}")
    # else:
        # print("Not enough elements in mu_history to calculate the average of the last 50 values.")
    
    return control
    
    
def predict(input):
    
    
    
    example_input = [input]

    example_input = input_scaler.transform(example_input)
    # example_input = tf.convert_to_tensor(example_input, dtype=tf.float32)
    input_history.append(example_input) # shape (1, 65)
    if len(input_history) < sequence_length:
        print("Not enough elements in input_history to form a sequence of length 20.")
        return [[0,0]] 
    
    example_input = np.array(input_history[-sequence_length:]) # Add timestep dimension  (20, 1, 65)
  
    example_input_sequence = example_input.reshape(1, sequence_length, -1) # (1,20,65)


    prediction = _predict(example_input_sequence) # (1, 20, 3)
    prediction = prediction.numpy()
    prediction = prediction.reshape(1, -1)
    
    if len(prediction.shape) == 3:
        prediction = prediction.reshape(prediction.shape[0], -1) #(1, 60)
    
    prediction = output_scaler.inverse_transform(prediction) 
    
    return prediction


@tf.function
def _predict(input):

    # Make a prediction

    prediction = deploy_model([input])
    return prediction




if __name__ == "__main__":
    
    print("inputs" , network_yaml["input_cols"])
    print("outputs", network_yaml["output_cols"])
    
    example_input = np.array([0, 0., 0., 0.3232363, 0., 18.092684, 18.492521, 18.892359, 19.292198, 19.692036, 20.091873, 20.491713, 20.89155, 21.29139, 21.691227, 22.091066, 22.490904, 22.890741, 23.29058, 23.690418, 24.090258, 24.490095, 24.889935, 25.289772, 25.68961, -12.683541, -12.289727, -11.892686, -11.494469, -11.096025, -10.697667, -10.299485, -9.90103, -9.501698, -9.101922, -8.703446, -8.309871, -7.927363, -7.565548, -7.238788, -6.967873, -6.785771, -6.733249, -6.79366, -6.926713, 5.158623, 5.227309, 5.27428, 5.310179, 5.343515, 5.377885, 5.414228, 5.447311, 5.466889, 5.464843, 5.433206, 5.363721, 5.248224, 5.079041, 4.849722, 4.556875, 4.203009, 3.808486, 3.414118, 3.037381])    # should return -0.070	-0.013
    
    for i in range(21):
        prediction = predict(example_input)
    print(prediction)
