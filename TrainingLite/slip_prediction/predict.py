
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import os
import math
import tensorflow as tf
import yaml

from utilities.state_utilities import *
from utilities.waypoint_utils import *
from utilities.imu_simulator import *

experiment_path =os.path.dirname(os.path.realpath(__file__))

model_name = "GRU_Example"

# Load the model
model = load_model(experiment_path + '/models/'+model_name+'/my_model.keras')
input_scaler = load(experiment_path + '/models/'+model_name+'/input_scaler.joblib')
output_scaler = load(experiment_path + '/models/'+model_name+'/output_scaler.joblib')
with open(experiment_path + '/models/'+model_name+'/network.yaml', 'r') as file:
    network_yaml = yaml.safe_load(file)

mu_history = []


def predict_slip_angle(state: np.ndarray, imu_data: np.ndarray) -> float:
    
    state_dict = StateUtilities.state_to_dict(state)
    imu_dict = IMUSimulator.array_to_dict(imu_data)
    
    input_dict = {**state_dict, **imu_dict}
    
    input_cols = network_yaml['input_cols']
    
    input = [input_dict[col] for col in input_cols]


    output = predict(input)
    
    slip_angle = output[0]
    
    return slip_angle
    
    
def predict(input):
    
    
    example_input = [input]

    example_input = input_scaler.transform(example_input)
    example_input = tf.convert_to_tensor(example_input, dtype=tf.float32)
    
    example_input = np.expand_dims(example_input, axis=1)  # Add timestep dimension

    prediction = _predict(example_input)
    prediction = prediction.numpy()
    prediction = output_scaler.inverse_transform(prediction)
    
    return prediction


@tf.function
def _predict(input):

    # Make a prediction

    prediction = model([input])
    return prediction




if __name__ == "__main__":
    
    print("inputs" , network_yaml["input_cols"])
    print("outputs", network_yaml["output_cols"])
    
    s = np.array([0.0,0.006,-0.355,0.938,-0.347,-7.891,5.376,0.0, 0])
    imu = np.array([0.0,0.0,0.0])
    slip_angle = predict_slip_angle(s, imu)
    
    print(slip_angle)
