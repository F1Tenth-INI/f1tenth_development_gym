
from tensorflow.keras.models import load_model
from joblib import load
import numpy as np
import os
import math
import tensorflow as tf
import yaml

from utilities.state_utilities import *

experiment_path =os.path.dirname(os.path.realpath(__file__))


# Load the model
model = load_model(experiment_path + '/models/my_model.h5')
input_scaler = load(experiment_path + '/models/input_scaler.joblib')
output_scaler = load(experiment_path + '/models/output_scaler.joblib')
with open(experiment_path + '/models/network.yaml', 'r') as file:
    network_yaml = yaml.safe_load(file)




def predict_next_state(s, u):
    
    # input_cols = ["angular_vel_z","linear_vel_x","pose_theta_sin","pose_theta_cos","steering_angle", 'slip_angle', 'angular_control_applied','translational_control_applied']
    # output_cols = ["d_angular_vel_z", "d_linear_vel_x", "d_pose_theta_sin" , "d_pose_theta_cos", 'd_steering_angle', 'd_slip_angle', 'd_pose_x', 'd_pose_y']

    input = [s[ANGULAR_VEL_Z_IDX], s[LINEAR_VEL_X_IDX],s[POSE_THETA_COS_IDX],s[POSE_THETA_SIN_IDX],s[STEERING_ANGLE_IDX],s[SLIP_ANGLE_IDX], u[ANGULAR_CONTROL_IDX], u[TRANSLATIONAL_CONTROL_IDX]]
    d_state = predict(input)
    
    d_state = np.insert(d_state, POSE_THETA_IDX, 0)
    
    print("s", s)
    print("d_state", 1/50 * d_state)
    
    next_state = s + 1/50 * d_state
    next_state[POSE_THETA_IDX] = np.arctan2(s[POSE_THETA_SIN_IDX], s[POSE_THETA_COS_IDX])    
    print("next_state", next_state)
       
    return next_state
    
    
def predict(input):
    
    example_input = [input]
    example_input = input_scaler.transform(example_input)
    example_input = tf.convert_to_tensor(example_input, dtype=tf.float32)
    prediction = _predict(example_input)
    prediction = prediction.numpy()
    prediction = output_scaler.inverse_transform(prediction)
    
    return prediction


# @tf.function
def _predict(input):

    # Make a prediction 
    # input = tf.expand_dims(input, axis=1)  # This will reshape your input from (1, 8) to (1, 1, 8)

    prediction = model(input)
    return prediction




if __name__ == "__main__":
    
    print("inputs" , network_yaml["input_cols"])
    print("outputs", network_yaml["output_cols"])
    
    
    example_s = np.array([0.0,0.0,1.319,0.249,0.969,-9.527,1.281,0.0,0.0]) 
    example_u = np.array([0.0,10.0])
    
    prediction = predict_next_state(example_s, example_u)
    # Should return 0.0,0.19,1.319,0.249,0.969,-9.527,1.282,0.0,0.0,

    
    # print("next_state: ", prediction)
