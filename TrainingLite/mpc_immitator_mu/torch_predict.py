import sys


import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn as nn
import torch.optim as optim
import numpy as np

from joblib import load

from utilities.state_utilities import *
from utilities.waypoint_utils import *


current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
from TrainingHelper import TrainingHelper
from TorchNetworks import LSTM as Network


model_name = "LSTM11_medium"

experiment_path = os.path.dirname(os.path.realpath(__file__))


training_helper = TrainingHelper(experiment_path, model_name)
network_yaml, input_scaler, output_scaler = training_helper.load_network_meta_data_and_scalers()

mu_history = []
input_history = []

def predict_next_control(s, waypoints_relative, waypoints ):
    
    # state = [s[ANGULAR_VEL_Z_IDX], s[LINEAR_VEL_X_IDX], s[STEERING_ANGLE_IDX], imu_data[0], imu_data[1], imu_data[2]]
    state = [s[ANGULAR_VEL_Z_IDX], s[LINEAR_VEL_X_IDX], s[STEERING_ANGLE_IDX]]
    print(state)
    waypoints_x = waypoints_relative[:25, 0]
    waypoints_y = waypoints_relative[:25, 1]
    waypoints_vx = waypoints[:25, WP_VX_IDX]
    
    # print('state_here', state)
    # print('waypoints_x', waypoints_x)
    # print('waypoints_y', waypoints_y)
    # print('waypoints_vx', waypoints_vx)
    
    input = np.concatenate((state, waypoints_x, waypoints_y, waypoints_vx))
    output = predict(input)
    control = [output]

    return control

# Recreate the model class with LSTM


# Specify the model parameters (same as during training)
input_size = len(network_yaml["input_cols"])  # Number of input features
output_size = len(network_yaml["output_cols"])  # Number of output features
hidden_size = network_yaml["hidden_size"]
num_layers = network_yaml["num_layers"]

# Create an instance of the model
model = Network(input_size, hidden_size, output_size, num_layers)

# Load the saved model state dictionary
model.load_state_dict(torch.load(os.path.join(training_helper.model_dir, "model.pth")))

# Transfer the model to the appropriate device (GPU or CPU)
device = 'cpu'
model = model.to(device)

example_input = torch.rand(1, 1, input_size)
traced_model = torch.jit.trace(model, example_input)

# Save the TorchScript model
script_model_path = os.path.join(training_helper.model_dir, "model_traced.pth")
traced_model.save(script_model_path)

print(f"TorchScript model saved to {script_model_path}")
exit()
# Set the model to evaluation mode
model.eval()

def predict(input):



    X = [input]
    X_scaled = input_scaler.transform(X)

    # Convert the preprocessed data into a PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension

    # Move the tensor to the same device as the model
    X_tensor = X_tensor.to(next(model.parameters()).device)

    # Perform inference
    
    with torch.no_grad():
        output, model.hidden = model(X_tensor, model.hidden)
    return [[0,0]]

    # Inverse transform the output (if scaling was used)
    predictions = output_scaler.inverse_transform(output.cpu().numpy())
    return predictions[0]

if __name__ == "__main__":
    print("inputs", network_yaml["input_cols"])
    print("outputs", network_yaml["output_cols"])

    example_input = np.array([0, 0., 0., 0.3232363, 0., 18.092684, 18.492521, 18.892359, 19.292198, 19.692036, 20.091873, 20.491713, 20.89155, 21.29139, 21.691227, 22.091066, 22.490904, 22.890741, 23.29058, 23.690418, 24.090258, 24.490095, 24.889935, 25.289772, 25.68961, -12.683541, -12.289727, -11.892686, -11.494469, -11.096025, -10.697667, -10.299485, -9.90103, -9.501698, -9.101922, -8.703446, -8.309871, -7.927363, -7.565548, -7.238788, -6.967873, -6.785771, -6.733249, -6.79366, -6.926713, 5.158623, 5.227309, 5.27428, 5.310179, 5.343515, 5.377885, 5.414228, 5.447311, 5.466889, 5.464843, 5.433206, 5.363721, 5.248224, 5.079041, 4.849722, 4.556875, 4.203009, 3.808486, 3.414118, 3.037381])  # should return -0.070 -0.013

    # Initialize hidden state for prediction
    model.hidden = model.reset_hidden_state(1)

    for i in range(21):
        prediction = predict(example_input)
    print(prediction)