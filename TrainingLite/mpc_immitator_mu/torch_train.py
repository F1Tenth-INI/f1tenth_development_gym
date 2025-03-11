import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from tqdm import trange

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
from TrainingHelper import TrainingHelper
from TorchNetworks import LSTM as Network

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from utilities.lidar_utils import *
from utilities.state_utilities import *


# Setup experiment paths and parameters
experiment_path = os.path.dirname(os.path.realpath(__file__))
model_name = "2025-03-10_mu075-2"
dataset_name = "2025-03-10-delay"
number_of_epochs = 30
batch_size = 16
save_historgrams = False


shuffle_data = True
train_on_output_sequences = False

model_dir = os.path.join(experiment_path, 'models', model_name)
dataset_dir = os.path.join(experiment_path, '..', 'Datasets', dataset_name)


training_helper = TrainingHelper(experiment_path, model_name, dataset_name)
training_helper.create_and_clear_model_folder(model_dir)
training_helper.save_training_scripts(os.path.realpath(__file__))

lidar_utils = LidarHelper()

df, file_change_indices = training_helper.load_dataset(reduce_size_by=5)

batches = []

df = df.dropna()

# Define input and output columns
# state_cols = ["angular_vel_z", "linear_vel_x", "steering_angle", "imu_dd_x", "imu_dd_y", "imu_dd_yaw"]
# state_cols = ["angular_vel_z", "linear_vel_x", "prev_angular_control_calculated", "prev_translational_control_calculated"]
# state_cols = [ "linear_vel_x"]
state_cols = ["angular_vel_z", "linear_vel_x", "linear_vel_y", "steering_angle"]
lidar_cols = lidar_utils.get_processed_ranges_names()


wypt_x_cols = ["WYPT_REL_X_{:02d}".format(i) for i in range(0, 30)]
wypt_y_cols = ["WYPT_REL_Y_{:02d}".format(i) for i in range(0, 30)]
wypt_vx_cols = ["WYPT_VX_{:02d}".format(i) for i in range(0, 30)]
input_cols = state_cols + wypt_x_cols + wypt_y_cols + wypt_vx_cols + lidar_cols
output_cols = ["angular_control_calculated", "translational_control_calculated"]


# Shift output to counteract delay
# for col in ["angular_control_calculated", "translational_control_calculated"]:
#     df[col] = df[col].shift(-4)

df = df.dropna()

if save_historgrams:
    training_helper.create_histograms(df, input_cols, output_cols)


# Scaling input and output data
X = df[input_cols].to_numpy()
y = df[output_cols].to_numpy()


X, y = training_helper.fit_trainsform_save_scalers(X, y)



losses = []

input_size = len(input_cols)
output_size = len(output_cols)
hidden_size = 128
num_layers = 3

batch_size = 64
window_size = 100
step_size = 10
washout_period = 20  # First 20 timesteps ignored for loss

# Initialize the model
model = Network(input_size, hidden_size, output_size, num_layers)
training_helper.save_network_metadata(input_cols, output_cols, model)

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Cut training data such that it is divisible by batch size (all batches have the same size)
# num_batches = len(X_train) // batch_size
# X_train = X_train[:num_batches * batch_size]
# y_train = y_train[:num_batches * batch_size]



X, y, new_file_change_indices = training_helper.shuffle_dataset_by_chunks(X, y, batch_size, window_size, step_size)


# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Dataset and DataLoader
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

dataset = TensorDataset(X_train, y_train)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Training loop
train_losses = []
val_losses = []

print(f"Training {model_name} started.")





# Training loop
model.train()
for epoch in range(number_of_epochs):
    epoch_loss = 0.0
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(data_loader)  # initialize once per epoch
    for batch_idx in trange(len(data_loader), desc=f'Epoch {epoch+1}/{number_of_epochs}'):
        batch_x, batch_y = next(data_iter)
        hidden = model.reset_hidden_state(batch_x.size(0))
        optimizer.zero_grad()
        output, hidden = model(batch_x, hidden)

        # Apply washout: ignore initial timesteps
        output_washed = output[:, washout_period:, :]
        target_washed = batch_y[:, washout_period:, :]

        loss = criterion(output_washed, target_washed)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_loader)
    print(f'Epoch [{epoch + 1}/{number_of_epochs}], Loss: {avg_loss:.6f}')


    # Save the model after every epoch
    training_helper.save_torch_model(model, train_losses, val_losses) 

print('Training completed.')

# finally save the model
training_helper.save_torch_model(model, train_losses, val_losses) 



print(f"Model {model_name} training completed successfully")