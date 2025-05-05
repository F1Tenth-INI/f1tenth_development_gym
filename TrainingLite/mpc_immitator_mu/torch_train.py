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

from tqdm import trange, tqdm

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
model_name = "03_26_RCA1-1"
dataset_name = "03_26_RCA1"
number_of_epochs = 30
batch_size = 64
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
df = df.dropna()

# Define input and output columns
state_cols = ["angular_vel_z", "linear_vel_x", "linear_vel_y", "steering_angle"]
lidar_cols = lidar_utils.get_processed_ranges_names()

wypt_x_cols = ["WYPT_REL_X_{:02d}".format(i) for i in range(0, 30)]
wypt_y_cols = ["WYPT_REL_Y_{:02d}".format(i) for i in range(0, 30)]
wypt_vx_cols = ["WYPT_VX_{:02d}".format(i) for i in range(0, 30)]
input_cols = state_cols + wypt_x_cols + wypt_y_cols + wypt_vx_cols + lidar_cols
output_cols = ["angular_control_calculated", "translational_control_calculated",]

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

window_size = 100
step_size = 5
washout_period = 30  # First 20 timesteps ignored for loss

# Initialize the model
model = Network(input_size, hidden_size, output_size, num_layers)
training_helper.save_network_metadata(input_cols, output_cols, model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

X, y, new_file_change_indices = training_helper.shuffle_dataset_by_chunks(X, y, batch_size, window_size, step_size)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

dataset_train = TensorDataset(X_train, y_train)
dataset_val = TensorDataset(X_val, y_val)

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# Training loop
train_losses = []
val_losses = []

print(f"Training {model_name} started.")

for epoch in range(number_of_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in tqdm(data_loader_train, desc=f'Epoch {epoch+1}/{number_of_epochs}'):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        hidden = model.reset_hidden_state(batch_x.size(0))
        optimizer.zero_grad()
        output, hidden = model(batch_x, hidden)
        
        output_washed = output[:, washout_period:, :]
        target_washed = batch_y[:, washout_period:, :]
        
        loss = criterion(output_washed, target_washed)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(data_loader_train)
    train_losses.append(avg_train_loss)
    print(f'Epoch [{epoch + 1}/{number_of_epochs}], Train Loss: {avg_train_loss:.6f}')
    
    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in data_loader_val:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            hidden = model.reset_hidden_state(batch_x.size(0))
            output, hidden = model(batch_x, hidden)
            output_washed = output[:, washout_period:, :]
            target_washed = batch_y[:, washout_period:, :]
            loss = criterion(output_washed, target_washed)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(data_loader_val)
    val_losses.append(avg_val_loss)
    print(f'Validation Loss: {avg_val_loss:.6f}')
    
    # Adjust learning rate based on validation loss
    scheduler.step(avg_val_loss)
    
    # Save the model after every epoch
    training_helper.save_torch_model(model, train_losses, val_losses) 

print('Training completed.')
training_helper.save_torch_model(model, train_losses, val_losses) 
print(f"Model {model_name} training completed successfully")
