import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import trange, tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path setup
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
from TrainingHelper import TrainingHelper
from TorchNetworks import GRU as Network

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from utilities.lidar_utils import *
from utilities.state_utilities import *


# Setup experiment paths and parameters
experiment_path = os.path.dirname(os.path.realpath(__file__))
model_name = "04_08_RCA1_noise"
dataset_name = "04_08_RCA1_noise"
number_of_epochs = 2
batch_size = 16
save_historgrams = False

shuffle_data = True

model_dir = os.path.join(experiment_path, 'models', model_name)
dataset_dir = os.path.join(experiment_path, '..', 'Datasets', dataset_name)

training_helper = TrainingHelper(experiment_path, model_name, dataset_name)
training_helper.create_and_clear_model_folder(model_dir)
training_helper.save_training_scripts(os.path.realpath(__file__))

lidar_utils = LidarHelper()

df, file_change_indices = training_helper.load_dataset(reduce_size_by=1)

df = df.dropna()

# Define input and output columns
state_cols = ["angular_vel_z", "linear_vel_x", "linear_vel_y", "steering_angle"]
lidar_cols = lidar_utils.get_processed_ranges_names()

wypt_x_cols = ["WYPT_REL_X_{:02d}".format(i) for i in range(0, 30)]
wypt_y_cols = ["WYPT_REL_Y_{:02d}".format(i) for i in range(0, 30)]
wypt_vx_cols = ["WYPT_VX_{:02d}".format(i) for i in range(0, 30)]
input_cols = state_cols + wypt_x_cols + wypt_y_cols + wypt_vx_cols + lidar_cols
output_cols = ["angular_control_calculated", "translational_control_calculated"]

df = df.dropna()

if save_historgrams:
    training_helper.create_histograms(df, input_cols, output_cols)

# Scaling input and output data
X = df[input_cols].to_numpy().astype(np.float32)
y = df[output_cols].to_numpy().astype(np.float32)

X, y = training_helper.fit_trainsform_save_scalers(X, y)

# Window parameters
window_size = 300
step_size = 1
washout_period = 20

# Split the dataset into training and validation sets
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom streamed windowing dataset
class SequenceWindowDataset(Dataset):
    def __init__(self, X, y, window_size, step_size=1):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.step_size = step_size
        self.indices = list(range(0, len(X) - window_size + 1, step_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x_window = self.X[i:i + self.window_size]
        y_window = self.y[i:i + self.window_size]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_window, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = SequenceWindowDataset(X_train_raw, y_train_raw, window_size, step_size)
val_dataset = SequenceWindowDataset(X_val_raw, y_val_raw, window_size, step_size)

data_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_data)
data_loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_data)

# Model setup
input_size = len(input_cols)
output_size = len(output_cols)
hidden_size = 128
num_layers = 3

model = Network(input_size, hidden_size, output_size, num_layers)
training_helper.save_network_metadata(input_cols, output_cols, model)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

train_losses = []
val_losses = []

print(f"Training {model_name} started.")

# Training loop
for epoch in range(number_of_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in tqdm(data_loader_train, desc=f'Epoch {epoch+1}/{number_of_epochs}'):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        hidden = model.reset_hidden_state(batch_x.size(0))
        optimizer.zero_grad()
        output, hidden = model(batch_x, hidden)  # output: (batch_size, seq_length, output_size)
        
        # Apply washout period
        output_washed = output[:, washout_period:, :]  # Keep only the steps after the washout period
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
    
    # Adjust learning rate
    scheduler.step(avg_val_loss)
    
    # Save model after each epoch
    training_helper.save_torch_model(model, train_losses, val_losses)

print('Training completed.')
training_helper.save_torch_model(model, train_losses, val_losses)
print(f"Model {model_name} training completed successfully")
