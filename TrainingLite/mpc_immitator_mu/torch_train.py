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
print(f"Using device: {device}")
# exit()

# Path setup
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
from TrainingHelper import TrainingHelper, SequenceWindowDataset
from TorchNetworks import GRU as Network

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
from utilities.lidar_utils import *
from utilities.state_utilities import *


class ImmitationTraining:
    def __init__(self, model_name, dataset_name):

        # Setup experiment paths and parameters
        self.experiment_path = os.path.dirname(os.path.realpath(__file__))
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.reduce_dataset_size_by = 1
        self.number_of_epochs = 5
        self.batch_size = 16
        self.save_historgrams = False

        self.model_dir = os.path.join(self.experiment_path, 'models', model_name)
        self.dataset_dir = os.path.join(self.experiment_path, '..', 'Datasets', dataset_name)

        self.lidar_utils = LidarHelper()


        # Training Sequence parameters
        self.window_size = 320
        self.step_size = 1
        self.washout_period = 20
        self.shuffle_data = True


        # Training data
        state_cols = ["angular_vel_z", "linear_vel_x", "linear_vel_y", "steering_angle"]
        lidar_cols = self.lidar_utils.get_processed_ranges_names()

        wypt_x_cols = ["WYPT_REL_X_{:02d}".format(i) for i in range(0, 30)]
        wypt_y_cols = ["WYPT_REL_Y_{:02d}".format(i) for i in range(0, 30)]
        wypt_vx_cols = ["WYPT_VX_{:02d}".format(i) for i in range(0, 30)]
        self.input_cols = state_cols + wypt_x_cols + wypt_y_cols + wypt_vx_cols  # + lidar_cols
        self.output_cols = ["angular_control_calculated", "translational_control_calculated"]

        # Network parameters
        self.input_size = len(self.input_cols)
        self.output_size = len(self.output_cols)
        self.hidden_size = 128
        self.num_layers = 3


        self.training_helper = TrainingHelper(self.experiment_path, model_name, dataset_name)
        self.training_helper.create_and_clear_model_folder(self.model_dir)
        self.training_helper.save_training_scripts(os.path.realpath(__file__))


    def load_and_normalize_dataset(self, ):

        self.df, file_change_indices = self.training_helper.load_dataset(dataset_dir=self.dataset_dir, reduce_size_by=self.reduce_dataset_size_by)
        self.df = self.df.dropna()

        # Histogram creation
        if self.save_historgrams:
            self.training_helper.create_histograms(df, self.input_cols, self.output_cols)

        X = self.df[self.input_cols].to_numpy().astype(np.float32)
        y = self.df[self.output_cols].to_numpy().astype(np.float32)

        if hasattr(self, "input_scaler") and hasattr(self, "output_scaler"):
            # Already loaded scalers from previous training
            X = self.input_scaler.transform(X)
            y = self.output_scaler.transform(y)
        else:
            # First-time training — fit and save scalers
            X, y = self.training_helper.fit_trainsform_save_scalers(X, y)

        self.X = X
        self.y = y

        return X, y


    def create_model(self):
    
        self.model = Network(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        self.training_helper.save_network_metadata(self.input_cols, self.output_cols, self.model)
        self.model = self.model.to(device)


    def load_model(self):

        self.network_yaml, self.input_scaler, self.output_scaler = self.training_helper.load_network_meta_data_and_scalers()
        self.model = Network(self.input_size, self.hidden_size, self.output_size, self.num_layers)
        self.model.load_state_dict(
        torch.load(
            os.path.join(self.training_helper.model_dir, "model.pth"),
            map_location=device
            )
        )
        self.model.to(device)  # Move the model to the correct device
        print(f"Model {self.model_name} at {self.training_helper.model_dir} loaded successfully.")


    def train_network(self, X = None, y = None):
        
        if X is None or y is None:
            X, y = self.X, self.y

        # Split the dataset into training and validation sets
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X, y, test_size=0.1, random_state=42)


        # Create datasets and dataloaders
        train_dataset = SequenceWindowDataset(X_train_raw, y_train_raw, self.window_size, self.step_size)
        val_dataset = SequenceWindowDataset(X_val_raw, y_val_raw, self.window_size, self.step_size)

        self.data_loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_data)
        # self.data_loader_val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_data)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1, verbose=True, min_lr=1e-5)

        train_losses = []
        val_losses = []

        print(f"Training {self.model_name} started.")

        # Training loop
        for epoch in range(self.number_of_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in tqdm(self.data_loader_train, desc=f'Epoch {epoch+1}/{self.number_of_epochs}'):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                hidden = self.model.reset_hidden_state(batch_x.size(0))
                optimizer.zero_grad()
                output, hidden = self.model(batch_x, hidden)  # output: (batch_size, seq_length, output_size)
                
                # Apply washout period
                output_washed = output[:, self.washout_period:, :]  # Keep only the steps after the washout period
                target_washed = batch_y[:, self.washout_period:, :]
                
                loss = criterion(output_washed, target_washed)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(self.data_loader_train)
            train_losses.append(avg_train_loss)
            print(f'Epoch [{epoch + 1}/{self.number_of_epochs}], Train Loss: {avg_train_loss:.6f}')

            # Validation step
            self.model.eval()
            val_loss = 0.0
            if False: 
                with torch.no_grad():
                    for batch_x, batch_y in self.data_loader_val:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        hidden = self.model.reset_hidden_state(batch_x.size(0))
                        output, hidden = self.model(batch_x, hidden)
                        output_washed = output[:, self.washout_period:, :]
                        target_washed = batch_y[:, self.washout_period:, :]
                        loss = criterion(output_washed, target_washed)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(self.data_loader_val)
                val_losses.append(avg_val_loss)
                print(f'Validation Loss: {avg_val_loss:.6f}')
            
                # Adjust learning rate
                scheduler.step(avg_val_loss)
            
            # Save model after each epoch
            self.training_helper.save_torch_model(self.model, train_losses, val_losses)

        print('Training completed.')
        self.training_helper.save_torch_model(self.model, train_losses, val_losses)
        print(f"Model {self.model_name} training completed successfully")





if __name__ == "__main__":
    model_name = "04_08_RCA1_noise"
    dataset_name = "04_08_RCA1_noise"
    # immitation_training = ImmitationTraining(model_name, 'correction')
    # immitation_training.load_model()

    immitation_training = ImmitationTraining(model_name, dataset_name)
    immitation_training.create_model()
    immitation_training.load_and_normalize_dataset()
    immitation_training.train_network()