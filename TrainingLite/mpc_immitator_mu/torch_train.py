import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import zipfile
from joblib import dump
import yaml



# Setup experiment paths and parameters
experiment_path = os.path.dirname(os.path.realpath(__file__))
model_name = "tGRU_6_stateful4"
dataset_name = "_MPC_Noise2"
seq_length = 1
washout_steps = 0
number_of_epochs = 5
shuffle_data = False
train_on_output_sequences = False

base_dir = os.path.join(experiment_path, 'models')
model_dir = os.path.join(base_dir, model_name)
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)

# Zip the training script for reconstruction
this_file_path = os.path.realpath(__file__)
zip_file_path = os.path.join(model_dir, 'train.py.zip')
with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(this_file_path, arcname=os.path.basename(this_file_path))

# Load CSV files and preprocess data
csv_files = glob.glob(experiment_path + '/../Datasets/' + dataset_name + '/*.csv')

df_list = []
for i, file in enumerate(csv_files):
    df = pd.read_csv(file, comment='#')
    df['pose_theta_cos'] = np.cos(df['pose_theta'])
    df['pose_theta_sin'] = np.sin(df['pose_theta'])
    df['d_time'] = df['time'].diff()

    state_variables = ['angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
    for var in state_variables:
        df['d_' + var] = df[var].diff() / df['d_time']

    df = df[df['d_angular_vel_z'] <= 60]
    df = df[df['linear_vel_x'] <= 20]
    df = df[df['d_pose_x'] <= 20.]
    df = df[df['d_pose_y'] <= 20.]

    # if i % 2 == 0:
    #     df_list.append(df)
    df_list.append(df)


df = pd.concat(df_list, ignore_index=True)
df = df.dropna()

# Define input and output columns
state_cols = ["angular_vel_z", "linear_vel_x", "pose_theta", "steering_angle", "slip_angle"]
wypt_x_cols = ["WYPT_REL_X_{:02d}".format(i) for i in range(0, 20)]
wypt_y_cols = ["WYPT_REL_Y_{:02d}".format(i) for i in range(0, 20)]
wypt_vx_cols = ["WYPT_VX_{:02d}".format(i) for i in range(0, 20)]
input_cols = state_cols + wypt_x_cols + wypt_y_cols + wypt_vx_cols
output_cols = ["angular_control_calculated", "translational_control_calculated", "mu"]

for col in ["angular_control_calculated", "translational_control_calculated"]:
    df[col] = df[col].shift(-4)

df = df.dropna()


cols_dict = {"input_cols": input_cols, "output_cols": output_cols}
with open(model_dir+'/network.yaml', 'w') as file:
    yaml.dump(cols_dict, file)
    

# Scaling input and output data
X = df[input_cols].to_numpy()
y = df[output_cols].to_numpy()

input_scaler = MinMaxScaler(feature_range=(-1, 1))
output_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = output_scaler.fit_transform(y)


# save the scaler for denormalization
dump(input_scaler, model_dir+'/input_scaler.joblib')
dump(output_scaler, model_dir+'/output_scaler.joblib')

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

features = X_train.shape[1]  # Number of input features

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRUNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.gru1(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the output at the last time step
        return out, hidden

    def reset_hidden_state(self, batch_size):
        # Use model's parameters to get the device
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)




      
input_size = X_train.shape[1]
output_size = len(output_cols)
hidden_size = 64
batch_size = 32


# Initialize the model
model = GRUNetwork(input_size, hidden_size, output_size, num_layers=2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(number_of_epochs):
    model.train()
    
    hidden = model.reset_hidden_state(batch_size)  # Initialize hidden state once per epoch
    
    # Loop over batches
    for i in range(0, len(X_train), batch_size):
        X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(model.fc.weight.device)
        y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(model.fc.weight.device)

        # Adjust hidden state size to match current batch size (in case the last batch is smaller)
        hidden = model.reset_hidden_state(X_batch.size(0))  # Adjust batch size dynamically

        optimizer.zero_grad()
        outputs, hidden = model(X_batch, hidden)  # Pass the hidden state from the previous batch
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        # Detach hidden state to prevent backprop through time
        hidden = hidden.detach()

    # Validation step
    model.eval()
    val_loss = 0.0
    hidden_val = model.reset_hidden_state(batch_size)  # Initialize hidden state for validation
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            X_batch_val = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(model.fc.weight.device)
            y_batch_val = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(model.fc.weight.device)

            # Adjust hidden state size for validation batch size
            hidden_val = model.reset_hidden_state(X_batch_val.size(0))

            val_outputs, hidden_val = model(X_batch_val, hidden_val)
            val_loss += criterion(val_outputs, y_batch_val).item()

            hidden_val = hidden_val.detach()

    val_loss /= len(X_val) / batch_size
    print(f"Epoch [{epoch+1}/{number_of_epochs}], Validation Loss: {val_loss}")

# Save the model
torch.save(model.state_dict(), os.path.join(model_dir, "gru_model.pth"))
