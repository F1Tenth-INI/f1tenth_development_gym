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

from tqdm import trange


# Setup experiment paths and parameters
experiment_path = os.path.dirname(os.path.realpath(__file__))
model_name = "tLSTM7_b16_files_shuf_mpc2"
dataset_name = "_MPC_Noise2"
seq_length = 1
washout_steps = 0
number_of_epochs = 20
batch_size = 16


shuffle_data = True
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

    df['source'] = file
    
    # if i % 4 == 0:
    #     df_list.append(df)
    df_list.append(df)


df = pd.concat(df_list, ignore_index=True)
file_change_indices = df.index[df['source'].ne(df['source'].shift())].tolist()


batches = []


# batch = []
# for i in range(3):  # Adjust the range as needed
#     batch.append(df.iloc[i][index])


# for i, df in enumerate(df_list):
#     batch.append(df.iloc[i][index])
    
    

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

losses = []

class GRUNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(GRUNetwork, self).__init__()

        self.num_layers = 2
        self.hidden_size = 64
        
        self.gru1 = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers , batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # print(hidden)
        out, hidden = self.gru1(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the output at the last time step
        return out, hidden

    def reset_hidden_state(self, batch_size):
        # Use model's parameters to get the device
        device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)




      
input_size = len(input_cols)
output_size = len(output_cols)
hidden_size = 64


# Initialize the model
model = GRUNetwork(input_size, output_size)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Cut training data suchthat it is dividable by batch size (all batches have the same size)
num_batches = len(X_train) // batch_size
X_train = X_train[:num_batches * batch_size]
y_train = y_train[:num_batches * batch_size]

# Training loop
train_losses = []
val_losses = []

print(f"Training {model_name} started.")


def shuffle_by_files(X_train, y_train, file_change_indices):
    # Function to split arrays at given indices
    def split_at_indices(arr, indices):
        return np.split(arr, indices)

    # Split X_train and Y_train at file_change_indices
    X_train_splits = split_at_indices(X_train, file_change_indices)
    y_train_splits = split_at_indices(y_train, file_change_indices)

    # Shuffle the sequences
    shuffled_indices = np.random.permutation(len(X_train_splits))
    X_train_shuffled = [X_train_splits[i] for i in shuffled_indices]
    y_train_shuffled = [y_train_splits[i] for i in shuffled_indices]

    # Concatenate the shuffled sequences back together
    X_train = np.concatenate(X_train_shuffled)
    y_train = np.concatenate(y_train_shuffled)

    # Update file_change_indices
    file_change_indices = np.cumsum([len(seq) for seq in X_train_shuffled[:-1]])
    
    return X_train, y_train, file_change_indices

# Epoch loop
for epoch in range(number_of_epochs):
    
    model.train()

    hidden = model.reset_hidden_state(batch_size)  # Initialize hidden state once per epoch

    if shuffle_data:
        X_train, y_train, file_change_indices = shuffle_by_files(X_train, y_train, file_change_indices)
    
    
    
    # Batch loop
    with trange(0, len(X_train), batch_size, desc=f"Epoch {epoch+1}/{number_of_epochs}") as pbar:
        for i in pbar:
            
            
            num_batches = len(X_train) / batch_size
            
            X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(model.fc.weight.device)
            y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(model.fc.weight.device)

    
            
            for value in file_change_indices:
                if i <= value <= i+batch_size:
                    hidden = model.reset_hidden_state(X_batch.size(0))
                    break # discard batches that overlap file borders
                    
            
            optimizer.zero_grad()
            outputs, hidden = model(X_batch, hidden)  # Pass the hidden state from the previous batch
            loss = criterion(outputs, y_batch)
            


            loss.backward()
            optimizer.step()
            hidden = hidden.detach()
            
            # print losses during training
            pbar.set_postfix(loss=f"{loss.item():.5f}")
            
            train_losses.append(loss.item())
            

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
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{number_of_epochs}], Validation Loss: {val_loss}")
    torch.save(model.state_dict(), os.path.join(model_dir, "gru_model.pth"))

# Save the model
torch.save(model.state_dict(), os.path.join(model_dir, "gru_model.pth"))

# Plot the loss values
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.savefig(os.path.join(model_dir, 'loss_plot.png'))

print(f"Model {model_name} training completed successfully")



