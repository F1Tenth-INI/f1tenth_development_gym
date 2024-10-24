import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from  TrainingHelper import TrainingHelper

from tqdm import trange


# Setup experiment paths and parameters
experiment_path = os.path.dirname(os.path.realpath(__file__))
model_name = "tLSTM7_b16_files_shuf_mpc2_reduce_lr_test"
dataset_name = "_MPC_Noise2"
seq_length = 1
washout_steps = 0
number_of_epochs = 20
batch_size = 16


shuffle_data = True
train_on_output_sequences = False

model_dir = os.path.join(experiment_path,'models', model_name)
dataset_dir = os.path.join(experiment_path,'..','Datasets', dataset_name)


training_helper = TrainingHelper(experiment_path, model_name, dataset_name)
training_helper.create_and_clear_model_folder(model_dir)
training_helper.save_training_scripts(os.path.realpath(__file__))


df, file_change_indices = training_helper.load_dataset()

batches = []

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


X, y = training_helper.fit_trainsform_save_scalers(X, y)

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

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.num_layers = 2
        self.hidden_size = 64
        
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Take the output at the last time step
        return out, hidden

    def reset_hidden_state(self, batch_size):
        # Use model's parameters to get the device
        device = next(self.parameters()).device
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


input_size = len(input_cols)
output_size = len(output_cols)
hidden_size = 64


# Initialize the model
model = Network(input_size, output_size)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


# Cut training data such that it is divisible by batch size (all batches have the same size)
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
                    break  # discard batches that overlap file borders
            
            optimizer.zero_grad()
            outputs, hidden = model(X_batch, hidden)  # Pass the hidden state from the previous batch
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()
            hidden = (hidden[0].detach(), hidden[1].detach())
            
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

            hidden_val = (hidden_val[0].detach(), hidden_val[1].detach())

    val_loss /= len(X_val) / batch_size
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{number_of_epochs}], Validation Loss: {val_loss}")
    
    # Reduce LR
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != current_lr:
        print(f"Learning rate changed from {current_lr} to {new_lr}")

    # Save the model after every epoch
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

# finally save the model
torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

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