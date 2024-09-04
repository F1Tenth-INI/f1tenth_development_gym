import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import glob
import numpy as np
import time
import yaml
import zipfile
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
import shutil
from keras.optimizers import Adam
from tensorflow.keras import backend as K

experiment_path = os.path.dirname(os.path.realpath(__file__))

model_name = "GRU2-Example"

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

# Get a list of all CSV files in the 'data' directory
csv_files = glob.glob(experiment_path + '/data/*.csv')

# Select the input and output columns
input_cols = ["angular_vel_z", "linear_vel_x", "pose_theta_sin", "pose_theta_cos", "steering_angle", 'slip_angle', 'angular_control_applied', 'translational_control_applied']
output_cols = ["d_angular_vel_z", "d_linear_vel_x", "d_pose_theta_sin", "d_pose_theta_cos", 'd_steering_angle', 'd_slip_angle', 'd_pose_x', 'd_pose_y']

print("Input cols: ", input_cols)
print("Output cols: ", output_cols)

cols_dict = {"input_cols": input_cols, "output_cols": output_cols}
with open(model_dir + '/network.yaml', 'w') as file:
    yaml.dump(cols_dict, file)

def load_and_process_data(files):
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(file, comment='#')
        except Exception as e:
            print(f"An exception occurred for file: {file}")
            print(f"Exception: {e}")
            continue

        # Augment data
        df['pose_theta_cos'] = np.cos(df['pose_theta'])
        df['pose_theta_sin'] = np.sin(df['pose_theta'])
        df['d_time'] = df['time'].diff()

        # Derivatives of states
        state_variables = ['angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
        for var in state_variables:
            df['d_' + var] = df[var].diff().shift(-1) / df['d_time']

        # Sort out invalid data
        df = df[df['d_angular_vel_z'] <= 60]
        df = df[df['linear_vel_x'] <= 20]
        df = df[df['d_pose_x'] <= 20.]
        df = df[df['d_pose_y'] <= 20.]

        df = df.dropna()

        df_list.append(df)

    return df_list

df_list = load_and_process_data(csv_files)

# Concatenate all dataframes for scaler fitting
df = pd.concat(df_list, ignore_index=True)

# Convert data to numpy arrays
X = df[input_cols].to_numpy()
y = df[output_cols].to_numpy()

# Fit the scaler to the data and transform it
input_scaler = MinMaxScaler(feature_range=(-1, 1))
output_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = output_scaler.fit_transform(y)

# Save the scaler for denormalization
dump(input_scaler, model_dir + '/input_scaler.joblib')
dump(output_scaler, model_dir + '/output_scaler.joblib')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length + 1):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

seq_length = 50  # Use a meaningful sequence length

# Define the GRU network
model = Sequential()
model.add(GRU(64, input_shape=(seq_length, len(input_cols)), return_sequences=True, stateful=False))
model.add(GRU(64, return_sequences=False, stateful=False))
model.add(Dense(len(output_cols)))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.002))

# Custom training loop
epochs = 10
batch_size = 8

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for df in df_list:
        X = df[input_cols].to_numpy()
        y = df[output_cols].to_numpy()

        X = input_scaler.transform(X)
        y = output_scaler.transform(y)

        X_sequences = create_sequences(X, seq_length)
        y_sequences = y[seq_length - 1:]

        model.reset_states()

        model.fit(X_sequences, y_sequences, epochs=1, batch_size=batch_size, shuffle=False, verbose=1)
        model.save(f"{model_dir}/my_model_epoch_{epoch + 1}.keras")

# Save the model
model.save(model_dir + '/my_model.keras')
