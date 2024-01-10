import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import glob
import numpy as np
import time
import yaml

import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam

experiment_path =os.path.dirname(os.path.realpath(__file__))



# Get a list of all CSV files in the 'data' directory
csv_files = glob.glob(experiment_path + '/data/*.csv')

# List to hold dataframes
df_list = []

for file in csv_files:
    df = pd.read_csv(file, comment='#')
    
    # Augment data
    df['pose_theta_cos'] = np.cos(df['pose_theta'])
    df['pose_theta_sin'] = np.sin(df['pose_theta'])
    df['d_time'] = df['time'].diff()

    # Derivatives of states
    state_variables = ['angular_vel_z', 'linear_vel_x', 'pose_theta_cos', 'pose_theta_sin', 'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
    for var in state_variables:
        df['d_' + var] = df[var].diff() / df['d_time']

    # Sort out invalid data
    df = df[df['d_angular_vel_z'] <= 60]
    df = df[df['linear_vel_x'] <= 20]
    df = df[df['d_pose_x'] <= 20.]
    df = df[df['d_pose_y'] <= 20.]

    df_list.append(df)

# Concatenate all dataframes in the list into a single dataframe
df = pd.concat(df_list, ignore_index=True)

# print first 3 rows of dataframe
print(df.head(3))

test = df.isnull().sum()
print("NAN numbers: ", test)

print("Number of data points: ", len(df))

df = df.dropna()


# Select the input and output columns
input_cols = ["angular_vel_z","linear_vel_x","pose_theta_cos","pose_theta_sin","slip_angle","steering_angle","angular_control_calculated","translational_control_calculated"]
output_cols = ["d_angular_vel_z","d_linear_vel_x","d_pose_theta_cos","d_pose_theta_sin", "d_pose_x", "d_pose_y", "d_slip_angle","d_steering_angle", ]

print("Input cols: ", input_cols)
print("output cols: ", output_cols)

cols_dict = {"input_cols": input_cols, "output_cols": output_cols}
with open(experiment_path+'/models/network.yaml', 'w') as file:
    yaml.dump(cols_dict, file)
    
    
# Plot data for NN
time.sleep(0.1)  # Sleep for 50 milliseconds

for col in df[input_cols]:
    plt.figure()
    df[col].hist(bins=100)  # Increase the number of bins to 100
    plt.title(col)
    plt.savefig(experiment_path + '/figures/' + col + '.png')

    time.sleep(0.15)  # Sleep for 50 milliseconds
    
for col in df[output_cols]:
    plt.figure()
    df[col].hist(bins=100)  # Increase the number of bins to 100
    plt.title(col)
    plt.savefig(experiment_path + '/figures/' + col + '.png')

    time.sleep(0.15)  # Sleep for 50 milliseconds
    
    
X = df[input_cols].to_numpy()
y = df[output_cols].to_numpy()


# Fit the scaler to the training data and transform it
input_scaler = MinMaxScaler(feature_range=(-1, 1))
output_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = output_scaler.fit_transform(y)

# save the scaler for denormalization
dump(input_scaler, experiment_path + '/models/input_scaler.joblib')
dump(output_scaler, experiment_path + '/models/output_scaler.joblib')



# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
model = Sequential()
model.add(Dense(64, input_dim=len(input_cols), activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(len(output_cols)))

# Compile the model
model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.002))

# epoch callback for learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, min_lr=0.00001)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=8, shuffle=False, callbacks=[reduce_lr])

# Save the model
model_folder = experiment_path + '/models'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model.save(model_folder+'/my_model.h5')





