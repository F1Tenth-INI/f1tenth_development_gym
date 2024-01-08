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

import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import Adam


experiment_path = os.path.join("TrainingLite","NN_lidar_odom")


# Get a list of all .csv files in the directory
csv_files = glob.glob(experiment_path + '/data/*.csv')

# Read each .csv file and append it to the list 'dfs'
dfs = [pd.read_csv(f, comment='#') for f in csv_files]

# Concatenate all dataframes in the list 'dfs'
df = pd.concat(dfs, ignore_index=True)



# Augment data
lidar_keys = df.filter(like='LIDAR').columns
for key in lidar_keys:
    df[key + "_old"] = df[key].shift(1)
    # df["d_" + key ] = df[key].diff()

df['d_linear_vel_x'] = df['linear_vel_x'].diff()
# dfd_['pose_theta'] = df['pose_theta'].diff()
df['d_pose_theta_cos'] = df['pose_theta_cos'].diff()
df['d_pose_theta_sin'] = df['pose_theta_sin'].diff()
df['d_pose_x'] = df['pose_x'].diff()
df['d_pose_y'] = df['pose_y'].diff()
df['d_slip_angle'] = df['slip_angle'].diff()
df['d_steering_angle'] = df['steering_angle'].diff()

df['d_pose_x_car_frame'] = df['d_pose_x'] * df['pose_theta_cos'] + df['d_pose_y'] * df['pose_theta_sin']
df['d_pose_y_car_frame'] = -df['d_pose_x'] * df['pose_theta_sin'] + df['d_pose_y'] * df['pose_theta_cos']



# print first 3 rows of dataframe
print(df.head(3))

test = df.isnull().sum()
print("NAN numbers: ", test)

df = df.dropna()


# Select the input and output columns
lidar_keys = df.filter(like='LIDAR').columns
input_cols = lidar_keys
print("Input cols: ", input_cols)

output_cols = ['angular_vel_z', 'linear_vel_x' ]
print("output cols: ", output_cols)

time.sleep(0.1)  # Sleep for 50 milliseconds
for col in df[output_cols]:
    plt.figure()
    df[col].hist(bins=100)  # Increase the number of bins to 100
    plt.title(col)
    plt.savefig(experiment_path + '/figures/' + col + '.png')

    time.sleep(0.15)  # Sleep for 50 milliseconds
    
X = df[input_cols].to_numpy()
y = df[output_cols].to_numpy()

# for col in df[input_cols]:
#     plt.figure()
#     df[col].hist()
#     plt.title(col)



# Fit the scaler to the training data and transform it
input_scaler = MinMaxScaler(feature_range=(-1, 1))
output_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = output_scaler.fit_transform(y)




dump(input_scaler, experiment_path + '/models/input_scaler.joblib')
dump(output_scaler, experiment_path + '/models/output_scaler.joblib')



# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
model = Sequential()
model.add(Dense(64, input_dim=len(input_cols), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(len(output_cols)))

# Compile the model
model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.002))

# Train the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0005)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=1, shuffle=True, callbacks=[reduce_lr])

# Save the model
model_folder = experiment_path + '/models'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model.save(model_folder+'/my_model.h5')





