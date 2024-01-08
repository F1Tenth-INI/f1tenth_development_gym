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


# Concatenate all .csv files in the directory
csv_files = glob.glob(experiment_path + '/data/*.csv')
dfs = [pd.read_csv(f, comment='#') for f in csv_files]
df = pd.concat(dfs, ignore_index=True)



# Augment data
df['d_angular_vel_z'] = df['angular_vel_z'].diff()
df['d_linear_vel_x'] = df['linear_vel_x'].diff()
df['d_pose_theta_cos'] = df['pose_theta_cos'].diff()
df['d_pose_theta_sin'] = df['pose_theta_sin'].diff()
df['d_pose_x'] = df['pose_x'].diff()
df['d_pose_y'] = df['pose_y'].diff()
df['d_slip_angle'] = df['slip_angle'].diff()
df['d_steering_angle'] = df['steering_angle'].diff()

# Sort out invalid data
df = df[df['linear_vel_x'] >= 0.2]
df = df[df['d_pose_x'] <= 1.]
df = df[df['d_pose_y'] <= 1.]

# print first 3 rows of dataframe
print(df.head(3))

test = df.isnull().sum()
print("NAN numbers: ", test)

df = df.dropna()


# Select the input and output columns
input_cols = ["angular_vel_z","linear_vel_x","pose_theta_cos","pose_theta_sin","slip_angle","steering_angle","angular_control_calculated","translational_control_calculated"]
output_cols = ['d_pose_x', 'd_pose_y' ]

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
model.add(Dense(64, input_dim=len(input_cols), activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(len(output_cols)))

# Compile the model
model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.002))

# epoch callback for learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.0005)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=1, shuffle=False, callbacks=[reduce_lr])

# Save the model
model_folder = experiment_path + '/models'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model.save(model_folder+'/my_model.h5')





