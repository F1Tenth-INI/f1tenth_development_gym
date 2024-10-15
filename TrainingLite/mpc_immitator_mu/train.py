import os
import tensorflow as tf

# Anti logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from keras.layers import GRU, LSTM, Dense, TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from keras.layers import Reshape


from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import glob
import numpy as np
import time
import yaml
import zipfile

import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import shutil

from keras.optimizers import Adam

experiment_path =os.path.dirname(os.path.realpath(__file__))

model_name = "GRU_Test"
seq_length = 2
washout_steps = 0

number_of_epochs = 3
shuffle = False

train_on_output_sequences = True


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

# List to hold dataframes
df_list = []

for i, file in enumerate(csv_files):
    df = pd.read_csv(file, comment='#')
    
    # Augment data
    df['pose_theta_cos'] = np.cos(df['pose_theta'])
    df['pose_theta_sin'] = np.sin(df['pose_theta'])
    df['d_time'] = df['time'].diff()

    # Derivatives of states
    state_variables = ['angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 'pose_theta', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']
    for var in state_variables:
        df['d_' + var] = df[var].diff() / df['d_time']

    # Sort out invalid data
    df = df[df['d_angular_vel_z'] <= 60]
    df = df[df['linear_vel_x'] <= 20]
    df = df[df['d_pose_x'] <= 20.]
    df = df[df['d_pose_y'] <= 20.]
    
    
    # Append the dataframe to the list (size reduction)
    if i % 2 == 0:
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

state_cols = ["angular_vel_z","linear_vel_x","pose_theta","steering_angle", "slip_angle", ]
wypt_x_cols = ["WYPT_REL_X_{:02d}".format(i) for i in range(0, 20, 1)]
wypt_y_cols = ["WYPT_REL_Y_{:02d}".format(i) for i in range(0, 20, 1)]
wypt_vx_cols = ["WYPT_VX_{:02d}".format(i) for i in range(0, 20, 1)]

input_cols = state_cols + wypt_x_cols + wypt_y_cols + wypt_vx_cols


output_cols = ["angular_control_calculated", "translational_control_calculated", "mu" ]
shift_cols =  ["angular_control_calculated", "translational_control_calculated", ]

# Shift the output columns by -3
for col in shift_cols:
    df[col] = df[col].shift(-4)

# Remove rows with NaN values after shifting
df = df.dropna()


print("Input cols: ", input_cols)
print("output cols: ", output_cols)

nn_dict = {"input_cols": input_cols, "output_cols": output_cols, "sequence_length": seq_length, "washout_steps": washout_steps, "number_of_epochs": number_of_epochs}
with open(model_dir+'/network.yaml', 'w') as file:
    yaml.dump(nn_dict, file)
    
    
# Plot data for NN
time.sleep(0.1)  # Sleep for 50 milliseconds


# for col in df[input_cols]:
#     plt.figure()
#     df[col].hist(bins=100)  # Increase the number of bins to 100
#     plt.title(col)
#     plt.savefig(experiment_path + '/figures/' + col + '.png')

#     time.sleep(0.15)  # Sleep for 50 milliseconds
    
# for col in df[output_cols]:
#     plt.figure()
#     df[col].hist(bins=100)  # Increase the number of bins to 100
#     plt.title(col)
#     plt.savefig(experiment_path + '/figures/' + col + '.png')

#     time.sleep(0.15)  # Sleep for 50 milliseconds
    
    
X = df[input_cols].to_numpy()
y = df[output_cols].to_numpy()


# Fit the scaler to the training data and transform it
input_scaler = MinMaxScaler(feature_range=(-1, 1))
output_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = output_scaler.fit_transform(y) # (dataset_length, 3)

# save the scaler for denormalization
dump(input_scaler, model_dir+'/input_scaler.joblib')
dump(output_scaler, model_dir+'/output_scaler.joblib')



# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data)-seq_length+1):
        x = data[i:(i+seq_length)]
        xs.append(x)
    return np.array(xs)


features = X_train.shape[1]  # number of features

# Create sequences from the training and validation data
X_train = create_sequences(X_train, seq_length) # (dataset_length, seq_length, 65)
X_val = create_sequences(X_val, seq_length) # (dataset_length, seq_length, 65)


# For sequence training
if(train_on_output_sequences):
    y_train = create_sequences(y_train, seq_length) 
    y_val =  create_sequences(y_val, seq_length) 
    
# For single output trairing
if(not train_on_output_sequences):
    y_train = y_train[seq_length-1:]  # (dataset_length, 3)
    y_val = y_val[seq_length-1:]   # (dataset_length, 3)




# Shuffle sequences
if(shuffle):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)


# Create model folder
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Define the GRU network
model = Sequential()
model.add(LSTM(64, input_shape=(seq_length, features), return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(len(output_cols)))
# model.add(TimeDistributed(Dense(len(output_cols))))



# Dense Training
# X_train = X_train.reshape(-1, 65)
# X_val = X_val.reshape(-1, 65)
# model = Sequential()
# model.add(Dense(64, input_shape=(features,), activation='tanh'))
# model.add(Dense(64, activation='tanh'))
# model.add(Dense(len(output_cols)))


# Custom loss function
def custom_mse_loss_with_washout(y_true, y_pred):

    # Ignore the first 'washout_steps' timesteps in both y_true and y_pred
    if(washout_steps == 0):
        return mean_squared_error(y_true, y_pred)
    
    y_true_washout = y_true[:, washout_steps:, :]
    y_pred_washout = y_pred[:, washout_steps:, :]

    squared_diff = K.square(y_true_washout - y_pred_washout)

    return K.mean(squared_diff, axis=-1)

# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.002))

# epoch callback for learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, min_lr=0.00001)

# ModelCheckpoint callback to save the model after each epoch
checkpoint = ModelCheckpoint(filepath=model_dir+'/my_model.keras', save_weights_only=False, save_best_only=False, verbose=1)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=number_of_epochs, batch_size=8, shuffle=True, callbacks=[reduce_lr, checkpoint])

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(model_dir+'/training_validation_loss.png')

model.save(model_dir+'/my_model.keras')

