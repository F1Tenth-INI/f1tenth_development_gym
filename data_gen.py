import sys

import tensorflow as tf
import csv
import numpy as np
import pandas as pd

from tqdm import trange

import yaml

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.computation_library import TensorFlowLibrary

from DataGen.Utilities import get_initial_states
from utilities.Recorder import create_csv_header

try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

# TODO: Get rid of for loop
# TODO: Save file as in Cartpole -> with split for train test, validate
# TODO: State initialisation - what not loaded, randomly initialized as in CP
# TODO: Interpolator

config = yaml.load(open('DataGen/config_data_gen.yml', "r"), Loader=yaml.FullLoader)

predictor = PredictorWrapper()
number_of_initial_states = config['number_of_initial_states']
trajectory_length = config['trajectory_length']
number_of_trajectories_per_initial_state = config['number_of_trajectories_per_initial_state']
dt = config['dt']  # Saving & Control Update

strategy = config['control_inputs']['strategy']

path_save_to = config['path_save_to']
file_to_save = 'Data_gen.csv'

load_initial_states_from_file = config['initial_states']['from_file']['load_from_file']
if load_initial_states_from_file:
    initial_states_from_file_features = config['initial_states']['from_file']['features']
    file_with_initial_states = config['initial_states']['from_file']['file_with_initial_states']
else:
    initial_states_from_file_features = None
    file_with_initial_states = None

np.random.seed(config['seed'])

data = None

predictor.configure(
    batch_size=number_of_initial_states,  # Number of initial states
    horizon=trajectory_length-1,  # Number of Steps per Trajectory: the trajectory length include also initial state
    dt=dt,
    computation_library=TensorFlowLibrary,
    predictor_specification="ODE_TF_default"
)

states, number_of_initial_states = get_initial_states(number_of_initial_states, file_with_initial_states, initial_states_from_file_features)
number_of_experiments = number_of_initial_states * number_of_trajectories_per_initial_state

# ----> Angular control
mu_ac, sigma_ac = 0, 0.9  # mean and standard deviation
# ----> Translational control
# mu_tc, sigma_tc = 0, 0.5  # mean and standard deviation
min_tc, max_tc = -10.0, 10.0

for i in trange(number_of_trajectories_per_initial_state):

    if strategy == 'constant input':
        # ---------------Constant input control for each trajectory-------------------------
        u0_dist = np.random.normal(mu_ac, sigma_ac, number_of_initial_states)
        # u1_dist = np.random.normal(mu_tc, sigma_tc, number_of_initial_states)
        u1_dist = np.random.uniform(min_tc, max_tc, number_of_initial_states)
        # Each row of controls is a control input to be followed along a trajectory for the corresponding initial state
        controls = np.column_stack((u0_dist, u1_dist)) # dim = [initial_states, 2]

        # In the following 2 line, we duplicate each control input (in rows direction, axis=0)
        control = np.repeat(controls, trajectory_length, axis=0)

    elif strategy == 'random input':
        # ----------------Random input control for each trajectory-------------------------

        # ----> Angular control
        u0_dist = np.random.normal(mu_ac, sigma_ac, number_of_initial_states*trajectory_length)

        # ----> Translational control
        u1_dist = np.random.uniform(min_tc, max_tc, number_of_initial_states*trajectory_length)

        # Each row of controls is a control input to be followed along a trajectory for the corresponding initial state
        controls = np.column_stack((u0_dist, u1_dist)) # dim = [initial_states, 2]
    else:
        raise Exception('{} is not a valid strategy for generating control inputs'.format(strategy))

    control = controls.reshape(number_of_initial_states, trajectory_length, 2) # dim = [initial_states,trajectory_length,2]

    s = tf.convert_to_tensor(states.reshape([number_of_initial_states, -1]), dtype=tf.float32)
    Q = tf.convert_to_tensor(control.reshape(number_of_initial_states, trajectory_length, 2), dtype=tf.float32)[:, :-1, :] # We don't need last control inputs, they will not be applied
    predictions = np.array(predictor.predict_tf(s, Q))  # dim = [initial_states, trajectory_length, 9]
    # !!!!!!!! Check if the prediction are ordered well in raw, ie each row is the prediction of the previous

    # Convert prediction to one big nd array and add control input next to each state transition
    predictions = predictions.reshape(number_of_initial_states*trajectory_length, 9)
    control = control.reshape(number_of_initial_states*trajectory_length, 2)
    control_with_predictions = np.column_stack((control, predictions))

    # Stack control_with_predictions into the data file
    if data is None:
        data = control_with_predictions
    else:
        data = np.row_stack((data, control_with_predictions))


time_axis = (np.arange(data.shape[0])*dt).reshape(-1, 1)

experiment_index = np.repeat(np.arange(number_of_experiments, dtype=np.int32), trajectory_length)

data = np.column_stack((time_axis, data))

column_names = ['time',
                'angular_control',
                'translational_control',
                'angular_vel_z',
                'linear_vel_x',
                'pose_theta',
                'pose_theta_cos',
                'pose_theta_sin',
                'pose_x',
                'pose_y',
                'slip_angle',
                'steering_angle',
                ]

df = pd.DataFrame(data, columns=column_names)
df['experiment_index'] = experiment_index

# Sort out experiments with anomalous angular velocity

idx = df.index[df['angular_vel_z'].abs() > 25]
problematic_experiments = df.loc[idx]['experiment_index'].unique()
df = df[~(df.experiment_index.isin(problematic_experiments))].reset_index(drop=True)

experiment_index = np.repeat(np.arange(df.shape[0]/trajectory_length, dtype=np.int32), trajectory_length)
df['experiment_index'] = experiment_index
pass

# Compute the number of nan in the data
print('Number of nan element in the generated Data:', np.count_nonzero(np.isnan(data)))

csv_path = create_csv_header(path_save_to, controller_name='Random input', dt=dt)
df.to_csv(csv_path, index=False, mode='a')

print('Finished data generation')


