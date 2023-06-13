import os

import tensorflow as tf
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

def generate_random_inputs(config, total_number_of_trajectories, trajectory_length):
    # ----> Angular control
    mu_ac, sigma_ac = config['control_inputs']['angular_control_range']  # mean and standard deviation
    # ----> Translational control
    # mu_tc, sigma_tc = 0, 0.5  # mean and standard deviation
    min_tc, max_tc = config['control_inputs']['translational_control_range']
    strategy = config['control_inputs']['strategy']

    if strategy == 'constant':
        # ---------------Constant input control for each trajectory-------------------------
        u_angular = np.random.normal(mu_ac, sigma_ac, total_number_of_trajectories)
        u_translational = np.random.uniform(min_tc, max_tc, total_number_of_trajectories)
        # Each row of controls is a control input to be followed along a trajectory for the corresponding initial state
        controls = np.column_stack((u_angular, u_translational)) # dim = [initial_states, 2]
        controls = np.expand_dims(controls, 1)
        # In the following 2 line, we duplicate each control input (in rows direction, axis=0)
        control = np.repeat(controls, trajectory_length, axis=1)

    elif strategy == 'random':
        # ----------------Random input control for each trajectory-------------------------
        u_angular = np.random.normal(mu_ac, sigma_ac, total_number_of_trajectories*trajectory_length)
        u_translational = np.random.uniform(min_tc, max_tc, total_number_of_trajectories*trajectory_length)
        
        # TODO: Decide if this filter is good? --> Add +1 to length above to use!
        #u_angular = np.convolve(u_angular, np.ones(2), 'valid') / 2
        #u_translational =  np.convolve(u_translational, np.ones(2), 'valid') / 2
        
        # Each row of controls is a control input to be followed along a trajectory for the corresponding initial state
        controls = np.column_stack((u_angular, u_translational)) # dim = [initial_states, 2]
        control = controls.reshape(total_number_of_trajectories, trajectory_length, 2) # dim = [initial_states,trajectory_length,2]
    else:
        raise Exception('{} is not a valid strategy for generating control inputs'.format(strategy))

    return control

def save_dataframe_to_csv(df, config, run_for_ML_Pipeline, record_path, total_number_of_trajectories):
    index = 0
    if run_for_ML_Pipeline:
        path = record_path + '/Train/'
    elif record_path is None:
        path = config["path_save_to"]
    else:
        path = record_path

    strategy = config['control_inputs']['strategy']
    dt = config['dt']

    print('Saving data...')
    for experiment in trange(total_number_of_trajectories):
        if run_for_ML_Pipeline:
            split = config['split']
            if experiment == int(split[0]*total_number_of_trajectories):
                path = record_path + "/Validate/"
                index = 0
            elif experiment == int((split[0]+split[1])*total_number_of_trajectories):
                path = record_path + "/Test/"
                index = 0

            try:
                os.makedirs(path)
            except:
                pass

        csv_path = f'{path}Trajectory_{strategy}-{str(index)}.csv'
        if not df[df.experiment_index == experiment].empty:
            csv_path = create_csv_header(path, controller_name='Random input', dt=dt, csv_name=csv_path)
            df[df.experiment_index == experiment].to_csv(csv_path, index=False, mode='a')
        index += 1

def prediction_to_df(predictions, control, config, total_number_of_trajectories, trajectory_length):
    dt = config['dt']
    data = np.concatenate((control, predictions), axis=2)
    grid = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    time = grid[1][:, :, np.newaxis] * dt
    experiment_index = grid[0][:, :, np.newaxis]

    data = np.concatenate((time, data, experiment_index), axis=2)
    data = data.reshape(total_number_of_trajectories * trajectory_length, data.shape[2])

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
                    'experiment_index'
                    ]

    df = pd.DataFrame(data, columns=column_names)
    return (data, df)

def clean_dataframe(df):
    idx = df.index[df['angular_vel_z'].abs() > 25]
    problematic_experiments = df.loc[idx]['experiment_index'].unique()
    print(f'Found {len(problematic_experiments)} problematic experiments!: {problematic_experiments}')
    df = df[~(df.experiment_index.isin(problematic_experiments))].reset_index(drop=True)

    return df

def get_initial_state_configuration(config):
    load_initial_states_from_file = config['initial_states']['from_file']['load_from_file']
    if load_initial_states_from_file:
        initial_states_from_file_features = config['initial_states']['from_file']['features']
        file_with_initial_states = config['initial_states']['from_file']['file_with_initial_states']
    else:
        initial_states_from_file_features = None
        file_with_initial_states = None

    return (initial_states_from_file_features, file_with_initial_states)

def run_data_generator(run_for_ML_Pipeline=False, record_path=None):

    # TODO: Interpolator

    config = yaml.load(open('DataGen/config_data_gen.yml', "r"), Loader=yaml.FullLoader)
    number_of_initial_states = config['number_of_initial_states']
    trajectory_length = config['trajectory_length']
    number_of_trajectories_per_initial_state = config['number_of_trajectories_per_initial_state']
    
    initial_states_from_file_features, file_with_initial_states = get_initial_state_configuration(config)
    random_initial_states = config['initial_states']['random']

    np.random.seed(config['seed'])

    states, number_of_initial_states = get_initial_states(number_of_initial_states, random_initial_states, 
                                                          file_with_initial_states, initial_states_from_file_features)
    states = np.repeat(states, repeats=number_of_trajectories_per_initial_state, axis=0)
    total_number_of_trajectories = number_of_initial_states * number_of_trajectories_per_initial_state
    
    predictor = PredictorWrapper()
    predictor.configure(
        batch_size=total_number_of_trajectories,
        horizon=trajectory_length-1,  # Number of Steps per Trajectory: the trajectory length include also initial state
        dt=config['dt'],
        computation_library=TensorFlowLibrary,
        predictor_specification="ODE_TF_default"
    )

    control = generate_random_inputs(config, total_number_of_trajectories, trajectory_length)
    initial_states_tf = tf.convert_to_tensor(states.reshape([total_number_of_trajectories, -1]), dtype=tf.float32)
    control_tf = tf.convert_to_tensor(control.reshape(total_number_of_trajectories, trajectory_length, 2), dtype=tf.float32)[:, :-1, :] # We don't need last control inputs, they will not be applied
    
    print('Generating data...')
    predictions = np.array(predictor.predict_tf(initial_states_tf, control_tf))  # dim = [total_trajectories, trajectory_length, 9] 
    data, df = prediction_to_df(predictions, control, config, total_number_of_trajectories, trajectory_length)

    df = clean_dataframe(df)
    print('Number of nan element in the generated Data:', np.count_nonzero(np.isnan(data)))

    save_dataframe_to_csv(df, config, run_for_ML_Pipeline, record_path, total_number_of_trajectories)
    print('Finished data generation')


if __name__ == '__main__':
    run_data_generator()


