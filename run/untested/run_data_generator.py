import os

import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import trange

import yaml

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.load_and_normalize import append_derivatives_to_df

from DataGen.Utilities import get_initial_states
from utilities.csv_logger import create_csv_header, create_csv_file


try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass


def generate_random_inputs(config, total_number_of_trajectories, trajectory_length, initial_states=None):
    controls = []
    for control_input in ['steering', 'velocity']:
        distribution = config['control_inputs'][control_input]['distribution']
        if distribution == 'normal':
            distribution_func = np.random.normal
        elif distribution == 'uniform':
            distribution_func = np.random.uniform

        strategy = config['control_inputs'][control_input]['strategy']
        control_range = config['control_inputs'][control_input]['control_range']

        if config['control_inputs'][control_input]['generate_around_state']:
            state_index = {'steering': -1, 'velocity': 1}

            if strategy == 'constant':
                u = distribution_func(initial_states[:, state_index[control_input]], control_range[1])
                u = np.repeat(controls, trajectory_length, axis=1)
            else:
                raise NotImplementedError(f'The strategy {strategy} is not yet implemented for generation around the initial state')

        else:
            if strategy == 'constant':
                u = distribution_func(control_range[0], control_range[1], total_number_of_trajectories)
                u = np.repeat(u, trajectory_length, axis=1)
            elif strategy == 'random':
                u = distribution_func(control_range[0], control_range[1], (total_number_of_trajectories, trajectory_length))
            elif strategy == 'mixed':
                hold_constant_input = config['control_inputs'][control_input]['hold_constant_input']
                number_of_inputs_per_trajectory = int(config['trajectory_length'] / hold_constant_input) + 1
                u = distribution_func(control_range[0], control_range[1], (total_number_of_trajectories, number_of_inputs_per_trajectory))
                u = np.repeat(u, config['control_inputs'][control_input]['hold_constant_input'], axis=1)
                u = u[:, 0:trajectory_length]
            else:
                raise Exception(f'{strategy} is not a valid strategy for generating control inputs')

        controls.append(u)

    control = np.stack(controls, axis=2)
    return control


def save_dataframe_to_csv(df, config, run_for_ML_Pipeline, record_path, total_number_of_trajectories, single_file=False):
    index = 0
    if run_for_ML_Pipeline:
        path = record_path + '/Train/'
    elif record_path is None:
        path = config["path_save_to"]
    else:
        path = record_path

    dt = config['dt']

    print('Saving data...')
    experiment_indices = df.experiment_index.unique()
    if single_file:
        if run_for_ML_Pipeline:
            split = config['split']
            indices_split = np.split(experiment_indices, [int(split[0] * len(experiment_indices)), int((split[0] + split[1]) * len(experiment_indices))])
            for i, dataset_type in enumerate(['Train', 'Test', 'Validate']):
                path = f'{record_path}/{dataset_type}/'
                csv_path = f'{path}/Trajectories.csv'
                header = create_csv_header(Settings, controller_name='data_generator', dt=dt)
                csv_path = create_csv_file(path, csv_name=csv_path, header=header)
                df[df.experiment_index.isin(indices_split[i])].to_csv(csv_path, index=False, mode='a')
        else:
            csv_path = f'{record_path}/Trajectories.csv'
            header = create_csv_header(Settings, controller_name='data_generator', dt=dt)
            csv_path = create_csv_file(record_path, csv_name=csv_path, header=header)
            df[df.experiment_index.isin(indices_split[i])].to_csv(csv_path, index=False, mode='a')
    else:
        for experiment in trange(len(experiment_indices)):
            if run_for_ML_Pipeline:
                split = config['split']
                if experiment == int(split[0] * len(experiment_indices)):
                    path = record_path + "/Validate/"
                    index = 0
                elif experiment == int((split[0] + split[1]) * len(experiment_indices)):
                    path = record_path + "/Test/"
                    index = 0

                try:
                    os.makedirs(path)
                except:
                    pass

            csv_path = f'{path}Trajectory-{str(index)}.csv'
            if not df[df.experiment_index == experiment_indices[experiment]].empty:
                header = create_csv_header(Settings, controller_name='Random input', dt=dt)
                csv_path = create_csv_file(path, csv_name=csv_path, header=header)
                df[df.experiment_index == experiment_indices[experiment]].to_csv(csv_path, index=False, mode='a')
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
    return df


def remove_outliers(df):
    problematic_experiments = df[df.slip_angle.abs() > 0.5].experiment_index.unique()
    print(f'Found {len(problematic_experiments)} experiments with too high slip angle!')
    df = df[~(df.experiment_index.isin(problematic_experiments))].reset_index(drop=True)

    return df


def remove_delta_outliers(df):
    problematic_experiments = df[(df.D_angular_vel_z.abs() > 50) | (df.D_slip_angle.abs() > 3.0)].experiment_index.unique()
    print(f'Found {len(problematic_experiments)} experiments with too high delta values!')
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

def add_state_noise(df):
    noise_levels = {
       'angular_vel_z': 0.01,
       'linear_vel_x': 0.01,
       'pose_theta': 0.01,
       'pose_theta_cos': 0.01,
       'pose_theta_sin': 0.01,
       'pose_x': 0.1,
       'pose_y': 0.1,
       'slip_angle': 0.01,
       'steering_angle': 0.01
    }
    for column in ['angular_vel_z', 'linear_vel_x', 'pose_theta', 'pose_theta_cos', 'pose_theta_sin', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle']:
        df[column] += np.random.normal(scale=noise_levels[column])

    return df


def append_derivatives(df):
    print('Appending derivatives...')
    variables_for_derivative = ['pose_x',
                                'pose_y',
                                'pose_theta',
                                'pose_theta_sin',
                                'pose_theta_cos',
                                'linear_vel_x',
                                'angular_vel_z',
                                'slip_angle',
                                'steering_angle']
    derivative_algorithm = "backward_difference"
    # dfs = []
    # experiment_indices = df.experiment_index.unique()
    # for i in trange(len(experiment_indices)):
    #     dfs.append(append_derivatives_to_df(df[df.experiment_index == experiment_indices[i]], variables_for_derivative, derivative_algorithm))
    # df = pd.concat(dfs)
    df = append_derivatives_to_df(df, variables_for_derivative, derivative_algorithm)
    # df = df[df.time != 0.00]  # First delta value not defined, but also not relevant
    # df[df.time = 0.00]
    df.loc[df.time == 0.00, df.columns.str.startswith('D_')] = 0.00

    return df


def run_data_generator(run_for_ML_Pipeline=False, record_path=None):

    # TODO: Interpolator

    config = yaml.load(open('utilities/DataGen/config_data_gen.yml', "r"), Loader=yaml.FullLoader)
    number_of_initial_states = config['number_of_initial_states']
    trajectory_length = config['trajectory_length']
    number_of_trajectories_per_initial_state = config['number_of_trajectories_per_initial_state']
    total_number_of_trajectories = number_of_initial_states * number_of_trajectories_per_initial_state

    initial_states_from_file_features, file_with_initial_states = get_initial_state_configuration(config)
    random_initial_states = config['initial_states']['random']

    np.random.seed(config['seed'])

    states, number_of_initial_states = get_initial_states(number_of_initial_states, random_initial_states,
                                                          file_with_initial_states, initial_states_from_file_features)
    states = np.repeat(states, repeats=number_of_trajectories_per_initial_state, axis=0)

    controls = generate_random_inputs(config, total_number_of_trajectories, trajectory_length, initial_states=states)

    print('Generating data...')

    initial_states_tf = tf.convert_to_tensor(states.reshape([total_number_of_trajectories, -1]), dtype=tf.float32)
    control_tf = tf.convert_to_tensor(controls.reshape(total_number_of_trajectories, trajectory_length, 2), dtype=tf.float32)[:, :-1, :]  # We don't need last control inputs, they will not be applied
    predictor = PredictorWrapper()
    predictor.configure(
        batch_size=total_number_of_trajectories,
        horizon=trajectory_length - 1,  # Number of Steps per Trajectory: the trajectory length include also initial state
        dt=config['dt'],
        computation_library=TensorFlowLibrary,
        predictor_specification="ODE_TF_default"
    )
    predictions = np.array(predictor.predict_tf(initial_states_tf, control_tf))  # dim = [total_trajectories, trajectory_length, 9]
    df = prediction_to_df(predictions, controls, config, total_number_of_trajectories, trajectory_length)

    # df = remove_outliers(df)
    df = append_derivatives(df)
    # df = remove_delta_outliers(df)
    # df = add_state_noise(df)

    save_dataframe_to_csv(df, config, run_for_ML_Pipeline, record_path, total_number_of_trajectories)
    print('Finished data generation')


if __name__ == '__main__':
    run_data_generator()
