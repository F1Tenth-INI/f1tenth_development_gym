import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import sys
import os
sys.path.append(os.getcwd())

from utilities.map_utilities import MapUtilities


def load_data(recording):
    df = pd.read_csv(recording, skiprows=8, usecols=['pose_x', 'pose_y', 'time'])
    df = df.set_index('time')
    return df


def load_waypoints(map_name):
    df_wp = pd.read_csv(f'./utilities/maps/{map_name}/{map_name}_wp.csv', skipinitialspace=True, usecols=['# s_m', 'x_m', 'y_m'])
    df_wp = df_wp.set_index('# s_m')
    return df_wp


def append_laps(df):
    starting_x = df.iloc[0]['pose_x']
    relation_to_starting_x = (df['pose_x'] >= starting_x)
    laps = (relation_to_starting_x.diff().cumsum() / 2)
    laps.iloc[0] = 0.0
    laps = laps.astype(int).to_frame('lap')
    df = pd.concat([df, laps], axis=1)
    return df


def get_lap_times(df):
    lap_change = df['lap'].diff() != 0
    lap_times = df[lap_change]['lap'].reset_index()
    lap_times['time'] = lap_times['time'].diff()
    lap_times = lap_times.drop(0).set_index('lap')
    return lap_times


def get_deviation_from_optimal_line(df, df_wp):
    map_utilities = MapUtilities()
    nearest_wp = df[['pose_x', 'pose_y']].apply(lambda x: map_utilities.get_closest_point(x.to_numpy(), df_wp.to_numpy())[1], axis=1, result_type='expand')
    nearest_wp.columns = (['nearest_wp_x', 'nearest_wp_y'])
    df = pd.concat([df, nearest_wp], axis=1)
    mse = df.groupby(['lap']).apply(lambda df: mean_squared_error(df[['nearest_wp_x', 'nearest_wp_y']], df[['pose_x', 'pose_y']]))
    mse = mse.to_frame('mse')
    return mse


def get_metrics_from_recording(recording, map_name):
    df = load_data(recording)
    df = append_laps(df)
    lap_times = get_lap_times(df)
    df_wp = load_waypoints(map_name)
    deviation_from_optimal_line = get_deviation_from_optimal_line(df, df_wp)

    return (lap_times, deviation_from_optimal_line)


def save_lap_times(lap_times, path):
    lap_times.to_csv(f'{path}/lap_times.csv')


def save_deviation(deviation, path):
    deviation.to_csv(f'{path}/optimal_line_deviation.csv')


def save_metrics(metrics, path):
    save_lap_times(metrics[0], path)
    save_deviation(metrics[1], path)


def plot_lap_times(lap_times, path):
    lap_times.plot()
    if path is not None:
        plt.savefig(f'{path}/lap_times.png')


def plot_deviation(deviation, path):
    deviation.plot()
    if path is not None:
        plt.savefig(f'{path}/optimal_line_deviation.png')


def plot_metrics(metrics, path):
    plot_lap_times(metrics[0], path)
    plot_deviation(metrics[1], path)


if __name__ == '__main__':
    folder = '0_DNN'
    # model = 'Custom-11IN-ODE_module.py-STModel_high_mu-9OUT-0'
    # recording_folder = f'./SI_Toolkit_ASF/Experiments/{folder}/Models/{model}/test_data'
    recording_folder = f'./SI_Toolkit_ASF/Experiments/{folder}/Sim_results/'
    recording = f'{recording_folder}/baseline_high_mu.csv'
    metrics = get_metrics_from_recording(recording=recording, map_name='hangar12')
    save_metrics(metrics, recording_folder)
    plot_metrics(metrics, recording_folder)
