import csv
import os
import glob
import numpy as np

from SI_Toolkit.load_and_normalize import load_data, append_derivatives

base_path = 'SI_Toolkit_ASF/Experiments/'
# folder = '0_DNN_tiv/Recordings'
folder = 'NNOde/Recordings'
path = f'./{base_path}/{folder}/**/*.csv'
csv_files = glob.glob(path)
file_paths = [os.path.relpath(file, f'{base_path}/{folder}') for file in csv_files]
dt = 0.04


save_to = f'{base_path}/{folder}_preprocessed'
try:
    os.makedirs(save_to)
except FileExistsError:
    pass

dfs = load_data(csv_files)

# variables_for_derivative = ['pose_x',
#                             'pose_y',
#                             'pose_theta',
#                             'pose_theta_sin',
#                             'pose_theta_cos',
#                             'linear_vel_x',
#                             'angular_vel_z',
#                             'slip_angle',
#                             'steering_angle']
# derivative_algorithm = "backward_difference"

# dfs, paths_with_derivative = append_derivatives(dfs, variables_for_derivative, derivative_algorithm, csv_files)

datapoints_removed = 0

for i, df in enumerate(dfs):
    # Fix delta values where pose_theta is wrapped
    df.D_pose_theta = df.D_pose_theta * dt
    df.D_pose_theta = np.where(df.D_pose_theta > 3.0, df.D_pose_theta - 2 * np.pi, df.D_pose_theta)  
    df.D_pose_theta = np.where(df.D_pose_theta < -3.0, 2 * np.pi + df.D_pose_theta, df.D_pose_theta)
    df.D_pose_theta = df.D_pose_theta / dt

    df = df.assign(D_car_pose_x=df.pose_theta_cos * df.D_pose_x + df.pose_theta_sin * df.D_pose_y)
    dfs[i] = df.assign(D_car_pose_y=-df.pose_theta_sin * df.D_pose_x + df.pose_theta_cos * df.D_pose_y)

for i, df in enumerate(dfs):

    start_size = len(df)
    threshold_violated = ((abs(df.slip_angle) >= np.pi / 4) | (abs(df.D_angular_vel_z) > 30) | (abs(df.D_car_pose_y) > 4))

    # Set experiment index
    diff = threshold_violated.diff()
    diff[0] = False
    df['experiment_index'] = np.floor(diff.cumsum() / 2)
    df = df[~threshold_violated]

    #Remove experiments with only little slipping
    # no_slip = df[df.slip_angle.between(-0.005, 0.005)].groupby('experiment_index').size()
    # no_slip_percentage = (no_slip / df.groupby('experiment_index').size().reindex(no_slip.index) * 100).sort_values(ascending=False)
    # df = df[~df.experiment_index.isin(no_slip_percentage.iloc[0:30].index)]

    # Remove datapoints
    df = df.groupby('experiment_index').filter(lambda x: len(x) >= 30)

    #Remove experiments with no change in velocity
    # no_vel_change = df[df.D_linear_vel_x.between(-0.005, 0.005)].groupby('experiment_index').size()
    # no_vel_change_percentage = (no_vel_change / df.groupby('experiment_index').size().reindex(no_vel_change.index) * 100).sort_values(ascending=False)
    # df = df[~df.experiment_index.isin(no_vel_change_percentage.iloc[0:30].index)]

    dfs[i] = df
    datapoints_removed += start_size - len(dfs[i])

print(f'Removed {datapoints_removed} datapoints')

for i in range(len(dfs)):
    old_file_path = csv_files[i]
    file_path = f'{save_to}/{file_paths[i]}'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(old_file_path, "r", newline='') as f_input, \
         open(file_path, "w", newline='') as f_output:
        for line in f_input:
            if line[0:len('#')] == '#':
                csv.writer(f_output).writerow([line.strip()])
            else:
                break

    dfs[i].to_csv(file_path, index=False, mode='a')
