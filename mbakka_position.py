import numpy as np
import pandas as pd
import os
"""
file_path = 'ExperimentRecordings/F1TENTH_Blank-MPPI-0__2022-11-23_19-22-06.csv'
num_rollouts = 10

file_path = "../f1tenth_development_gym/ExperimentRecordings/F1TENTH_Blank-MPPI-0__2022-11-23_19-22-06.csv"

def load_pose_x_y(file_path,num_rollouts):
    df = pd.read_csv(file_path, sep=',',skiprows=8,usecols = ['pose_x','pose_y'])
    position = df.to_numpy()
    if position.shape[0]>=num_rollouts:
        position = position[np.random.randint(position.shape[0], size=num_rollouts), :]
    else:
        raise Exception('The file is not long Enough to get position. Should be at least %s rows long',num_rollouts)

    return position

result = load_pose_x_y(file_path,num_rollouts)
print(result)
"""
N= 5000
dt=0.03
x_dist = np.random.uniform(-0, 0, N)
y_dist = np.random.uniform(1, 1, N)

# Steering of front wheels
delta_dist = np.random.uniform(-0, 0, N)

# velocity in face direction
v_dist = np.random.uniform(0, 0, N)

# Yaw Angle
yaw_dist = np.random.uniform(-0, 0, N)

# Yaw Angle cos and sin
yaw_cos = np.cos(yaw_dist)
yaw_sin = np.sin(yaw_dist)

# Yaw rate
yaw_rate_dist = np.random.uniform(-0, 0, N)

# Slip angle
slip_angle_dist = np.random.uniform(-0, 0, N)

u0_dist = np.random.uniform(-0, 0, N)
u1_dist = np.random.uniform(-0, 0, N)

time_axis = np.arange(0,N*dt,dt).reshape(-1,1)

states = np.column_stack((time_axis,u0_dist,u1_dist,yaw_rate_dist,v_dist,yaw_dist, yaw_cos, yaw_sin, x_dist, y_dist, slip_angle_dist, delta_dist))

np.savetxt('SI_Toolkit_ASF/Experiments/Tutorial/Recordings/Train/Data_gen_zero.csv', states, delimiter=",")

row_names = 'time,translational_control,angular_control,angular_vel_z,linear_vel_x,pose_theta,pose_theta_cos,pose_theta_sin,pose_x,pose_y,slip_angle,steering_angle'

with open('SI_Toolkit_ASF/Experiments/Tutorial/Recordings/Train/Data_gen_zero.csv','r+') as file:
    file_data = file.read()
    file.seek(0,0)
    file.write(row_names + '\n' + file_data)
