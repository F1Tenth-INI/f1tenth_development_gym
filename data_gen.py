import sys
sys.path.append('/Users/mehdi/Downloads/f1tenth_development_gym/SI_Toolkit/src')

from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit.computation_library import TensorFlowLibrary
import tensorflow as tf
import csv
import numpy as np
import pandas as pd

from math import pi
from tqdm import trange

predictor = PredictorWrapper()
num_rollouts = 1 # 200
mpc_horizon = 500 # 20
number_of_trajectories = 1 # 20
dt = 0.03
predictor.configure(
    batch_size=num_rollouts, # Number of initial states
    horizon=mpc_horizon, # Number of Steps per Trajectory
    dt = dt,
    computation_library=TensorFlowLibrary,
    predictor_specification= "ODE_TF_default"
)

def load_pose_x_y(file_path,num_rollouts):
    df = pd.read_csv(file_path, sep=',',skiprows=8,usecols = ['pose_x','pose_y'])
    position = df.to_numpy()
    if position.shape[0]>=num_rollouts:
        position = position[np.random.randint(position.shape[0], size=num_rollouts), :]
    else:
        raise Exception('The file is not long Enough to get position. Should be at least %s rows long',num_rollouts)

    return position
# Position => invariant since we are only interested in delta

file_path = "../f1tenth_development_gym/ExperimentRecordings/F1TENTH_Blank-MPPI-0__2022-11-29_10-19-19.csv"
pose_x_y = load_pose_x_y(file_path,num_rollouts)
x_dist = pose_x_y[:,0]
y_dist = pose_x_y[:,1]
"""
x_dist = np.random.uniform(-0, 0, num_rollouts)
y_dist = np.random.uniform(-0, 0, num_rollouts)
"""

# Steering of front wheels
delta_dist = np.random.uniform(-1, 1, num_rollouts)

# velocity in face direction
v_dist = np.random.uniform(10, 30, num_rollouts)

# Yaw Angle
yaw_dist = np.random.uniform(-pi, pi, num_rollouts)

# Yaw Angle cos and sin
yaw_cos = np.cos(yaw_dist)
yaw_sin = np.sin(yaw_dist)

# Yaw rate
yaw_rate_dist = np.random.uniform(-0, 0, num_rollouts)

# Slip angle
slip_angle_dist = np.random.uniform(-0, 0, num_rollouts)

# Collect states in a table
"""
    'angular_vel_z',  # x5: yaw rate
    'linear_vel_x',   # x3: velocity in x direction
    'pose_theta',  # x4: yaw angle
    'pose_theta_cos',
    'pose_theta_sin',
    'pose_x',  # x0: x position in global coordinates
    'pose_y',  # x1: y position in global coordinates
    'slip_angle',  # x6: slip angle at vehicle center
    'steering_angle' 
"""
# Order to follow for the network
#states = np.column_stack((x_dist,y_dist,yaw_dist,v_dist,yaw_rate_dist,yaw_cos, yaw_sin, slip_angle_dist, delta_dist))
# Order the predictor needs
states = np.column_stack((yaw_rate_dist,v_dist,yaw_dist, yaw_cos, yaw_sin, x_dist, y_dist, slip_angle_dist, delta_dist))

def order_data_for_nn(a):
    nn_required_order = np.column_stack((a[:,0],a[:,1],a[:,7], a[:,8], a[:,4], a[:,3], a[:,2], a[:,5], a[:,6], a[:,9], a[:,10]))
    return nn_required_order

# Initialize the data array with a zeros raw (just for convenience, will be removed later)
data = np.array([0,0,1,2,3,4,5,6,7,8,9])
"""
#---------------Constant input control for each trajectory-------------------------

for i in trange(number_of_trajectories):

    mu, sigma = 0, 0.4  # mean and standard deviation
    u0_dist = np.random.normal(mu, sigma, num_rollouts)
    mu, sigma = 0, 0.5  # mean and standard deviation
    u1_dist = np.random.normal(mu, sigma, num_rollouts)
    # Each row of controls is a control input to be followed along a trajectory for the corresponding initial state
    controls = np.column_stack((u0_dist, u1_dist)) # dim = [initial_states, 2]

    # In the following 2 line, we duplicate each control input (in rows direction, axis=0)
    control = np.repeat(controls,mpc_horizon,axis=0)
    control = control.reshape(num_rollouts,mpc_horizon,2) # dim = [initial_states,mpc_horizon,2]

    s = tf.convert_to_tensor(states.reshape([num_rollouts, -1]), dtype=tf.float32)
    Q = tf.convert_to_tensor(control.reshape(num_rollouts, mpc_horizon, 2), dtype=tf.float32)
    predictions = np.array(predictor.predict_tf(s, Q)) # dim = [initial_states, mpc_horizon+1, 9]

    # control_augmented is same as control but we repeat the last raw of the 2 dimension
    # only done for coding convenience, to be able to add the control next to the last state in the data file
    control_augmented = np.repeat(control, [1] * (control.shape[1] - 1) + [2], axis=1)
    control_augmented = control_augmented.reshape(num_rollouts*(mpc_horizon+1),2)

    # Convert prediction to one big nd array and add control input next to each state transition
    predictions =predictions.reshape(num_rollouts*(mpc_horizon+1),9)
    control_with_predictions = np.column_stack((control_augmented,predictions))

    # Stack control_with_predictions into the data file
    data = np.row_stack((data,control_with_predictions))
"""

# ----------------Random input control for each trajectory-------------------------

for i in trange(number_of_trajectories):

    mu, sigma = 0, 0.4  # mean and standard deviation
    u0_dist = np.random.normal(mu, sigma, num_rollouts*mpc_horizon)
    mu, sigma = 0, 0.5  # mean and standard deviation
    u1_dist = np.random.normal(mu, sigma, num_rollouts*mpc_horizon)
    # Each row of controls is a control input to be followed along a trajectory for the corresponding initial state
    controls = np.column_stack((u0_dist, u1_dist)) # dim = [initial_states, 2]

    # In the following 2 line, we duplicate each control input (in rows direction, axis=0)
    control = controls.reshape(num_rollouts,mpc_horizon,2) # dim = [initial_states,mpc_horizon,2]

    s = tf.convert_to_tensor(states.reshape([num_rollouts, -1]), dtype=tf.float32)
    Q = tf.convert_to_tensor(control.reshape(num_rollouts, mpc_horizon, 2), dtype=tf.float32)
    predictions = np.array(predictor.predict(s, Q)) # dim = [initial_states, mpc_horizon+1, 9]
    # !!!!!!!! Check if the prediction are ordered well in raw, ie each row is the prediction of the previous

    # control_augmented is same as control but we repeat the last raw of the 2 dimension
    # only done for coding convenience, to be able to add the control next to the last state in the data file
    control_augmented = np.repeat(control, [1] * (control.shape[1] - 1) + [2], axis=1)
    control_augmented = control_augmented.reshape(num_rollouts*(mpc_horizon+1),2)

    # Convert prediction to one big nd array and add control input next to each state transition
    predictions = predictions.reshape(num_rollouts*(mpc_horizon+1),9)
    control_with_predictions = np.column_stack((control_augmented,predictions))

    # Stack control_with_predictions into the data file
    data = np.row_stack((data,control_with_predictions))



# Eliminate the first all zeros raw
data = data[1:,:]

#Reorder data columns
#data = order_data_for_nn(data)

time_axis = np.arange(0,data.shape[0]*dt,dt).reshape(-1,1)
data = np.column_stack((time_axis,data))
# Compute the number of nan in the data
print('Number of nan element in the generated Data:', np.count_nonzero(np.isnan(data)))

# Save data into a csv file
np.savetxt('SI_Toolkit_ASF/Experiments/Tutorial/Recordings/Train/Data_gen_1_exp.csv', data, delimiter=",")

# Add state names as a first raw
# order the network needs
#row_names = 'time,translational_control,angular_control,pose_x,pose_y,pose_theta,linear_vel_x,angular_vel_z,pose_theta_cos,pose_theta_sin,slip_angle,steering_angle'
# Order the predictor needs
row_names = 'time,translational_control,angular_control,angular_vel_z,linear_vel_x,pose_theta,pose_theta_cos,pose_theta_sin,pose_x,pose_y,slip_angle,steering_angle'

with open('SI_Toolkit_ASF/Experiments/Tutorial/Recordings/Train/Data_gen_1_exp.csv','r+') as file:
    file_data = file.read()
    file.seek(0,0)
    file.write(row_names + '\n' + file_data)

print('Yes')


