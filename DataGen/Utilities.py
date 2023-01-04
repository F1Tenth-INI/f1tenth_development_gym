import numpy as np
import pandas as pd


def load_initial_states_from_file(file_path, initial_states_from_file, number_of_initial_states):
    df = pd.read_csv(file_path, sep=',', skiprows=8, usecols=initial_states_from_file)
    if number_of_initial_states == 'max':
        number_of_initial_states = df.shape[0]
        if number_of_initial_states == 0:
            raise Exception('Check your datafile, it is too short, seems (almost) empty.')
        else:
            print("Will load {} initial states".format(number_of_initial_states))
    elif df.shape[0] >= number_of_initial_states:
        df = df[np.random.randint(df.shape[0], size=number_of_initial_states), :]
    else:
        raise Exception('The file is not long Enough to get position. Should be at least %s rows long', number_of_initial_states)
    position = df.to_numpy()
    return position, number_of_initial_states


def get_initial_states(number_of_initial_states, file_with_initial_states=None, initial_states_from_file=None):

    if file_with_initial_states is not None:
        initial_states_from_file = []
        pose_x_y, number_of_initial_states = load_initial_states_from_file(file_with_initial_states,initial_states_from_file, number_of_initial_states)
        x_dist = pose_x_y[:, 0]
        y_dist = pose_x_y[:, 1]
    else:
        x_dist = np.random.uniform(-50, 50, number_of_initial_states)
        y_dist = np.random.uniform(-50, 50, number_of_initial_states)

    # Steering of front wheels
    delta_dist = np.random.uniform(-0.5, 0.5, number_of_initial_states)


    # velocity in face direction
    # v_dist = np.random.uniform(5.0, 20, number_of_initial_states)
    v_dist = np.random.uniform(-2.0, 20.0, number_of_initial_states)
    # v_dist = np.random.uniform(10, 20, number_of_initial_states)

    # Yaw Angle
    yaw_dist = np.random.uniform(-np.pi, np.pi, number_of_initial_states)

    # Yaw Angle cos and sin
    yaw_cos = np.cos(yaw_dist)
    yaw_sin = np.sin(yaw_dist)

    # Yaw rate
    yaw_rate_dist = np.random.normal(0.0, 3.25, number_of_initial_states)
    # yaw_rate_dist = np.random.normal(0.0, 0.0, number_of_initial_states)

    # Slip angle
    slip_angle_dist = np.random.uniform(-2.0, 2.0, number_of_initial_states)
    # slip_angle_dist = np.random.uniform(-0.0, 0.0, number_of_initial_states)

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

    return states, number_of_initial_states

def order_data_for_nn(a):
    nn_required_order = np.column_stack((a[:,0],a[:,1],a[:,7], a[:,8], a[:,4], a[:,3], a[:,2], a[:,5], a[:,6], a[:,9], a[:,10]))
    return nn_required_order