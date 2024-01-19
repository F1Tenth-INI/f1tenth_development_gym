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


def get_state(value, limit, length, distribution='uniform'):
    if value is not None:
        return np.repeat(value, length)
    else:
        if distribution == 'uniform':
            return np.random.uniform(limit[0], limit[1], length)
        elif distribution == 'normal':
            return np.random.normal(limit[0], limit[1], length)


def get_initial_states(number_of_initial_states, initial_states,
                       file_with_initial_states=None, initial_states_from_file=None):
    init_limits = initial_states.pop('init_limits')

    if file_with_initial_states is not None:
        initial_states_from_file = []
        pose_x_y, number_of_initial_states = load_initial_states_from_file(file_with_initial_states, initial_states_from_file, number_of_initial_states)
        x_dist = pose_x_y[:, 0]
        y_dist = pose_x_y[:, 1]
    else:
        x_dist = get_state(initial_states['x'], init_limits['x'], number_of_initial_states)
        y_dist = get_state(initial_states['y'], init_limits['y'], number_of_initial_states)

    steering_dist = get_state(initial_states['steering_angle'], init_limits['steering_angle'], number_of_initial_states)
    v_dist = get_state(initial_states['velocity_x'], init_limits['velocity_x'], number_of_initial_states)
    yaw_dist = get_state(initial_states['yaw_angle'], init_limits['yaw_angle'], number_of_initial_states)
    yaw_cos = np.cos(yaw_dist)
    yaw_sin = np.sin(yaw_dist)
    yaw_rate_dist = get_state(initial_states['yaw_rate'], init_limits['yaw_rate'], number_of_initial_states)
    slip_angle_dist = get_state(initial_states['slip_angle'], init_limits['slip_angle'], number_of_initial_states)

    # Order to follow for the network
    #states = np.column_stack((x_dist,y_dist,yaw_dist,v_dist,yaw_rate_dist,yaw_cos, yaw_sin, slip_angle_dist, steering_dist))
    # Order the predictor needs
    states = np.column_stack((yaw_rate_dist, v_dist, yaw_dist, yaw_cos, yaw_sin, x_dist, y_dist, slip_angle_dist, steering_dist))

    return states, number_of_initial_states

def order_data_for_nn(a):
    nn_required_order = np.column_stack((a[:,0],a[:,1],a[:,7], a[:,8], a[:,4], a[:,3], a[:,2], a[:,5], a[:,6], a[:,9], a[:,10]))
    return nn_required_order