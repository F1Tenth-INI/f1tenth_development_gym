from numpy import dtype
import tensorflow as tf
import yaml

distance_normalization = 6.0

#load constants from config file
config = yaml.load(open("MPPI_Marcin/config.yml", "r"), Loader=yaml.FullLoader)

dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = tf.convert_to_tensor(config["controller"]["mppi"]["cc_weight"])
R = config["controller"]["mppi"]["R"]

ccrc_weight = config["controller"]["mppi"]["ccrc_weight"]

acceleration_cost_weight = config["controller"]["mppi"]["acceleration_cost_weight"]
steering_cost_weight = config["controller"]["mppi"]["steering_cost_weight"]

#cost for distance from track edge
def distance_difference_cost(position, target_position):
    """Compute penalty for distance of cart to the target position"""
    return tf.math.reduce_sum(((position - target_position) / distance_normalization) ** 2, axis=-1)

def terminal_distance_cost(positions, initial_position):
    """Compute penalty for distance of cart to the target position"""
    return tf.math.reduce_sum(((positions - initial_position) / distance_normalization) ** 2, axis=-1)


#actuation cost
def CC_cost(u):
    return R * (u ** 2)

#final stage cost
def phi(s, target):
    """Calculate terminal cost of a set of trajectories

    Williams et al use an indicator function type of terminal cost in
    "Information theoretic MPC for model-based reinforcement learning"

    Here it checks the distance to the target at the endpoint
    @param s: (batch_size, horizon, len(state)) The parallel state evolutions of the car
    """
        
    target_position = target[0, :]
    lidar_scans = target[1:217]
    waypoints = target[218:]
    
        
    dd = dd_weight * distance_difference_cost(
        s[:, -1, :2], target_position
    )
    
    # For later: Initial state to calculate differences between initial and terminal state
    initial_state = s[:, 0, :]
    terminal_state = s[:, -1, :]
    initial_position = initial_state[:, :2]
    initial_speed = initial_state[:,3]
    
    terminal_speed = terminal_state[:, 3]
    
    terminal_speed_cost = terminal_speed_cost_weight * terminal_speed
    terminal_cost = dd + terminal_speed_cost 

    return terminal_cost

#cost of changeing control to fast
def control_change_rate_cost(u, u_prev):
    """Compute penalty of control jerk, i.e. difference to previous control input"""
    u_prev_vec = tf.concat((tf.ones((u.shape[0], 1, u.shape[-1]))*u_prev, u[:, :-1, :]), axis=1)
    return (u - u_prev_vec) ** 2


def get_acceleration_cost(u):
    accelerations = u[:,:,0]
    max_acceleration = 9.2 # From car parameters
    acceleration_cost = max_acceleration - accelerations
    acceleration_cost = tf.abs(acceleration_cost)
    acceleration_cost = acceleration_cost_weight * acceleration_cost
    
    return acceleration_cost

def get_steering_cost(u):
    steering = u[:,:,1]
    steering = tf.abs(steering)
    steering_cost = steering_cost_weight * steering
    
    return steering_cost
    

#all stage costs together
def q(s,u,target, u_prev):
    
    target_position = target[0]
    lidar_scans = target[1:217]
    waypoints = target[218:]
    
    cc = tf.math.reduce_sum(cc_weight * CC_cost(u), axis=-1)
    ccrc = tf.math.reduce_sum(ccrc_weight * control_change_rate_cost(u,u_prev), axis=-1)
    
    crash_penelty = get_crash_penelty(s[:, :, :2], lidar_scans)
    acceleration_cost = get_acceleration_cost(u)
    steering_cost = get_steering_cost(u)
    
    stage_cost = cc + ccrc + crash_penelty + acceleration_cost + steering_cost 
    
    
    # Read out values for cost weight callibration: Uncomment for debugging
    
    # acceleration_cost_numpy = acceleration_cost.numpy()[:20]
    # steering_cost_numpy= steering_cost.numpy()[:20]
    # crash_penelty_numpy= crash_penelty.numpy()[:20]
    # cc_numpy= cc.numpy()[:20]
    # ccrc_numpy= ccrc.numpy()[:20]
    # stage_cost_numpy= stage_cost.numpy()[:20]
    
    return stage_cost 


@tf.function
def get_crash_penelty(trajectories, border_points):
    trajectories_shape = tf.shape(trajectories)
    number_of_rollouts = trajectories_shape[0]
    number_of_steps = trajectories_shape[1]
    
    points_of_trajectories = tf.reshape(trajectories, [trajectories_shape[0] * trajectories_shape[1],2])
    squared_distances = distances_from_list_to_list_of_points(points_of_trajectories,border_points)
    summed_squared_distances = tf.reduce_sum(squared_distances, axis = 1)
    summed_squared_distances = tf.reshape(summed_squared_distances, (number_of_rollouts, number_of_steps))
    
    minima = tf.math.reduce_min(squared_distances, axis=1)
    
    distance_threshold = tf.constant([0.36]) #0.6 ^2
    indices_too_close = tf.math.less(minima, distance_threshold)
    crash_cost = tf.cast(indices_too_close, tf.float32) * 1000000 # Disqualify trajectories too close to sensor points
    
    crash_cost = tf.reshape(crash_cost, [trajectories_shape[0],trajectories_shape[1]])

    return crash_cost



def distances_to_list_of_points(point, points2):
    length = tf.shape(points2)[0]
    points1 = tf.tile([point], [length, 1])
    
    diff = points2 - points1
    squared_diff = tf.math.square(diff)
    squared_dist = tf.reduce_sum(squared_diff, axis=1)
    return squared_dist

@tf.function
def distances_from_list_to_list_of_points(points1, points2):
    
    length1 = tf.shape(points1)[0]
    length2 = tf.shape(points2)[0]
    
    points1 = tf.tile([points1], [1, 1, length2])
    points1 = tf.reshape(points1, (length1 * length2, 2))
    
    points2 = tf.tile([points2], [length1, 1,1])
    points2 = tf.reshape(points2, (length1 * length2, 2))
    
    diff = points2 - points1
    squared_diff = tf.math.square(diff)
    squared_dist = tf.reduce_sum(squared_diff, axis=1)
    
    squared_dist = tf.reshape(squared_dist, [length1,length2])
    
    return squared_dist
    