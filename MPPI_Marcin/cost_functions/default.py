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

#cost for distance from track edge
def distance_difference_cost(position, target):
    """Compute penalty for distance of cart to the target position"""
    return tf.math.reduce_sum(((position - target[0, :]) / distance_normalization) ** 2, axis=-1)

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
    @param s: (2000, 11, 7) The parallel state evolutions of the car
    """
    dd = dd_weight * distance_difference_cost(
        s[:, -1, :2], target
    )
    
    initial_state = s[0, 0, :]
    initial_position = initial_state[:2]

    
    i_shoud_be_this_fast = tf.fill([2000], 7.0)
    
    # go_fast_cost = terminal_distance_cost(
    #     s[:, -1, :2], initial_position
    # )
    
    i_am_this_fast = s[:, -1, 3]
    
    
    go_fast_cost = tf.math.abs(tf.add(i_shoud_be_this_fast, - i_am_this_fast))
    terminal_cost = go_fast_cost
    
    # t = terminal_cost.numpy()
    # print("t", t[:5])
    return terminal_cost

#cost of changeing control to fast
def control_change_rate_cost(u, u_prev):
    """Compute penalty of control jerk, i.e. difference to previous control input"""
    u_prev_vec = tf.concat((tf.ones((u.shape[0], 1, u.shape[-1]))*u_prev, u[:, :-1, :]), axis=1)
    return (u - u_prev_vec) ** 2


def low_speed_cost(s):
    speed = tf.sqrt(tf.reduce_sum((s[:, 1:, 1:]-s[:, :-1, 1:])**2, axis=-1))  # 100.0 is for default dt
    speed = tf.concat((tf.zeros((speed.shape[0], 1), dtype=tf.float32), speed), axis=1)
    return -speed

#all stage costs together
def q(s,u,target, u_prev):
    crash_penelty = get_crash_penelty(s[:, :, :2], target)
    cc = tf.math.reduce_sum(cc_weight * CC_cost(u), axis=-1)
    ccrc = tf.math.reduce_sum(ccrc_weight * control_change_rate_cost(u,u_prev), axis=-1)
    
    accelerations = u[:,:,0]
    max_accelerations = tf.fill([2000, 10], 9.2)
    acceleration_costs = max_accelerations - accelerations
    acceleration_costs = 0.01 * acceleration_costs
    
    steering = u[:,:,1]
    steering = tf.abs(steering)
    do_not_steer_cost = 5 * steering
    
    # steering_numpy = steering.numpy()[:20]
    # accelerations_numpy = accelerations.numpy()[:20]
    # max_accelerations_numpy = max_accelerations.numpy()[:20]
    # acceleration_costs_numpy = acceleration_costs.numpy()[:20]
    # do_not_steer_cost_numpy= do_not_steer_cost.numpy()[:20]
    # crash_penelty_numpy= crash_penelty.numpy()[:20]
    # cc_numpy= cc.numpy()[:20]
    # ccrc_numpy= ccrc.numpy()[:20]
    
    # acceleration_sums = tf.reduce_sum(accelerations, axis=1)
    # print(acceleration_costs.numpy())
    stage_cost = cc+ccrc+crash_penelty +acceleration_costs
    return stage_cost + do_not_steer_cost 


# @tf.function
def get_crash_penelty(trajectories, target):
    trajectories_shape = tf.shape(trajectories)
    points_of_trajectories = tf.reshape(trajectories, [trajectories_shape[0] * trajectories_shape[1],2])
    squared_distances = distances_from_list_to_list_of_points(points_of_trajectories,target)
    summed_distances = tf.reduce_sum(squared_distances, axis = 1)
    summed_distances = tf.reshape(summed_distances, (2000, 10))
    # summed_distances = tf.reduce_sum(summed_distances, axis = 1)
    
    
    # max_value = tf.reduce_max(summed_distances, axis = 1 )
        
    distance_cost = tf.divide(1, summed_distances)
    

    
    distance_cost = tf.multiply(100000.00, distance_cost)
    

    
    minima = tf.math.reduce_min(squared_distances, axis=1)
    
    
    distance_threshold = tf.constant([0.2])

    indices_too_close = tf.math.less(minima, distance_threshold)
    crash_cost = tf.cast(indices_too_close, tf.float32) * 10000
    
    crash_cost = tf.reshape(crash_cost, [trajectories_shape[0],trajectories_shape[1]])
    # print(minima.numpy())
    # print(crash_cost.numpy())
    return crash_cost # +distance_cost



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
    