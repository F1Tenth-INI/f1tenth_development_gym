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

#actuation cost
def CC_cost(u):
    return R * (u ** 2)

#final stage cost
def phi(s, target):
    """Calculate terminal cost of a set of trajectories

    Williams et al use an indicator function type of terminal cost in
    "Information theoretic MPC for model-based reinforcement learning"

    Here it checks the distance to the target at the endpoint
    """
    dd = dd_weight * distance_difference_cost(
        s[:, -1, :2], target
    )
    terminal_cost = dd
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
    # crash_penelty = check_collision(s[:, :, :2], target)
    cc = tf.math.reduce_sum(cc_weight * CC_cost(u), axis=-1)
    ccrc = tf.math.reduce_sum(ccrc_weight * control_change_rate_cost(u,u_prev), axis=-1)
    stage_cost = cc+ccrc  #+crash_penelty
    return stage_cost


def check_collision(position, target):
    tolerance = tf.constant(0.3**2, dtype=tf.float32)

    squared_distances = tf.reduce_sum((target[1, :] - position) ** 2, axis=-1)
    crash_penelty = tf.cast(squared_distances < tolerance, tf.float32) * 1000  # Soft constraint: Do not crash into border
    for i in tf.range(1, target.shape[0]-1):
        squared_distances = tf.reduce_sum((target[i+1,:] - position) ** 2, axis=-1)
        crash_penelty += tf.cast(squared_distances < tolerance, tf.float32) * 1000  # Soft constraint: Do not crash into border
    return crash_penelty


