import yaml

import tensorflow as tf

from utilities.state_utilities import *

from Control_Toolkit_ASF.Controllers.MPPI_Marcin.cost_functions.cost_components import (
                                                        get_actuation_cost,
                                                        get_control_change_rate_cost,
                                                        get_terminal_speed_cost,
                                                        get_acceleration_cost,
                                                        get_steering_cost,
                                                        get_distance_to_waypoints_cost,
                                                        get_crash_cost,
)

#final stage cost
def phi(s, target):
    """Calculate terminal cost of a set of trajectories

    Williams et al use an indicator function type of terminal cost in
    "Information theoretic MPC for model-based reinforcement learning"

    @param s: (batch_size, horizon, len(state)) The parallel state evolutions of the car
    """

    terminal_state = s[:, -1, :]

    terminal_speed_cost = get_terminal_speed_cost(terminal_state)
    terminal_cost = terminal_speed_cost

    return terminal_cost

#all stage costs together
def q(s,u,target, u_prev):

    target_position = target[0]
    lidar_scans = target[1:217]
    waypoints = target[218:]
    trajectories = s[:, :, POSE_X_IDX:POSE_Y_IDX+1]

    # It is not used while writing...
    cc = get_actuation_cost(u)
    ccrc = get_control_change_rate_cost(u, u_prev)

    crash_cost = get_crash_cost(trajectories, lidar_scans)
    acceleration_cost = get_acceleration_cost(u)
    steering_cost = get_steering_cost(u)

    if waypoints.shape[0]:
        distance_to_waypoints_cost = get_distance_to_waypoints_cost(trajectories, waypoints)
    else:
        distance_to_waypoints_cost = tf.zeros_like(steering_cost)

    stage_cost = cc + ccrc + distance_to_waypoints_cost + crash_cost + steering_cost  + acceleration_cost

    # Read out values for cost weight callibration: Uncomment for debugging

    # distance_to_waypoints_cost_numpy = distance_to_waypoints_cost.numpy()[:20]
    # acceleration_cost_numpy = acceleration_cost.numpy()[:20]
    # steering_cost_numpy= steering_cost.numpy()[:20]
    # crash_penelty_numpy= crash_penelty.numpy()[:20]
    # cc_numpy= cc.numpy()[:20]
    # ccrc_numpy= ccrc.numpy()[:20]
    # stage_cost_numpy= stage_cost.numpy()[:20]

    return stage_cost