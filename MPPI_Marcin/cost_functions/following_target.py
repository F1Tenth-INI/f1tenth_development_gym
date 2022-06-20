import yaml

import tensorflow as tf

from main.state_utilities import *

from MPPI_Marcin.cost_functions.cost_components import (
                                                        get_target_distance_cost,
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
    trajectories = s[:, :, POSE_X_IDX:POSE_Y_IDX + 1]
    target_position = target[:1]

    terminal_speed_cost = get_terminal_speed_cost(terminal_state)

    target_distance_cost = get_target_distance_cost(trajectories, target_position)

    terminal_cost = terminal_speed_cost + target_distance_cost

    return terminal_cost


# All stage costs together
def q(s, u, target, u_prev):

    acceleration_cost = get_acceleration_cost(u)
    steering_cost = get_steering_cost(u)

    stage_cost = steering_cost + acceleration_cost

    # Read out values for cost weight callibration: Uncomment for debugging

    # distance_to_waypoints_cost_numpy = distance_to_waypoints_cost.numpy()[:20]
    # acceleration_cost_numpy = acceleration_cost.numpy()[:20]
    # steering_cost_numpy= steering_cost.numpy()[:20]
    # crash_penelty_numpy= crash_penelty.numpy()[:20]
    # cc_numpy= cc.numpy()[:20]
    # ccrc_numpy= ccrc.numpy()[:20]
    # stage_cost_numpy= stage_cost.numpy()[:20]

    return stage_cost