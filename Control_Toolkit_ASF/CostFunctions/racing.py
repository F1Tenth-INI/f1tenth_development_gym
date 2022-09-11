import yaml

import tensorflow as tf

from utilities.state_utilities import *

from Control_Toolkit_ASF.CostFunctions.f1t_cost_function import f1t_cost_function


class racing(f1t_cost_function):
    def __init__(self, env):
        super().__init__(env)

    def get_terminal_cost(self, s_hor):
        terminal_state = s_hor[:, -1, :]
        terminal_speed_cost = self.get_terminal_speed_cost(terminal_state)
        terminal_cost = terminal_speed_cost

        return terminal_cost

    def get_stage_cost(self, s, u, u_prev):

        # target_position = target[0]
        # lidar_scans = target[1:217]
        # waypoints = target[218:]

        trajectories = s[:, :, POSE_X_IDX:POSE_Y_IDX + 1]  # TODO: Maybe better access separatelly X&Y and concat them afterwards.

        # It is not used while writing...
        cc = self.get_actuation_cost(u)
        ccrc = self.get_control_change_rate_cost(u, u_prev)

        crash_cost = self.get_crash_cost(trajectories, self.LIDAR)
        acceleration_cost = self.get_acceleration_cost(u)
        steering_cost = self.get_steering_cost(u)

        if self.waypoints.shape[0]:
            distance_to_waypoints_cost = self.get_distance_to_waypoints_cost(trajectories, self.waypoints)
        else:
            distance_to_waypoints_cost = tf.zeros_like(steering_cost)

        stage_cost = cc + ccrc + distance_to_waypoints_cost + crash_cost + steering_cost + acceleration_cost

        # Read out values for cost weight callibration: Uncomment for debugging

        # distance_to_waypoints_cost_numpy = distance_to_waypoints_cost.numpy()[:20]
        # acceleration_cost_numpy = acceleration_cost.numpy()[:20]
        # steering_cost_numpy= steering_cost.numpy()[:20]
        # crash_penelty_numpy= crash_penelty.numpy()[:20]
        # cc_numpy= cc.numpy()[:20]
        # ccrc_numpy= ccrc.numpy()[:20]
        # stage_cost_numpy= stage_cost.numpy()[:20]

        return stage_cost

    def get_trajectory_cost(self, s_hor, u, u_prev=None):
        return (
                self.env_mock.lib.sum(self.get_stage_cost(s_hor[:, :-1, :], u, u_prev), 1)
                + self.get_terminal_cost(s_hor)
        )