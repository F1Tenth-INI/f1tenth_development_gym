import tensorflow as tf

from utilities.state_utilities import *

from Control_Toolkit_ASF.CostFunctions.Car.f1t_cost_function import f1t_cost_function


class racing(f1t_cost_function):

    def get_terminal_cost(self, terminal_state):
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

        # crash_cost = tf.stop_gradient(self.get_crash_cost(trajectories, self.LIDAR))
        acceleration_cost = self.get_acceleration_cost(u)
        # steering_cost = self.get_steering_cost(u)

        if self.waypoints.shape[0]:
            distance_to_waypoints_cost = self.get_distance_to_waypoints_cost(trajectories, self.waypoints)
            # distance_to_waypoints_cost = self.get_distance_to_nearest_segment_cost(trajectories, self.waypoints)
        else:
            distance_to_waypoints_cost = tf.zeros_like(acceleration_cost)

        stage_cost = (
                cc
                + ccrc
                + distance_to_waypoints_cost
                # + steering_cost
                + acceleration_cost
                # + distance_to_border_cost#+ crash_cost
            )

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
        """ This is the top level cost function that is called by all the MPC optimizers.
        """
        # self.update_waypoints(s_hor)
        return (
                self.lib.sum(self.get_stage_cost(s_hor[:, :-1, :], u, u_prev), 1)
                + self.get_terminal_cost(s_hor)
        )

    def update_waypoints(self, s_hor):
        self.P1, self.P2 = self.get_P1_and_P2(s_hor[:, :, POSE_X_IDX:POSE_Y_IDX + 1], self.waypoints)
        # Get the list of nearest waypoints -1 till 15, checke that variable is assigned
        # Get the arrrays  P = P2-P1 and P1, these should be assigned
