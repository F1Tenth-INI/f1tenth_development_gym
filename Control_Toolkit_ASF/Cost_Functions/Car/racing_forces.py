import tensorflow as tf

from utilities.state_utilities import *

from Control_Toolkit_ASF.Cost_Functions.Car.f1t_cost_function_forces import f1t_cost_function_forces
import casadi

class racing_forces(f1t_cost_function_forces):

    def get_terminal_cost(self, terminal_state):
        terminal_speed_cost = self.get_terminal_speed_cost(terminal_state)
        terminal_cost = terminal_speed_cost

        return terminal_cost

    def get_stage_cost(self, s, u, p):
        u_prev = p[:2]
        # next_waypoints = p[2:].reshape((20, 7))
        next_waypoints = p[2:].reshape((7, 20)).T
        # It is not used while writing...
        # cc = self.get_actuation_cost(u)
        # ccrc = self.get_control_change_rate_cost(u, u_prev)

        ## Crash cost: comment out for faster calculation...
        # car_positions = s[:, :, POSE_X_IDX:POSE_Y_IDX + 1]
        # crash_cost = tf.stop_gradient(self.get_crash_cost(car_positions, self.controller.lidar_points))
        
        # Cost related to control
        # acceleration_cost = self.get_acceleration_cost(u)
        # steering_cost = self.get_steering_cost(u)

        # Costs related to waypoints 
        if next_waypoints.shape[0]:
            distance_to_wp_segments_cost = self.get_distance_to_wp_segments_cost(s.T, next_waypoints)
            velocity_difference_to_wp_cost = self.get_velocity_difference_to_wp_cost(s.T, next_waypoints)
        # else:
        #     distance_to_wp_segments_cost = tf.zeros_like(acceleration_cost)
        #     velocity_difference_to_wp_cost = tf.zeros_like(acceleration_cost)

        ## Old waypoint cost function: only distance to nearest waypoint
        # if self.controller.next_waypoints.shape[0]:
        #     distance_to_waypoints_cost = self.get_distance_to_waypoints_cost(s, self.controller.next_waypoints)
        # else:
        #     distance_to_waypoints_cost = tf.zeros_like(acceleration_cost)


        stage_cost = (
                # cc
                # + ccrc
                distance_to_wp_segments_cost
                # + steering_cost
                # + acceleration_cost
                + velocity_difference_to_wp_cost
                # + crash_cost
                # + distance_to_waypoints_cost
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