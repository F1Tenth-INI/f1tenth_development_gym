import tensorflow as tf

from utilities.state_utilities import *

from Control_Toolkit_ASF.Cost_Functions.Car.f1t_cost_function import f1t_cost_function

distance_to_middle_line_weight = 10.0
velocity_diff_to_waypoints_cost_weight = 1.0
v_perpendicular_weight = 0.0

class racing_v2(f1t_cost_function):

    def get_terminal_cost(self, terminal_state):
        terminal_cost = 0.0*terminal_state[:, 0]

        return terminal_cost

    def get_stage_cost(self, s, u, u_prev):

        waypoints = self.controller.next_waypoints
        waypoint_positions = waypoints[:, 1:3]

        car_positions = s[:, :, POSE_X_IDX:POSE_Y_IDX + 1]

        nearest_waypoint_indices = self.get_nearest_waypoints_indices(car_positions, waypoint_positions[:-1])

        car_vel_x = s[:, :, LINEAR_VEL_X_IDX]

        angle = s[:, :, POSE_THETA_IDX] + s[:, :, SLIP_ANGLE_IDX]
        v_vector = self.lib.stack([car_vel_x*self.lib.cos(angle), car_vel_x*self.lib.sin(angle)], -1)
        v_vector_norm = tf.norm(v_vector, axis=2)
        v_vector_norm_squared = tf.multiply(v_vector_norm, v_vector_norm)

        wp_segment_vectors = self.get_wp_segment_vectors(waypoint_positions, nearest_waypoint_indices)
        wp_segment_vector_norms = tf.norm(wp_segment_vectors, axis=2)

        # Get projection of car velocity on nearest wp_segment
        v_parallel =  tf.reduce_sum(v_vector * wp_segment_vectors, axis=2)/wp_segment_vector_norms
        v_parallel_norm_squared = tf.multiply(v_parallel, v_parallel)
        v_perpendicular_squared = v_vector_norm_squared - v_parallel_norm_squared
        v_perpendicular_squared_normalized = v_perpendicular_squared/(v_vector_norm_squared+0.01)

        velocity_difference_to_wp = self.get_velocity_difference_to_wp(v_parallel, waypoints, nearest_waypoint_indices)

        distance_to_wp_segments_cost = distance_to_middle_line_weight * self.get_squared_distances_to_nearest_wp_segment(car_positions, waypoint_positions, nearest_waypoint_indices)

        velocity_difference_to_wp_cost = velocity_diff_to_waypoints_cost_weight * velocity_difference_to_wp

        v_perpendicular_cost = v_perpendicular_weight * v_perpendicular_squared_normalized
        # v_perpendicular_cost = 0.0

        cost_for_stopping = tf.cast(tf.less(v_parallel, 0.5), dtype=tf.float32) * 1000.0


        stage_cost = (
                distance_to_wp_segments_cost
                + velocity_difference_to_wp_cost
                + v_perpendicular_cost
                + cost_for_stopping
            )

        # AAA = distance_to_wp_segments_cost.numpy()

        return stage_cost
