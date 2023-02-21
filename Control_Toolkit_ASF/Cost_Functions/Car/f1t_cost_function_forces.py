import yaml
import os
import tensorflow as tf
import casadi

from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.computation_library import ComputationLibrary

from utilities.state_utilities import *

distance_normalization = 6.0

# TODO: Config should be loaded at specific functions
# load constants from config file config_controllers
config = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"), Loader=yaml.FullLoader)
config_controllers = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"), "r"),
                               Loader=yaml.FullLoader)

mpc_type = config_controllers["mpc"]['optimizer']

R = config["Car"]["racing"]["R"]

cc_weight = tf.convert_to_tensor(config["Car"]["racing"]["cc_weight"])
ccrc_weight = config["Car"]["racing"]["ccrc_weight"]
distance_to_waypoints_cost_weight = config["Car"]["racing"]["distance_to_waypoints_cost_weight"]
velocity_diff_to_waypoints_cost_weight = config["Car"]["racing"]["velocity_diff_to_waypoints_cost_weight"]
steering_cost_weight = config["Car"]["racing"]["steering_cost_weight"]
terminal_speed_cost_weight = config["Car"]["racing"]["terminal_speed_cost_weight"]
target_distance_cost_weight = config["Car"]["racing"]["target_distance_cost_weight"]

acceleration_cost_weight = config["Car"]["racing"][mpc_type]["acceleration_cost_weight"]
max_acceleration = config["Car"]["racing"][mpc_type]["max_acceleration"]
desired_max_speed = config["Car"]["racing"][mpc_type]["desired_max_speed"]
waypoint_velocity_factor = config["Car"]["racing"][mpc_type]["waypoint_velocity_factor"]


class f1t_cost_function_forces(cost_function_base):
    def __init__(self, controller: template_controller, ComputationLib: "type[ComputationLibrary]") -> None:
        super(f1t_cost_function_forces, self).__init__(controller, ComputationLib)
        self._P1 = None
        self._P2 = None

    # region updating P1 & P2
    @property
    def P1(self):
        return self._P1

    @property
    def P1_tf(self):
        return self._P1_tf

    @P1.setter
    def P1(self, P1):
        self._P1 = P1
        if not hasattr(self, "_P1_tf"):
            self._P1_tf = tf.Variable(P1, dtype=tf.float32)
        else:
            self._P1_tf.assign(P1)

    @property
    def P2(self):
        return self._P2

    @property
    def P2_tf(self):
        return self._P2_tf

    @P2.setter
    def P2(self, P2):
        self._P2 = P2
        if not hasattr(self, "_P2_tf"):
            self._P2_tf = tf.Variable(P2, dtype=tf.float32)
        else:
            self._P2_tf.assign(P2)

    # endregion

    # region Cost components
    # TODO: Make it library agnostic. This also justifies why some methods are not static, although currently they could be
    def get_actuation_cost(self, u):
        cc_cost = R * (u ** 2)
        return casadi.cumsum(cc_weight.numpy() * cc_cost, -1)

    def get_terminal_speed_cost(self, terminal_state):
        ''' Compute penality for deviation from desired max speed'''
        terminal_speed = terminal_state[:, LINEAR_VEL_X_IDX]

        speed_diff = tf.abs(terminal_speed - desired_max_speed)
        terminal_speed_cost = terminal_speed_cost_weight * speed_diff

        return terminal_speed_cost

    # cost of changeing control to fast
    def get_control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input"""
        if ccrc_weight != 0.0:
            u_prev_vec = np.concat((np.ones((u.shape[0], 1, u.shape[-1])) * u_prev, u[:, :-1, :]), axis=1)
            ccrc = (u - u_prev_vec) ** 2
            return tf.math.reduce_sum(ccrc_weight * ccrc, axis=-1)
            raise Exception('ccrc not implemented with nlp_forces')
        else:
            return casadi.SX([[0.0], [0.0]])

    def get_acceleration_cost(self, u):
        ''' Calculate cost for deviation from desired acceleration at every timestep'''
        accelerations = u[:, :, TRANSLATIONAL_CONTROL_IDX]
        acceleration_cost = max_acceleration - accelerations
        acceleration_cost = tf.abs(acceleration_cost)
        acceleration_cost = acceleration_cost_weight * acceleration_cost

        return acceleration_cost

    def get_steering_cost(self, u):
        ''' Calculate cost for steering at every timestep'''
        steering = u[:, :, ANGULAR_CONTROL_IDX]
        steering = tf.abs(steering)
        steering_cost = steering_cost_weight * steering

        return steering_cost

    def get_distance_to_waypoints_per_rollout(self, trajectory_points, waypoints):
        trajectories_shape = trajectory_points.shape
        number_of_rollouts = trajectories_shape[0]
        number_of_waypoints = waypoints.shape[0]

        positions_of_trajectories = trajectory_points.reshape((trajectories_shape[0], 2))
        distance_to_waypoints = self.distances_from_list_to_list_of_points(positions_of_trajectories, waypoints)
        distance_to_waypoints_per_rollout = casadi.reshape(distance_to_waypoints,
                                                           (number_of_rollouts, number_of_waypoints))
        return distance_to_waypoints_per_rollout

    def get_distances_from_trajectory_points_to_closest_target_point(self, trajectory_points, waypoints):
        distance_to_waypoints_per_rollout = self.get_distance_to_waypoints_per_rollout(trajectory_points, waypoints)
        min_distance_to_waypoints_per_rollout = tf.reduce_min(distance_to_waypoints_per_rollout, axis=2)
        return min_distance_to_waypoints_per_rollout

    def get_nearest_waypoints_mask(self, trajectory_points, waypoints):
        distance_to_waypoints_per_rollout = self.get_distance_to_waypoints_per_rollout(trajectory_points, waypoints)
        mask_nearest_waypoints = distance_to_waypoints_per_rollout == casadi.mmin(distance_to_waypoints_per_rollout)
        return mask_nearest_waypoints

    def distances_to_list_of_points(self, point, points2):
        length = tf.shape(points2)[0]
        points1 = tf.tile([point], [length, 1])

        diff = points2 - points1
        squared_diff = tf.math.square(diff)
        squared_dist = tf.reduce_sum(squared_diff, axis=1)
        return squared_dist

    def get_crash_cost_normed(self, trajectories, border_points):

        trajectories_shape = tf.shape(trajectories)

        points_of_trajectories = tf.reshape(trajectories, [trajectories_shape[0] * trajectories_shape[1], 2])
        squared_distances = self.distances_from_list_to_list_of_points(points_of_trajectories, border_points)

        minima = tf.math.reduce_min(squared_distances, axis=1)

        distance_threshold = tf.constant([0.36])  # 0.6 ^2
        indices_too_close = tf.math.less(minima, distance_threshold)
        crash_cost_normed = tf.cast(indices_too_close, tf.float32)

        crash_cost_normed = tf.reshape(crash_cost_normed, [trajectories_shape[0], trajectories_shape[1]])

        return crash_cost_normed

    def get_crash_cost(self, trajectories, border_points):
        return self.get_crash_cost_normed(trajectories,
                                          border_points) * 1000000  # Disqualify trajectories too close to sensor points

    def get_target_distance_cost_normed(self, trajectories, target_points):

        min_distance_to_trajectory_points = \
            self.get_distances_from_trajectory_points_to_closest_target_point(trajectories, target_points)

        min_distance_from_trajectory = tf.math.reduce_min(min_distance_to_trajectory_points, axis=1)

        min_distance_from_trajectory_normed = min_distance_from_trajectory / tf.math.reduce_max(
            min_distance_from_trajectory)

        return min_distance_from_trajectory_normed

    def get_target_distance_cost(self, trajectories, target_points):
        return target_distance_cost_weight * self.get_target_distance_cost_normed(trajectories, target_points)

    def get_distance_to_waypoints_cost(self, s, waypoint_positions):
        car_positions = s[:, :,
                        POSE_X_IDX:POSE_Y_IDX + 1]  # TODO: Maybe better access separatelly X&Y and concat them afterwards.
        return self.get_distances_from_trajectory_points_to_closest_target_point(car_positions,
                                                                                 waypoint_positions) * distance_to_waypoints_cost_weight

    def distances_from_list_to_list_of_points(self, points1, points2):

        length1 = points1.shape[0]
        length2 = points2.shape[0]

        points1 = casadi.repmat(points1, (1, length2))
        points1 = casadi.reshape(points1, (2, length1 * length2)).T

        points2 = casadi.repmat(points2, (length1, 1))
        points2 = casadi.reshape(points2, (length1 * length2, 2))  # doesn't change it

        diff = points2 - points1
        squared_diff = casadi.power(diff, 2)
        squared_dist = casadi.cumsum(squared_diff, 1)[:, 1]

        # squared_dist = casadi.reshape(squared_dist, [length1, length2])

        return squared_dist

    def get_distance_to_wp_segments_cost(self, s, waypoints):
        car_positions = s[:,
                        0:2]  # TODO: Maybe better access separatelly X&Y and concat them afterwards.
        waypoint_positions = waypoints[:, 1:3]

        # Get nearest and the nearest_next waypoint for every position on the car's rollout
        nearest_waypoint_mask = self.get_nearest_waypoints_mask(car_positions, waypoint_positions[:-1, :])
        return self.get_squared_distances_to_nearest_wp_segment(car_positions, waypoint_positions,
                                                                nearest_waypoint_mask)

    def get_wp_segment_vectors(self, waypoint_positions, nearest_waypoint_indices):
        nearest_waypoints = tf.gather(waypoint_positions, nearest_waypoint_indices)
        nearest_waypoints_next = tf.gather(waypoint_positions, nearest_waypoint_indices + 1)

        # Get next waypoint segment vector for every position on the car's rollout
        wp_segment_vectors = nearest_waypoints_next - nearest_waypoints
        return wp_segment_vectors

    def get_squared_distances_to_nearest_wp_segment(self, car_positions, waypoint_positions, nearest_waypoint_mask):
        # nearest_waypoints = tf.gather(waypoint_positions, nearest_waypoint_mask)
        # nearest_waypoints_next = tf.gather(waypoint_positions, nearest_waypoint_mask + 1)
        nearest_waypoints = casadi.hcat([nearest_waypoint_mask, casadi.SX.zeros(1, 1)]) @ waypoint_positions
        nearest_waypoints_next =  casadi.hcat([casadi.SX.zeros(1, 1), nearest_waypoint_mask]) @ waypoint_positions

        # Get next waypoint segment vector for every position on the car's rollout
        wp_segment_vectors = nearest_waypoints_next - nearest_waypoints
        wp_segment_vector_norms = casadi.norm_2(wp_segment_vectors)

        # Get the vector from the next waypoint to the car position
        vector_car_pos_to_nearest_wp = car_positions - nearest_waypoints
        vector_car_pos_to_nearest_wp_norm_square = vector_car_pos_to_nearest_wp @ vector_car_pos_to_nearest_wp.T

        # Get projection of car_pos_to_nearest_wp on nearest wp_segment
        car_pos_to_nearest_wp_dot_wp_segment = vector_car_pos_to_nearest_wp @ wp_segment_vectors.T
        projection_car_pos_to_wp_on_wp_segment = car_pos_to_nearest_wp_dot_wp_segment / wp_segment_vector_norms
        projection_square = projection_car_pos_to_wp_on_wp_segment @ projection_car_pos_to_wp_on_wp_segment.T

        # Pytagoras
        distance_to_segment_square = vector_car_pos_to_nearest_wp_norm_square - projection_square
        distance_to_wp_segments_cost = distance_to_segment_square
        return distance_to_wp_segments_cost

    def get_velocity_difference_to_wp_cost(self, s, waypoints):
        # Get nearest and the nearest_next waypoint for every position on the car's rollout
        car_positions = s[:,
                        0:2]  # TODO: Maybe better access separatelly X&Y and concat them afterwards.
        waypoint_positions = waypoints[:, 1:3]
        nearest_waypoint_mask = self.get_nearest_waypoints_mask(car_positions, waypoint_positions[:-1, :])
        car_vel_x = s[:, 3]
        velocity_difference_to_wp = self.get_velocity_difference_to_wp(car_vel_x, waypoints, nearest_waypoint_mask)
        return velocity_diff_to_waypoints_cost_weight * velocity_difference_to_wp

    def get_velocity_difference_to_wp(self, car_vel, waypoints, nearest_waypoint_mask):

        nearest_waypoints = casadi.hcat([nearest_waypoint_mask, casadi.SX.zeros(1, 1)]) @ waypoints

        nearest_waypoint_vel_x = waypoint_velocity_factor * nearest_waypoints[ :, 5]

        vel_difference = casadi.norm_1(nearest_waypoint_vel_x - car_vel)
        return vel_difference
