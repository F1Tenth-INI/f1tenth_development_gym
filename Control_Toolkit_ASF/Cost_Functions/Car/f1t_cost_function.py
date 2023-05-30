import yaml
import os
import tensorflow as tf
from types import SimpleNamespace

from Control_Toolkit.Cost_Functions import cost_function_base
from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.computation_library import ComputationLibrary

from utilities.state_utilities import *
from utilities.path_helper_ros import *

distance_normalization = 6.0

# TODO: Config should be loaded at specific functions
# load constants from config file config_controllers
gym_path = get_gym_path()
config = yaml.load(open(os.path.join(gym_path , "Control_Toolkit_ASF", "config_cost_function.yml"), "r"), Loader=yaml.FullLoader)
config_controllers = yaml.load(open(os.path.join(gym_path, "Control_Toolkit_ASF", "config_controllers.yml"), "r"), Loader=yaml.FullLoader)

mpc_type = config_controllers["mpc"]['optimizer']

R = config["Car"]["racing"]["R"]

cc_weight = tf.convert_to_tensor(config["Car"]["racing"]["cc_weight"])
ccrc_weight = config["Car"]["racing"]["ccrc_weight"]
ccocrc_weight = config["Car"]["racing"]["ccocrc_weight"]
icdc_weight = config["Car"]["racing"]["icdc_weight"]

distance_to_waypoints_cost_weight = config["Car"]["racing"]["distance_to_waypoints_cost_weight"]
velocity_diff_to_waypoints_cost_weight = config["Car"]["racing"]["velocity_diff_to_waypoints_cost_weight"]
speed_control_diff_to_waypoints_cost_weight = config["Car"]["racing"]["speed_control_diff_to_waypoints_cost_weight"]
steering_cost_weight = config["Car"]["racing"]["steering_cost_weight"]
angular_velocity_cost_weight = config["Car"]["racing"]["angular_velocity_cost_weight"]
angle_difference_to_wp_cost_weight = config["Car"]["racing"]["angle_difference_to_wp_cost_weight"]
slipping_cost_weight = config["Car"]["racing"]["slipping_cost_weight"]
terminal_speed_cost_weight = config["Car"]["racing"]["terminal_speed_cost_weight"]
target_distance_cost_weight = config["Car"]["racing"]["target_distance_cost_weight"]

acceleration_cost_weight = config["Car"]["racing"]["acceleration_cost_weight"]
max_acceleration = config["Car"]["racing"]["max_acceleration"]
desired_max_speed = config["Car"]["racing"]["desired_max_speed"]
waypoint_velocity_factor = config["Car"]["racing"]["waypoint_velocity_factor"]

crash_cost_slope = config["Car"]["racing"]["crash_cost_slope"]
crash_cost_safe_margin = config["Car"]["racing"]["crash_cost_safe_margin"]
crash_cost_max_cost = config["Car"]["racing"]["crash_cost_max_cost"]

from SI_Toolkit.Functions.General.hyperbolic_functions import return_hyperbolic_function

class f1t_cost_function(cost_function_base):
    def __init__(self, variable_parameters: SimpleNamespace, ComputationLib: "type[ComputationLibrary]") -> None:
        super(f1t_cost_function, self).__init__(variable_parameters, ComputationLib)
        self._P1 = None
        self._P2 = None

        self.hyperbolic_function_for_crash_cost, _, _ = return_hyperbolic_function((0.0, crash_cost_max_cost), (crash_cost_safe_margin, 0.0), slope=crash_cost_slope, mode=1)



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

    # TODO: Make it library agnostic. This also justifies why some methods are not static, although currently they could be
    def get_actuation_cost(self, u):
        cc_cost = R * (u ** 2)
        return tf.math.reduce_sum(cc_weight * cc_cost, axis=-1)

    def get_terminal_speed_cost(self, terminal_state):
        ''' Compute penality for deviation from desired max speed'''
        terminal_speed = terminal_state[:, LINEAR_VEL_X_IDX]

        speed_diff = tf.abs(terminal_speed - desired_max_speed)
        terminal_speed_cost = terminal_speed_cost_weight * speed_diff

        return terminal_speed_cost

    # cost of changeing control to fast
    def get_control_change_rate_cost(self, u, u_prev):
        """
        Compute penalty of instant control change, i.e. differences to previous control input

        We use absolute difference instead of relative one as we care for absolute change of control input ~ reaction time required by the car
        We use L2 norm as we want to penalize big changes and are fine with small ones - we want a gentle control in general
        The weight of this cost should be in general small, it should help to eliminate sudden changes of big magnitude which the physical car might not handle correctly
        """
        u_prev_vec = tf.concat((tf.ones((u.shape[0], 1, u.shape[-1])) * u_prev, u[:, :-1, :]), axis=1)
        ccrc = ((u - u_prev_vec)/control_limits_max_abs) ** 2

        return tf.math.reduce_sum(ccrc_weight * ccrc, axis=-1)

    def get_control_change_of_change_rate_cost(self, u, u_prev):
        """
        Removing jerk, keeping change constant and penalizing |∆u(t)-∆u(t-1)|
        We use L1 norm: We are fine with few big changes - e.g. when car accelerates.
        We use absolute difference: the control input should stay smooth same for big and low control inputs
        """
        u_prev_vec = tf.concat((tf.ones((u.shape[0], 1, u.shape[-1])) * u_prev, u[:, :-1, :]), axis=1)
        u_next_vec = tf.concat((u[:, 1:, :], u[:, -1:, :]), axis=1)
        ccocrc = self.lib.abs((u_next_vec + u_prev_vec - 2*u)/control_limits_max_abs)

        return tf.math.reduce_sum(ccocrc_weight * ccocrc, axis=-1)


    def get_immediate_control_discontinuity_cost(self, u, u_prev):

        u_prev_vec = tf.ones((u.shape[0], u.shape[1], u.shape[-1])) * u_prev  # The vector is just to keep dimensions right and scaling with gr
        icdc = (u_prev_vec-u[:, :1, :])**2

        return tf.math.reduce_sum(icdc_weight * icdc, axis=-1)



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
    
    def get_angular_velocity_cost(self, s):
        angular_velovities = s[:, :, ANGULAR_VEL_Z_IDX]
        angula_velocity_cost = angular_velocity_cost_weight * tf.square(angular_velovities)
        return angula_velocity_cost

    def get_slipping_cost(self, s):
        slipping_angles = s[:, :, SLIP_ANGLE_IDX]
        slipping_cost = slipping_cost_weight * tf.square(slipping_angles)
        return slipping_cost
    
    def get_distance_to_waypoints_per_rollout(self, trajectory_points, waypoints):
        trajectories_shape = tf.shape(trajectory_points)
        number_of_rollouts = trajectories_shape[0]
        number_of_steps = trajectories_shape[1]
        number_of_waypoints = tf.shape(waypoints)[0]

        positions_of_trajectories = tf.reshape(trajectory_points, [trajectories_shape[0] * trajectories_shape[1], 2])
        distance_to_waypoints = self.distances_from_list_to_list_of_points(positions_of_trajectories, waypoints)
        distance_to_waypoints_per_rollout = tf.reshape(distance_to_waypoints,
                                                       (number_of_rollouts, number_of_steps, number_of_waypoints))
        return distance_to_waypoints_per_rollout

    def get_distances_from_trajectory_points_to_closest_target_point(self, trajectory_points, waypoints):
        distance_to_waypoints_per_rollout = self.get_distance_to_waypoints_per_rollout(trajectory_points, waypoints)
        min_distance_to_waypoints_per_rollout = tf.reduce_min(distance_to_waypoints_per_rollout, axis=2)
        return min_distance_to_waypoints_per_rollout

    def get_nearest_waypoints_indices(self, trajectory_points, waypoints):
        distance_to_waypoints_per_rollout = self.get_distance_to_waypoints_per_rollout(trajectory_points, waypoints)
        indices_nearest_waypoints = tf.argmin(distance_to_waypoints_per_rollout, axis=2)
        return indices_nearest_waypoints

    def distances_to_list_of_points(self, point, points2):
        length = tf.shape(points2)[0]
        points1 = tf.tile([point], [length, 1])

        diff = points2 - points1
        squared_diff = tf.math.square(diff)
        squared_dist = tf.reduce_sum(squared_diff, axis=1)
        return squared_dist


    def get_crash_cost(self, trajectories, border_points):

        trajectories_shape = tf.shape(trajectories)

        points_of_trajectories = tf.reshape(trajectories, [trajectories_shape[0] * trajectories_shape[1], 2])
        squared_distances = self.distances_from_list_to_list_of_points(points_of_trajectories, border_points)

        minima = tf.math.reduce_min(squared_distances, axis=1)

        minima = tf.reshape(minima, [trajectories_shape[0], trajectories_shape[1]])

        minima = tf.clip_by_value(minima, 0.0, crash_cost_safe_margin)

        cost_for_passing_close = self.hyperbolic_function_for_crash_cost(minima)

        return cost_for_passing_close

    def get_target_distance_cost_normed(self, trajectories, target_points):

        min_distance_to_trajectory_points = \
            self.get_distances_from_trajectory_points_to_closest_target_point(trajectories, target_points)

        min_distance_from_trajectory = tf.math.reduce_min(min_distance_to_trajectory_points, axis=1)

        min_distance_from_trajectory_normed = min_distance_from_trajectory / tf.math.reduce_max(
            min_distance_from_trajectory)

        return min_distance_from_trajectory_normed

    def get_target_distance_cost(self, trajectories, target_points):
        return target_distance_cost_weight * self.get_target_distance_cost_normed(trajectories, target_points)


    def distances_from_list_to_list_of_points(self, points1, points2):

        # TODO: Cast both as float
        points1 = tf.cast(points1, tf.float32)
        points2 = tf.cast(points2, tf.float32)
        
        
        length1 = tf.shape(points1)[0]
        length2 = tf.shape(points2)[0]

        points1 = tf.tile([points1], [1, 1, length2])
        points1 = tf.reshape(points1, (length1 * length2, 2))

        points2 = tf.tile([points2], [length1, 1, 1])
        points2 = tf.reshape(points2, (length1 * length2, 2))

        diff = points2 - points1
        squared_diff = tf.math.square(diff)
        squared_dist = tf.reduce_sum(squared_diff, axis=1)

        squared_dist = tf.reshape(squared_dist, [length1, length2])

        return squared_dist

    def get_distance_to_wp_segments_cost(self, s, waypoints, nearest_waypoint_indices):
        car_positions = s[:, :, POSE_X_IDX:POSE_Y_IDX + 1]  # TODO: Maybe better access separatelly X&Y and concat them afterwards.
        waypoint_positions = waypoints[:,1:3]

        return distance_to_waypoints_cost_weight * self.get_squared_distances_to_nearest_wp_segment(car_positions, waypoint_positions, nearest_waypoint_indices)

    def get_wp_segment_vectors(self, waypoint_positions, nearest_waypoint_indices):
        nearest_waypoints = tf.gather(waypoint_positions, nearest_waypoint_indices)
        nearest_waypoints_next = tf.gather(waypoint_positions, nearest_waypoint_indices + 1)

        # Get next waypoint segment vector for every position on the car's rollout
        wp_segment_vectors = nearest_waypoints_next - nearest_waypoints
        return wp_segment_vectors

    def get_squared_distances_to_nearest_wp_segment(self, car_positions, waypoint_positions, nearest_waypoint_indices):
        nearest_waypoints = tf.gather(waypoint_positions, nearest_waypoint_indices)
        nearest_waypoints_next = tf.gather(waypoint_positions, nearest_waypoint_indices + 1)

        # Get next waypoint segment vector for every position on the car's rollout
        wp_segment_vectors = nearest_waypoints_next - nearest_waypoints
        wp_segment_vector_norms = tf.norm(wp_segment_vectors, axis=2)
        
        # Get the vector from the next waypoint to the car position
        vector_car_pos_to_nearest_wp = tf.subtract(car_positions,nearest_waypoints)
        vector_car_pos_to_nearest_wp_norm = tf.norm(vector_car_pos_to_nearest_wp, axis=2)
        vector_car_pos_to_nearest_wp_norm_square = tf.multiply(vector_car_pos_to_nearest_wp_norm, vector_car_pos_to_nearest_wp_norm)
        
        # Get projection of car_pos_to_nearest_wp on nearest wp_segment
        car_pos_to_nearest_wp_dot_wp_segment =  tf.reduce_sum(vector_car_pos_to_nearest_wp * wp_segment_vectors, axis=2)
        projection_car_pos_to_wp_on_wp_segment = car_pos_to_nearest_wp_dot_wp_segment / wp_segment_vector_norms
        projection_square = tf.multiply(projection_car_pos_to_wp_on_wp_segment, projection_car_pos_to_wp_on_wp_segment)
        
        # Pytagoras
        distance_to_segment_square = vector_car_pos_to_nearest_wp_norm_square - projection_square
        distance_to_wp_segments_cost = distance_to_segment_square
        return distance_to_wp_segments_cost
    
    def get_velocity_difference_to_wp_cost(self, s, waypoints, nearest_waypoint_indices):
        
        # if (velocity_diff_to_waypoints_cost_weight == 0): # Don't calculate if cost is 0
        #     return tf.zeros_like(s[:,:,0])

        car_vel_x = s[:, :, LINEAR_VEL_X_IDX]
        velocity_difference_to_wp = self.get_velocity_difference_to_wp(car_vel_x, waypoints, nearest_waypoint_indices)
        horizon = tf.cast(tf.shape(s)[1], dtype=tf.float32)
        velocity_difference_to_wp_normed = velocity_difference_to_wp # / 1 horizon
        return velocity_diff_to_waypoints_cost_weight * velocity_difference_to_wp_normed
    
    def get_distance_to_wp_cost(self, s, waypoints, nearest_waypoint_indices):
        car_positions = s[:, :, POSE_X_IDX:POSE_Y_IDX + 1]  # TODO: Maybe better access separatelly X&Y and concat them afterwards.
        nearest_waypoints = tf.gather(waypoints, nearest_waypoint_indices)
        waypoint_positions = nearest_waypoints[:,:, 1:3]

        wp_car_vector = car_positions - waypoint_positions
        wp_car_vector_norms = tf.norm(wp_car_vector, axis=2)
        return 1.0 * wp_car_vector_norms
        
    
    def get_speed_control_difference_to_wp_cost(self, u, s, waypoints, nearest_waypoint_indices):
        
        # Get nearest and the nearest_next waypoint for every position on the car's rollout
        speed_control = u[:, :, TRANSLATIONAL_CONTROL_IDX]
        velocity_difference_to_wp = self.get_velocity_difference_to_wp(speed_control, waypoints, nearest_waypoint_indices)
        horizon = tf.cast(tf.shape(s)[1], dtype=tf.float32)
        velocity_difference_to_wp_normed = velocity_difference_to_wp / horizon
        return speed_control_diff_to_waypoints_cost_weight * velocity_difference_to_wp_normed
        
    def get_velocity_difference_to_wp(self, car_vel, waypoints, nearest_waypoint_indices):

        nearest_waypoints = tf.gather(waypoints, nearest_waypoint_indices)

        nearest_waypoint_vel_x = waypoint_velocity_factor * nearest_waypoints[:,:,5]

        # nearest_waypoint_vel_x = self.lib.clip(nearest_waypoint_vel_x, 0.5, 17.5)
        
        vel_difference = tf.abs(nearest_waypoint_vel_x - car_vel)
        return vel_difference
    
    def get_angle_difference_to_wp_cost(self, s, waypoints, nearest_waypoint_indices):

        nearest_waypoints = tf.gather(waypoints, nearest_waypoint_indices)
        nearest_waypoint_psi_rad_sin = tf.sin(nearest_waypoints[:,:,3])
        nearest_waypoint_psi_rad_cos = tf.cos(nearest_waypoints[:,:,3])
        
        car_angle_sin = s[:, :, POSE_THETA_SIN_IDX]
        car_angle_cos = s[:, :, POSE_THETA_COS_IDX]
        
        angle_difference_sin = tf.square(nearest_waypoint_psi_rad_sin - car_angle_sin)
        angle_difference_cos= tf.square(nearest_waypoint_psi_rad_cos - car_angle_cos)
        return angle_difference_to_wp_cost_weight * (angle_difference_sin + angle_difference_cos)
        
    def normed_discount(self, array_to_discount, model_array, discount_factor):
        discount_vector = self.lib.ones_like(model_array) * discount_factor
        discount_vector = self.lib.cumprod(discount_vector, 0)
        norm_factor = self.lib.to_tensor(self.lib.sum(self.lib.ones_like(discount_vector), 0), self.lib.float32)/self.lib.sum(discount_vector, 0)
        discount_vector = norm_factor*discount_vector
        return array_to_discount*discount_vector

