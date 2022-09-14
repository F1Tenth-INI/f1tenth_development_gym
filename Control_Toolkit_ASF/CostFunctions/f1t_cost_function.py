
import yaml
from typing import Union
import tensorflow as tf


from Control_Toolkit.others.environment import NumpyLibrary, TensorFlowLibrary

from utilities.state_utilities import LINEAR_VEL_X_IDX, TRANSLATIONAL_CONTROL_IDX, ANGULAR_CONTROL_IDX

distance_normalization = 6.0

# load constants from config file
config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

mpc_type = config["controller"]['general']['mpc_type']

if mpc_type == 'MPPI':
    mpc_type = 'mppi-tf'
elif mpc_type == 'RPGD':
    mpc_type = 'dist-adam-resamp2'
else:
    raise NotImplementedError


R = config["controller"][mpc_type]["R"]

max_acceleration = config["controller"][mpc_type]["max_acceleration"]
desired_max_speed = config["controller"][mpc_type]["desired_max_speed"]

cc_weight = tf.convert_to_tensor(config["controller"][mpc_type]["cc_weight"])
ccrc_weight = config["controller"][mpc_type]["ccrc_weight"]
distance_to_waypoints_cost_weight = config["controller"][mpc_type]["distance_to_waypoints_cost_weight"]
acceleration_cost_weight = config["controller"][mpc_type]["acceleration_cost_weight"]
steering_cost_weight = config["controller"][mpc_type]["steering_cost_weight"]
terminal_speed_cost_weight = config["controller"][mpc_type]["terminal_speed_cost_weight"]
target_distance_cost_weight = config["controller"][mpc_type]["target_distance_cost_weight"]


class f1t_cost_function:
    def __init__(self, environment) -> None:
        self.env_mock = environment


        if self.env_mock.lib == TensorFlowLibrary:
            self.LIDAR_attribute = "LIDAR_tf"
            self.waypoints_attribute = "waypoints_tf"
            self.target_position_attribute = "target_position_tf"
        elif self.env_mock.lib == NumpyLibrary:
            self.LIDAR_attribute = "LIDAR"
            self.waypoints_attribute = "waypoints"
            self.target_position_attribute = "target_position"
        else:
            raise ValueError(
                "Currently, this cost function only supports environment written in TensorFlow or NumPy (not PyTorch etc.)"
            )

    # region Getting changing variables of the cost function

    @property
    def LIDAR(self) -> Union[float, tf.Variable]:
        return getattr(self.env_mock, self.LIDAR_attribute)

    @property
    def waypoints(self) -> Union[float, tf.Variable]:
        return getattr(self.env_mock, self.waypoints_attribute)

    @property
    def target_position(self) -> Union[float, tf.Variable]:
        return getattr(self.env_mock, self.target_position_attribute)

    # endregion

    # region Cost components
    # TODO: Make it library agnostic. This also justifies why some methods are not static, although currently they could be
    def get_actuation_cost(self, u):
        cc_cost = R * (u ** 2)
        return tf.math.reduce_sum(cc_weight * cc_cost, axis=-1)

    def get_terminal_speed_cost(self, terminal_state):
        ''' Compute penality for deviation from desired max speed'''
        terminal_speed = terminal_state[:, LINEAR_VEL_X_IDX]

        desired_speed = tf.fill(tf.shape(terminal_speed), desired_max_speed)
        speed_diff = tf.abs(terminal_speed - desired_speed)
        terminal_speed_cost = terminal_speed_cost_weight * speed_diff

        return terminal_speed_cost

    # cost of changeing control to fast
    def get_control_change_rate_cost(self, u, u_prev):
        """Compute penalty of control jerk, i.e. difference to previous control input"""
        u_prev_vec = tf.concat((tf.ones((u.shape[0], 1, u.shape[-1])) * u_prev, u[:, :-1, :]), axis=1)
        ccrc = (u - u_prev_vec) ** 2
        return tf.math.reduce_sum(ccrc_weight * ccrc, axis=-1)

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

    def get_distances_from_trajectory_points_to_closest_target_point(self, trajectory_points, target_points):
        trajectories_shape = tf.shape(trajectory_points)
        number_of_rollouts = trajectories_shape[0]
        number_of_steps = trajectories_shape[1]
        number_of_target_points = tf.shape(target_points)[0]

        positions_of_trajectories = tf.reshape(trajectory_points, [trajectories_shape[0] * trajectories_shape[1], 2])
        distance_to_waypoints = self.distances_from_list_to_list_of_points(positions_of_trajectories, target_points)
        distance_to_waypoints_per_rollout = tf.reshape(distance_to_waypoints,
                                                       (number_of_rollouts, number_of_steps, number_of_target_points))
        min_distance_to_waypoints_per_rollout = tf.reduce_min(distance_to_waypoints_per_rollout, axis=2)

        return min_distance_to_waypoints_per_rollout

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

    def get_distance_to_waypoints_cost(self, trajectories, waypoints):
        return self.get_distances_from_trajectory_points_to_closest_target_point(trajectories,
                                                                            waypoints) * distance_to_waypoints_cost_weight


    def distances_from_list_to_list_of_points(self, points1, points2):

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
