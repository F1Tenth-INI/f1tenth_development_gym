import sys

sys.path.insert(1, 'FollowtheGap')

from pyglet.gl import GL_POINTS
import pyglet.gl as gl
from pyglet import shapes

import numpy as np
import pandas as pd
import yaml

from utilities.Settings import Settings
from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    LINEAR_VEL_X_IDX,
    odometry_dict_to_state
)

from Control_Toolkit_ASF.Controllers.FollowTheGap.ftg_planner import find_largest_gap_middle_point

if Settings.LOOK_FORWARD_ONLY:
    lidar_range_min = 200
    lidar_range_max = 880
else:
    lidar_range_min = 0
    lidar_range_max = -1

NUM_TRAJECTORIES_TO_PLOT = Settings.NUM_TRAJECTORIES_TO_PLOT

from Control_Toolkit.Controllers.controller_mpc import controller_mpc

from Control_Toolkit_ASF.Controllers.MPC.TargetGenerator import TargetGenerator
from Control_Toolkit_ASF.Controllers.MPC.SpeedGenerator import SpeedGenerator

class mpc_planner:
    """
    Example Planner
    """

    def __init__(self):

        print("Controller initialized")

        self.translational_control = None
        self.angular_control = None

        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35, 2.35, 1080)
        self.simulation_index = 0
        self.plot_lidar_data = False

        self.largest_gap_middle_point = None
        self.largest_gap_middle_point = None

        self.lidar_points = None
        self.waypoints = None
        self.target_point = np.array([0, 0])

        self.nearest_waypoint_index = None

        # Get waypoints
        self.wpts_opt = None
        if Settings.MAP_WAYPOINT_FILE is not None:
            path = Settings.MAP_WAYPOINT_FILE
            waypoints = pd.read_csv(path+'.csv', header=None).to_numpy()
            waypoints=waypoints[0:-1:1,1:3]
            self.wpts_opt=waypoints

        config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
        self.look_ahead_waypoints = config['planner']['LOOK_AHEAD_WAYPOINTS']
        self.interpolate_waypoints = config['planner']['INTERPOLATE_WAYPOINTS']

        if Settings.ENVIRONMENT_NAME == 'Car':
            num_states = 9
            num_control_inputs = 2
            if Settings.WITH_PID:  # MPC return velocity and steering angle
                control_limits_low = np.array([-22, -1.2])  # Possible negative velocity, mostly to force PID to stronger reaction
                control_limits_high = np.array([22, 1.2])
            else:  # MPC returns acceleration and steering velocity
                control_limits_low = np.array([-7, -3.5])  # Possible negative velocity, mostly to force PID to stronger reaction
                control_limits_high = np.array([7, 3.5])
        else:
            raise NotImplementedError('{} mpc not implemented yet'.format(Settings.ENVIRONMENT_NAME))

        if Settings.CONTROLLER:
            self.mpc = controller_mpc(
                environment_name="Car",
                initial_environment_attributes={
                    "lidar_points": self.lidar_points,
                    "waypoints": self.waypoints,
                    "target_point": self.target_point

                },
                num_states=num_states,
                num_control_inputs=num_control_inputs,
                control_limits=(control_limits_low, control_limits_high),
            )
        else:
            raise NotImplementedError

        self.Render = Render()
        self.car_state = [ 0 ,0, 0, 0, 0, 0, 0]
        self.TargetGenerator = TargetGenerator()
        self.SpeedGenerator = SpeedGenerator()



    def render(self, e):
        self.Render.render(e)

    def process_observation(self, ranges=None, ego_odom=None):
        """
        gives actuation given observation
        @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
        @ ego_odom: A dict with following indices:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
            'linear_vel_x': float,
            'linear_vel_y': float,
            'angular_vel_z': float,
        }
        """

        # Accelerate at the beginning (St model expoldes for small velocity)
        if self.simulation_index < 20:
            self.simulation_index += 1
            self.translational_control = 10
            self.angular_control = 0
            return self.translational_control, self.angular_control

        if Settings.ONLY_ODOMETRY_AVAILABLE:
            s = odometry_dict_to_state(ego_odom)
        else:
            s = self.car_state

        scans = np.array(ranges)

        distances = scans[lidar_range_min:lidar_range_max:5]  # Only use every 5th lidar point
        angles = self.lidar_scan_angles[lidar_range_min:lidar_range_max:5]

        p1 = s[POSE_X_IDX] + distances * np.cos(angles + s[POSE_THETA_IDX])
        p2 = s[POSE_Y_IDX] + distances * np.sin(angles + s[POSE_THETA_IDX])
        self.lidar_points = np.stack((p1, p2), axis=1)

        # self.largest_gap_middle_point, largest_gap_middle_point_distance, largest_gap_center = find_largest_gap_middle_point(pose_x, pose_y, pose_theta, distances, angles)
        # target_point = self.largest_gap_middle_point
        target_point = [0, 0]  # dont need the target point for racing anymore

        if (Settings.FOLLOW_RANDOM_TARGETS):
            target_point = self.TargetGenerator.step((s[POSE_X_IDX],  s[POSE_Y_IDX]), )

        self.f1t_model.LIDAR = self.lidar_points
        self.f1t_model.target_position = target_point

        self.f1t_model.waypoints, self.nearest_waypoint_index = get_nearest_waypoints(s,
                                                                                      self.wpts_opt,
                                                                                      self.f1t_model.waypoints,
                                                                                      self.nearest_waypoint_index,
                                                                                      look_ahead=self.look_ahead_waypoints)

        translational_control, angular_control = self.mpc.step(s)

        # This is the very fast controller: steering proportional to angle to the target, speed random
        # steering_angle = np.clip(self.TargetGenerator.angle_to_target((pose_x, pose_y), pose_theta), -0.2, 0.2)
        # translational_control = self.SpeedGenerator.step()
        # translational_control = 0.1

        self.Render.update(self.lidar_points, self.mpc.rollout_trajectory, self.mpc.traj_cost,
            self.mpc.optimal_trajectory, self.largest_gap_middle_point, target_point=target_point)
        self.Render.current_state = s

        self.simulation_index += 1

        self.translational_control = translational_control
        self.angular_control = angular_control

        return translational_control, angular_control


class Render:
    def __init__(self):

        self.draw_lidar_data = True
        self.draw_position_history = True

        self.lidar_visualization_color = (255, 0, 255)
        self.gap_visualization_color = (0, 255, 0)
        self.mppi_visualization_color = (250, 25, 30)
        self.optimal_trajectory_visualization_color = (255, 165, 0)
        self.target_point_visualization_color = (255, 204, 0)
        self.position_history_color = (0, 204, 0)

        self.lidar_vertices = None
        self.gap_vertex = None
        self.mppi_rollouts_vertices = None
        self.optimal_trajectory_vertices = None
        self.target_vertex = None

        self.lidar_border_points = None
        self.rollout_trajectory, self.traj_cost = None, None
        self.optimal_trajectory = None
        self.largest_gap_middle_point = None
        self.target_point = None
        self.current_state = None

    def update(self, lidar_points=None, rollout_trajectory=None, traj_cost=None, optimal_trajectory=None,
               largest_gap_middle_point=None, target_point=None):
        self.lidar_border_points = lidar_points
        if rollout_trajectory is not None:
            self.rollout_trajectory, self.traj_cost = rollout_trajectory, traj_cost
        if optimal_trajectory is not None:
            self.optimal_trajectory = optimal_trajectory
        self.largest_gap_middle_point = largest_gap_middle_point
        self.target_point = target_point

    def render(self, e):
        gl.glPointSize(3)

        if self.draw_position_history and self.current_state is not None:
            points = np.array([self.current_state[POSE_X_IDX], self.current_state[POSE_Y_IDX]])
            speed = self.current_state[LINEAR_VEL_X_IDX]

            scaled_points = 50.*points
            e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[0], scaled_points[1], 0.]),
                        ('c3B', (int(10 * speed), int(255- 10 * speed), 0)))


        if not self.draw_lidar_data: return

        if self.lidar_border_points is not None:
            scaled_points = 50.*np.array(self.lidar_border_points)
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            if self.lidar_vertices is None:
                self.lidar_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                               ('c3B', self.lidar_visualization_color * howmany))
            else:
                self.lidar_vertices.vertices = scaled_points_flat

        if self.largest_gap_middle_point is not None:

            scaled_point_gap = 50.0*np.array(self.largest_gap_middle_point)
            scaled_points_gap_flat = scaled_point_gap.flatten()
            self.gap_vertex = shapes.Circle(scaled_points_gap_flat[0], scaled_points_gap_flat[1], 5, color=self.gap_visualization_color, batch=e.batch)


        if self.rollout_trajectory is not None:
            num_trajectories_to_plot = np.minimum(NUM_TRAJECTORIES_TO_PLOT, self.rollout_trajectory.shape[0])
            trajectory_points = self.rollout_trajectory[:num_trajectories_to_plot, :, POSE_X_IDX:POSE_Y_IDX+1]

            scaled_trajectory_points = 50. * trajectory_points

            howmany_mppi = scaled_trajectory_points.shape[0]*scaled_trajectory_points.shape[1]
            scaled_trajectory_points_flat = scaled_trajectory_points.flatten()

            if self.mppi_rollouts_vertices is None:
                self.mppi_rollouts_vertices = e.batch.add(howmany_mppi, GL_POINTS, None, ('v2f/stream', scaled_trajectory_points_flat),
                                               ('c3B', self.mppi_visualization_color * howmany_mppi))
            else:
                self.mppi_rollouts_vertices.vertices = scaled_trajectory_points_flat

        if self.optimal_trajectory is not None:
            optimal_trajectory_points = self.optimal_trajectory[:, :, POSE_X_IDX:POSE_Y_IDX+1]

            scaled_optimal_trajectory_points = 50. * optimal_trajectory_points

            howmany_mppi_optimal = scaled_optimal_trajectory_points.shape[0]*scaled_optimal_trajectory_points.shape[1]
            scaled_optimal_trajectory_points_flat = scaled_optimal_trajectory_points.flatten()

            if self.optimal_trajectory_vertices is None:
                self.optimal_trajectory_vertices = e.batch.add(howmany_mppi_optimal, GL_POINTS, None, ('v2f/stream', scaled_optimal_trajectory_points_flat),
                                               ('c3B', self.optimal_trajectory_visualization_color * howmany_mppi_optimal))
            else:
                self.optimal_trajectory_vertices.vertices = scaled_optimal_trajectory_points_flat

        if self.target_point is not None and Settings.FOLLOW_RANDOM_TARGETS:

            scaled_target_point = 50.0*np.array(self.target_point)
            scaled_target_point_flat = scaled_target_point.flatten()
            self.target_vertex = shapes.Circle(scaled_target_point_flat[0], scaled_target_point_flat[1], 10, color=self.target_point_visualization_color, batch=e.batch)

def get_nearest_waypoint_index(s, wpts):

    min_dist = 10000
    min_dist_index = 0
    position = [s[POSE_X_IDX], s[POSE_Y_IDX]]

    def squared_distance(p1, p2):
        squared_distance = abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2
        return squared_distance

    for i in range(len(wpts)):

        dist = squared_distance(wpts[i], position)
        if (dist) < min_dist:
            min_dist = dist
            min_dist_index = i

    return min_dist_index

def get_nearest_waypoints(s, all_wpts, current_wpts, nearest_waypoint_index, look_ahead=15, interpolate_waypoints=4):
    if current_wpts is None:
        nearest_waypoint_index = get_nearest_waypoint_index(s, all_wpts)  # Run initial search of starting waypoint
    else:
        nearest_waypoint_index += get_nearest_waypoint_index(s, current_wpts)

    nearest_waypoints = []
    for j in range(look_ahead+1):
        next_waypoint = all_wpts[(nearest_waypoint_index + j) % len(all_wpts)]
        nearest_waypoints.append(next_waypoint)

    nearest_waypoints_interpolated = []
    for j in range(look_ahead):
        for i in range(interpolate_waypoints):
            interpolated_waypoint = nearest_waypoints[j] + (float(i)/interpolate_waypoints)*(nearest_waypoints[j+1]-nearest_waypoints[j])
            nearest_waypoints_interpolated.append(interpolated_waypoint)

    return nearest_waypoints_interpolated, nearest_waypoint_index