import sys

sys.path.insert(1, 'FollowtheGap')

from pyglet.gl import GL_POINTS
from pyglet import shapes

import numpy as np

from Settings import Settings

from FollowTheGap.ftg_planner import find_largest_gap_middle_point

if Settings.LOOK_FORWARD_ONLY:
    lidar_range_min = 200
    lidar_range_max = 880
else:
    lidar_range_min = 0
    lidar_range_max = -1

NUM_TRAJECTORIES_TO_PLOT = Settings.NUM_TRAJECTORIES_TO_PLOT

from MPPI_Marcin.controller_mppi_tf import controller_mppi_tf

from MPPI_Marcin.TargetGenerator import TargetGenerator
from MPPI_Marcin.SpeedGenerator import SpeedGenerator

class MPPI_F1TENTH:
    """
    Example Planner
    """

    def __init__(self):

        print("Controller initialized")

        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35, 2.35, 1080)
        self.simulation_index = 0
        self.plot_lidar_data = False

        self.largest_gap_middle_point = None
        self.lidar_points = None
        self.largest_gap_middle_point = None


        self.mppi = controller_mppi_tf()

        self.Render = Render()

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
        pose_x = ego_odom['pose_x']
        pose_y = ego_odom['pose_y']
        pose_theta = ego_odom['pose_theta']

        target_positions = self.TargetGenerator.step((pose_x, pose_y), )

        scans = np.array(ranges)
        # Take into account size of car
        # scans -= 0.3

        distances = scans[lidar_range_min:lidar_range_max:10] # Only use every 10th lidar point
        angles = self.lidar_scan_angles[lidar_range_min:lidar_range_max:10]

        p1 = pose_x + distances * np.cos(angles + pose_theta)
        p2 = pose_y + distances * np.sin(angles + pose_theta)
        self.lidar_points = np.stack((p1, p2), axis=1)

        self.largest_gap_middle_point, largest_gap_middle_point_distance, largest_gap_center = find_largest_gap_middle_point(pose_x, pose_y, pose_theta, distances, angles)

        target = np.vstack((self.largest_gap_middle_point, self.lidar_points))
        # target = np.vstack((target_positions, lidar_points))
        s = np.array((pose_x, pose_y, pose_theta))
        speed, steering_angle = self.mppi.step(s, target=target)

        # This is the very fast controller: steering proportional to angle to the target, speed random
        # steering_angle = np.clip(self.TargetGenerator.angle_to_target((pose_x, pose_y), pose_theta), -0.2, 0.2)
        # speed = self.SpeedGenerator.step()

        self.Render.update(self.lidar_points, self.mppi.rollout_trajectory, self.mppi.traj_cost,
                           self.mppi.optimal_trajectory, self.largest_gap_middle_point, target_point=target_positions)
        self.simulation_index += 1

        return speed, steering_angle


class Render:
    def __init__(self):

        self.draw_lidar_data = True

        self.lidar_visualization_color = (0, 0, 0)
        self.gap_visualization_color = (0, 255, 0)
        self.mppi_visualization_color = (250, 25, 30)
        self.optimal_trajectory_visualization_color = (255, 165, 0)
        self.target_point_visualization_color = (255, 204, 0)

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

    def update(self, lidar_points=None, rollout_trajectory=None, traj_cost=None, optimal_trajectory=None,
               largest_gap_middle_point=None, target_point=None):
        self.lidar_border_points = lidar_points
        self.rollout_trajectory, self.traj_cost = rollout_trajectory, traj_cost
        self.optimal_trajectory = optimal_trajectory
        self.largest_gap_middle_point = largest_gap_middle_point
        self.target_point = target_point

    def render(self, e):
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
            trajectory_points = self.rollout_trajectory[:num_trajectories_to_plot, :, :2]

            scaled_trajectory_points = 50. * trajectory_points

            howmany_mppi = scaled_trajectory_points.shape[0]*scaled_trajectory_points.shape[1]
            scaled_trajectory_points_flat = scaled_trajectory_points.flatten()

            if self.mppi_rollouts_vertices is None:
                self.mppi_rollouts_vertices = e.batch.add(howmany_mppi, GL_POINTS, None, ('v2f/stream', scaled_trajectory_points_flat),
                                               ('c3B', self.mppi_visualization_color * howmany_mppi))
            else:
                self.mppi_rollouts_vertices.vertices = scaled_trajectory_points_flat

        if self.optimal_trajectory is not None:
            optimal_trajectory_points = self.optimal_trajectory[:, :, :2]

            scaled_optimal_trajectory_points = 50. * optimal_trajectory_points

            howmany_mppi_optimal = scaled_optimal_trajectory_points.shape[0]*scaled_optimal_trajectory_points.shape[1]
            scaled_optimal_trajectory_points_flat = scaled_optimal_trajectory_points.flatten()

            if self.optimal_trajectory_vertices is None:
                self.optimal_trajectory_vertices = e.batch.add(howmany_mppi_optimal, GL_POINTS, None, ('v2f/stream', scaled_optimal_trajectory_points_flat),
                                               ('c3B', self.optimal_trajectory_visualization_color * howmany_mppi_optimal))
            else:
                self.optimal_trajectory_vertices.vertices = scaled_optimal_trajectory_points_flat

        if self.target_point is not None:

            scaled_target_point = 50.0*np.array(self.target_point)
            scaled_target_point_flat = scaled_target_point.flatten()
            self.gap_vertex = shapes.Circle(scaled_target_point_flat[0], scaled_target_point_flat[1], 10, color=self.target_point_visualization_color, batch=e.batch)

