import sys

sys.path.insert(1, 'FollowtheGap')

from pyglet.gl import GL_POINTS
import pyglet
import numpy as np

from Settings import Settings

from FollowTheGap.ftg_planner import find_largest_gap_middle_point

if Settings.LOOK_FORWARD_ONLY:
    lidar_range_min = 200
    lidar_range_max = 880
else:
    lidar_range_min = 0
    lidar_range_max = -1

from MPPI_Marcin.controller_mppi_tf import controller_mppi_tf

class MPPI_F1TENTH:
    """
    Example Planner
    """

    def __init__(self, speed_fraction=1):

        print("Controller initialized")

        self.lidar_border_points = 1080 * [[0, 0]]
        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35, 2.35, 1080)
        self.simulation_index = 0
        self.plot_lidar_data = False
        self.draw_lidar_data = False
        self.lidar_visualization_color = (0, 0, 0)

        self.vertex_list = pyglet.graphics.vertex_list(2,
                                                       ('v2i', (10, 15, 30, 35)),
                                                       ('c3B', (0, 0, 255, 0, 255, 0))
                                                       )

        self.mppi = controller_mppi_tf()

    def render(self, e):

        if not self.draw_lidar_data: return

        self.vertex_list.delete()

        scaled_points = np.array(self.lidar_border_points)
        howmany = scaled_points.shape[0]
        scaled_points_flat = scaled_points.flatten()

        self.vertex_list = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                       ('c3B', self.lidar_visualization_color * howmany))

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

        scans = np.array(ranges)
        # Take into account size of car
        # scans -= 0.3

        distances = scans[lidar_range_min:lidar_range_max:10] # Only use every 10th lidar point
        angles = self.lidar_scan_angles[lidar_range_min:lidar_range_max:10]

        p1 = pose_x + distances * np.cos(angles + pose_theta)
        p2 = pose_y + distances * np.sin(angles + pose_theta)
        lidar_points = np.stack((p1, p2), axis=1)

        self.lidar_border_points = 50*lidar_points

        largest_gap_middle_point, largest_gap_middle_point_distance, largest_gap_center = find_largest_gap_middle_point(pose_x, pose_y, pose_theta, distances, angles)

        target = np.vstack((largest_gap_middle_point, lidar_points))
        s = np.array((pose_x, pose_y, pose_theta))
        speed, steering_angle = self.mppi.step(s, target=target)
        print(speed)

        # Speed Calculation
        # speed = largest_gap_middle_point_distance
        # steering_angle = largest_gap_center


        self.simulation_index += 1
        return speed, steering_angle
