
import numpy as np
import math
from utilities.Settings import Settings
from utilities.waypoint_utils import WaypointUtils
from utilities.render_utilities import RenderUtils
from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    odometry_dict_to_state
)

from Control_Toolkit.Controllers.controller_mpc import controller_mpc
from Control_Toolkit_ASF.Controllers.MPC.TargetGenerator import TargetGenerator
from Control_Toolkit_ASF.Controllers.MPC.SpeedGenerator import SpeedGenerator


class mpc_planner:
    """
    Example Planner
    """

    def __init__(self):

        print("MPC planner initialized")

        self.translational_control = None
        self.angular_control = None

        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35, 2.35, 1080)
        self.simulation_index = 0

        self.largest_gap_middle_point = None
        self.largest_gap_middle_point = None

        self.nearest_waypoint_index = None

        self.time = 0.0
        self.waypoint_utils = WaypointUtils()

        self.lidar_points = np.zeros((216, 2), dtype=np.float32)
        self.target_point = np.array([0, 0], dtype=np.float32)

        if Settings.ENVIRONMENT_NAME == 'Car':
            num_states = 9
            num_control_inputs = 2
            if not Settings.WITH_PID:  # MPC return velocity and steering angle
                control_limits_low, control_limits_high = get_control_limits([7, 3.5])
            else:  # MPC returns acceleration and steering velocity
                control_limits_low, control_limits_high = get_control_limits([22, 1.2])
        else:
            raise NotImplementedError('{} mpc not implemented yet'.format(Settings.ENVIRONMENT_NAME))

        if Settings.CONTROLLER:
            self.mpc = controller_mpc(
                dt=Settings.TIMESTEP_CONTROL,
                environment_name="Car",
                initial_environment_attributes={
                    "lidar_points": self.lidar_points,
                    "next_waypoints": self.waypoint_utils.next_waypoints,
                    "target_point": self.target_point

                },
                num_states=num_states,
                num_control_inputs=num_control_inputs,
                control_limits=(control_limits_low, control_limits_high),
            )
        else:
            raise NotImplementedError

        self.mpc.configure()

        self.Render = RenderUtils()
        self.Render.waypoints = self.waypoint_utils.waypoint_positions
        
        self.car_state = [0, 0, 0, 0, 0, 0, 0]
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
        # Give it a little "Schupf"
        if self.simulation_index < 1:
            self.simulation_index += 1
            self.translational_control = 10
            self.angular_control = 0
            return self.translational_control, self.angular_control

        if Settings.ONLY_ODOMETRY_AVAILABLE:
            s = odometry_dict_to_state(ego_odom)
        else:
            s = self.car_state

        if Settings.LOOK_FORWARD_ONLY:
            lidar_range_min = 200
            lidar_range_max = 880
        else:
            lidar_range_min = 0
            lidar_range_max = -1
            
        scans = np.array(ranges)
        distances = scans[lidar_range_min:lidar_range_max:5]  # Only use every 5th lidar point
        angles = self.lidar_scan_angles[lidar_range_min:lidar_range_max:5]

        p1 = s[POSE_X_IDX] + distances * np.cos(angles + s[POSE_THETA_IDX])
        p2 = s[POSE_Y_IDX] + distances * np.sin(angles + s[POSE_THETA_IDX])
        self.lidar_points = np.stack((p1, p2), axis=1)


        self.target_point = [0, 0]  # dont need the target point for racing anymore

        if (Settings.FOLLOW_RANDOM_TARGETS):
            self.target_point = self.TargetGenerator.step((s[POSE_X_IDX],  s[POSE_Y_IDX]), )

        car_position = [s[POSE_X_IDX], s[POSE_Y_IDX]]
        self.waypoint_utils.update_next_waypoints(car_position)

        translational_control, angular_control = self.mpc.step(s,
                                                               self.time,
                                                               {
                                                                   "lidar_points": self.lidar_points,
                                                                   "next_waypoints": self.waypoint_utils.next_waypoints,
                                                                   "target_point": self.target_point,

                                                               })

        # This is the very fast controller: steering proportional to angle to the target, speed random
        # steering_angle = np.clip(self.TargetGenerator.angle_to_target((pose_x, pose_y), pose_theta), -0.2, 0.2)
        # translational_control = self.SpeedGenerator.step()
        # translational_control = 0.1

        # TODO: pass optimal trajectory
        self.Render.update(
            lidar_points=self.lidar_points,
            rollout_trajectory=self.mpc.logs['rollout_trajectories_logged'][-1],
            traj_cost=self.mpc.logs['J_logged'][-1],
            next_waypoints= self.waypoint_utils.next_waypoint_positions,
            car_state = s
        )
        
        self.translational_control = translational_control
        self.angular_control = angular_control
        
        # print("translational_control", translational_control)
        # print("angular_control", angular_control)
        
        self.simulation_index += 1

        return translational_control, angular_control




def get_control_limits(clip_control_input):
    if isinstance(clip_control_input[0], list):
        clip_control_input_low = np.array(clip_control_input[0])
        clip_control_input_high = np.array(clip_control_input[1])
    else:
        clip_control_input_high = np.array(clip_control_input)
        clip_control_input_low = -np.array(clip_control_input_high)

    return clip_control_input_low, clip_control_input_high

