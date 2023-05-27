
import numpy as np
import math
from utilities.Settings import Settings
from utilities.obstacle_detector import ObstacleDetector

from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    odometry_dict_to_state,
    control_limits_low,
    control_limits_high,
)

if(Settings.ROS_BRIDGE):
    from utilities.waypoint_utils_ros import WaypointUtils
    from utilities.render_utilities_ros import RenderUtils
else:
    from utilities.waypoint_utils import WaypointUtils
    from utilities.render_utilities import RenderUtils


from Control_Toolkit.Controllers.controller_mpc import controller_mpc
from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit_ASF.Controllers.MPC.TargetGenerator import TargetGenerator
from Control_Toolkit_ASF.Controllers.MPC.SpeedGenerator import SpeedGenerator


class mpc_planner(template_planner):
    """
    Example Planner
    """

    def __init__(self):

        super().__init__()

        print("MPC planner initialized")
        self.render_utils = RenderUtils()

        self.simulation_index = 0

        self.nearest_waypoint_index = None

        self.time = 0.0

        self.waypoint_utils=WaypointUtils()   # Only needed for initialization
        self.waypoints = self.waypoint_utils.next_waypoints

        self.obstacles = np.zeros((ObstacleDetector.number_of_fixed_length_array, 2), dtype=np.float32)

        self.target_point = np.array([0, 0], dtype=np.float32)


        self.mpc = controller_mpc(
            dt=Settings.TIMESTEP_PLANNER,
            environment_name="Car",
            initial_environment_attributes={
                "obstacles": self.obstacles,
                "lidar_points": self.lidar_points,
                "next_waypoints": self.waypoints,
                "target_point": self.target_point

            },
            control_limits=(control_limits_low, control_limits_high),
        )

        self.mpc.configure()
        
        self.car_state = None
        self.TargetGenerator = TargetGenerator()
        self.SpeedGenerator = SpeedGenerator()

    def set_obstacles(self, obstacles):

        self.obstacles =  ObstacleDetector.get_fixed_length_obstacle_array(obstacles)


    def process_observation(self, ranges=None, ego_odom=None):

        self.LIDAR.load_lidar_measurement(ranges)

        if Settings.LIDAR_CORRUPT:
            self.LIDAR.corrupt_lidar_set_indices()
            self.LIDAR.corrupt_scans()

        # self.LIDAR.plot_lidar_data()

        self.lidar_points = self.LIDAR.get_processed_lidar_points_in_map_coordinates(
            self.car_state[POSE_X_IDX], self.car_state[POSE_Y_IDX], self.car_state[POSE_THETA_IDX]
        )

        # Accelerate at the beginning (St model expoldes for small velocity)
        # Give it a little "Schupf"
        if self.simulation_index < Settings.ACCELERATION_TIME:
            self.simulation_index += 1
            self.translational_control = Settings.ACCELERATION_AMPLITUDE
            self.angular_control = 0
            return self.angular_control, self.translational_control

        if (Settings.FOLLOW_RANDOM_TARGETS):
            self.target_point = self.TargetGenerator.step((self.car_state[POSE_X_IDX],  self.car_state[POSE_Y_IDX]), )


        angular_control, translational_control  = self.mpc.step(self.car_state,
                                                               self.time,
                                                               {
                                                                   "obstacles": self.obstacles,
                                                                   "lidar_points": self.lidar_points,
                                                                   "next_waypoints": self.waypoint_utils.next_waypoints,
                                                                   "target_point": self.target_point,

                                                               })

        # This is the very fast controller: steering proportional to angle to the target, speed random
        # steering_angle = np.clip(self.TargetGenerator.angle_to_target((pose_x, pose_y), pose_theta), -0.2, 0.2)
        # translational_control = self.SpeedGenerator.step()
        # translational_control = 0.1


        rollout_trajectories = None
        optimal_trajectory = None
        traj_cost = None

        if hasattr(self.mpc.optimizer, 'rollout_trajectories'):
            rollout_trajectories = self.mpc.optimizer.rollout_trajectories
            self.rollout_trajectories = rollout_trajectories[:20,:,:].numpy()
        if hasattr(self.mpc.optimizer, 'optimal_trajectory'):
            optimal_trajectory = self.mpc.optimizer.optimal_trajectory
        if hasattr(self.mpc.optimizer, 'optimal_control_sequence') and self.mpc.optimizer.optimal_control_sequence is not None:
            self.optimal_control_sequence = self.mpc.optimizer.optimal_control_sequence[0]
        if self.mpc.controller_logging:
            traj_cost = self.mpc.logs['J_logged'][-1]

        self.render_utils.update_mpc(
            rollout_trajectory=rollout_trajectories,
            optimal_trajectory=optimal_trajectory,
        )

        self.translational_control = translational_control
        self.angular_control = angular_control
        
        self.simulation_index += 1

        return angular_control, translational_control

