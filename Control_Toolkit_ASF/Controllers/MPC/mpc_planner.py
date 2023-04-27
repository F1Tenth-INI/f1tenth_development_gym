
import numpy as np
import math
from utilities.Settings import Settings

from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    odometry_dict_to_state
)

if(Settings.ROS_BRIDGE):
    from utilities.waypoint_utils_ros import WaypointUtils
    from utilities.render_utilities_ros import RenderUtils
else:
    from utilities.waypoint_utils import WaypointUtils
    from utilities.render_utilities import RenderUtils


from Control_Toolkit.Controllers.controller_mpc import controller_mpc
from Control_Toolkit_ASF.Controllers.MPC.TargetGenerator import TargetGenerator
from Control_Toolkit_ASF.Controllers.MPC.SpeedGenerator import SpeedGenerator


class mpc_planner:
    """
    Example Planner
    """

    def __init__(self):

        print("MPC planner initialized")
        self.render_utils = RenderUtils()


        self.translational_control = None
        self.angular_control = None

        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35, 2.35, 1080)
        self.simulation_index = 0

        self.largest_gap_middle_point = None
        self.largest_gap_middle_point = None

        self.nearest_waypoint_index = None

        self.time = 0.0


        self.waypoint_utils=WaypointUtils()   # Only needed for initialization
        self.waypoints = self.waypoint_utils.next_waypoints

        self.lidar_points = np.zeros((216, 2), dtype=np.float32)
        self.target_point = np.array([0, 0], dtype=np.float32)

        if Settings.ENVIRONMENT_NAME == 'Car':
            num_states = 9
            num_control_inputs = 2
            if not Settings.WITH_PID:  # MPC return velocity and steering angle
                control_limits_low, control_limits_high = get_control_limits([[-3.2, -9.5], [3.2, 9.5]])
            else:  # MPC returns acceleration and steering velocity
                control_limits_low, control_limits_high = get_control_limits([[-1.066, -1], [1.066, 8]])
        else:
            raise NotImplementedError('{} mpc not implemented yet'.format(Settings.ENVIRONMENT_NAME))

        if Settings.CONTROLLER:
            self.mpc = controller_mpc(
                dt=Settings.TIMESTEP_CONTROL,
                environment_name="Car",
                initial_environment_attributes={
                    "lidar_points": self.lidar_points,
                    "next_waypoints": self.waypoints,
                    "target_point": self.target_point

                },
                num_states=num_states,
                num_control_inputs=num_control_inputs,
                control_limits=(control_limits_low, control_limits_high),
            )
        else:
            raise NotImplementedError

        self.mpc.configure()
        
        self.car_state = [0,0,0,0,0,0,0]
        self.TargetGenerator = TargetGenerator()
        self.SpeedGenerator = SpeedGenerator()

    # def render(self, e):
    #     self.render_utils.render(e)

    def set_waypoints(self, waypoints):
        self.waypoints =  np.array(waypoints).astype(np.float32)

    def set_car_state(self, car_state):
        self.car_state = np.array(car_state).astype(np.float32)


    def process_observation(self, ranges=None, ego_odom=None):

        s = self.car_state

        # Accelerate at the beginning (St model expoldes for small velocity)
        # Give it a little "Schupf"
        if self.simulation_index < Settings.ACCELERATION_TIME:
            self.simulation_index += 1
            self.translational_control = Settings.ACCELERATION_AMPLITUDE
            self.angular_control = 0
            return self.angular_control, self.translational_control

        if Settings.LOOK_FORWARD_ONLY:
            lidar_range_min = 200
            lidar_range_max = 880
        else:
            lidar_range_min = 0
            lidar_range_max = -1
            
        distances = ranges[lidar_range_min:lidar_range_max:5]  # Only use every 5th lidar point
        angles = self.lidar_scan_angles[lidar_range_min:lidar_range_max:5]

        p1 = s[POSE_X_IDX] + distances * np.cos(angles + s[POSE_THETA_IDX])
        p2 = s[POSE_Y_IDX] + distances * np.sin(angles + s[POSE_THETA_IDX])
        self.lidar_points = np.stack((p1, p2), axis=1)


        self.target_point = [0, 0]  # dont need the target point for racing anymore

        if (Settings.FOLLOW_RANDOM_TARGETS):
            self.target_point = self.TargetGenerator.step((s[POSE_X_IDX],  s[POSE_Y_IDX]), )


        angular_control, translational_control  = self.mpc.step(s,
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
        

        self.last_steering = angular_control
        self.translational_control = translational_control
        self.angular_control = angular_control
        
        # print("translational_control", translational_control)
        # print("angular_control", angular_control)
        
        self.simulation_index += 1

        return angular_control, translational_control




def get_control_limits(clip_control_input):
    if isinstance(clip_control_input[0], list):
        clip_control_input_low = np.array(clip_control_input[0])
        clip_control_input_high = np.array(clip_control_input[1])
    else:
        clip_control_input_high = np.array(clip_control_input)
        clip_control_input_low = -np.array(clip_control_input_high)

    return clip_control_input_low, clip_control_input_high

