
import numpy as np

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit.Controllers import template_controller

from utilities.Settings import Settings

if(Settings.ROS_BRIDGE):
    from utilities.waypoint_utils_ros import WaypointUtils
    from utilities.render_utilities_ros import RenderUtils
else:
    from utilities.waypoint_utils import WaypointUtils
    from utilities.render_utilities import RenderUtils

from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX, WP_PSI_IDX

from f110_gym.envs.base_classes import wrap_angle_rad
from SI_Toolkit.computation_library import TensorType

from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    LINEAR_VEL_X_IDX,
    control_limits_high,
    control_limits_low,
)

k = 0.5  # control gain

def calc_target_index(pose_x, pose_y, pose_theta, wypt_x, wypt_y):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """

    # Search nearest point index
    dx = pose_x-wypt_x
    dy = pose_y-wypt_y
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(pose_theta + np.pi / 2),
                      -np.sin(pose_theta + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle

class StanleyPlanner(template_planner):
    """
    Example Planner
    """

    def __init__(self):

        super().__init__()

        print("Stanley planner initialized")

        self.render_utils = RenderUtils()

        self.waypoint_utils=WaypointUtils()   # Only needed for initialization
        self.waypoints = self.waypoint_utils.next_waypoints

        self.stanley = controller_stanley(
            environment_name="Car",
            initial_environment_attributes={
                "next_waypoints": self.waypoints,

            },
            control_limits=(control_limits_low, control_limits_high),
        )

    def process_observation(self, ranges=None, ego_odom=None):


        angular_control, translational_control = self.stanley.step(self.car_state,
                                                               self.time,
                                                               {
                                                                   "next_waypoints": self.waypoints
                                                               })

        self.render_utils.update_pp(
            target_point=self.stanley.nearest_point,
        )

        self.translational_control = translational_control
        self.angular_control = angular_control

        self.simulation_index += 1

        return angular_control, translational_control


class controller_stanley(template_controller):

    def config(self):
        self.nearest_point = None
    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        """
        Stanley steering control.

        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        self.update_attributes(updated_attributes)


        wp_x = self.variable_parameters.next_waypoints[:, WP_X_IDX]
        wp_y = self.variable_parameters.next_waypoints[:, WP_Y_IDX]
        wp_vx = self.variable_parameters.next_waypoints[:, WP_VX_IDX]
        wp_yaw = self.variable_parameters.next_waypoints[:, WP_PSI_IDX]


        current_target_idx, error_front_axle = calc_target_index(
            s[POSE_X_IDX],
            s[POSE_Y_IDX],
            s[POSE_THETA_IDX],
            wp_x,
            wp_y,
        )

        self.nearest_point  = np.array((wp_x[current_target_idx], wp_y[current_target_idx], wp_vx[current_target_idx]))

        # theta_e corrects the heading error
        theta_e = wrap_angle_rad(wp_yaw[current_target_idx] - s[POSE_THETA_IDX])
        # theta_d corrects the cross track error
        theta_d = np.arctan2(k * error_front_axle, s[LINEAR_VEL_X_IDX])
        # Steering control
        angular_control = theta_e + theta_d
        translational_control = wp_vx[current_target_idx]

        control = np.array((angular_control, translational_control))
        control = np.clip(control, control_limits_low, control_limits_high)

        return control[0], control[1]


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle