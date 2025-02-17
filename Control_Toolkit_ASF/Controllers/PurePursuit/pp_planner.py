import numpy as np
from numba import jit
from utilities.Settings import Settings

from utilities.waypoint_utils import *
from utilities.car_files.vehicle_parameters import VehicleParameters

from Control_Toolkit_ASF.Controllers.PurePursuit.pp_helpers import *
from Control_Toolkit_ASF.Controllers import template_planner
from utilities.state_utilities import *

from SI_Toolkit.Functions.General.hyperbolic_functions import return_hyperbolic_function

'''
Example PP planner, adapted to our system
'''
class PurePursuitPlanner(template_planner):
    """
    Example Planner
    """

    def __init__(self):

        super().__init__()
    
        print("Initializing PP Planner")

        self.render_utils = RenderUtils()

        car_parameters = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
        self.lidar_points = 1080 * [[0,0]]
        self.lidar_scan_angles = np.linspace(-2.35,2.35, 1080)

        self.speed = 1.

        
        # Controller settings
        self.waypoint_velocity_factor = Settings.PP_WAYPOINT_VELOCITY_FACTOR
        self.lookahead_distance =  Settings.PP_LOOKAHEAD_DISTANCE 
        self.wheelbase = car_parameters.lf +  car_parameters.lr 
        self.max_reacquire = 20.
        
        self.simulation_index = 0
        self.correcting_index = 0

        self.current_position = None
        self.curvature_integral = 0
        self.translational_control = None
        self.angular_control = None
        
        self.angular_control = 0.
        self.translational_control = 0.


        self.hyperbolic_function_for_curvature_factor, _, _ = return_hyperbolic_function((0.0, 1.0), (1.0, 0.0) , fixed_point=Settings.PP_FIXPOINT_FOR_CURVATURE_FACTOR)

        self.pp_use_curvature_correction = Settings.PP_USE_CURVATURE_CORRECTION

        self.f_max = 0.0
        self.f_min = 1.0
        self.lookahead_point = None

        position = np.array([0., 0.], dtype=np.float32).copy()
        wpts = np.zeros((Settings.LOOK_AHEAD_STEPS, 2), dtype=np.float32)
        nearest_point_on_trajectory(position, wpts)

        print('Initialization done.')
 
        
        
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

        pose_x = self.car_state[POSE_X_IDX]
        pose_y = self.car_state[POSE_Y_IDX]
        pose_theta = self.car_state[POSE_THETA_IDX]
        v_x = self.car_state[LINEAR_VEL_X_IDX]

        position = np.array([pose_x, pose_y], dtype=np.float32).copy()
        
        if(Settings.PP_VEL2LOOKAHEAD):
            self.lookahead_distance = v_x * Settings.PP_VEL2LOOKAHEAD

        self.lookahead_distance = np.clip(self.lookahead_distance, a_min=(0.7), a_max=None)

        # Needs too much time
        lookahead_point, i, i2 = get_current_waypoint(self.waypoints, self.lookahead_distance, position, pose_theta)
        if self.waypoints[i, WP_VX_IDX] < 0:
            index_switch = 1
            for idx in range(1, len(self.waypoints[i:])):
                if self.waypoints[i+idx, WP_VX_IDX] > 0:
                    index_switch = i + idx
                    break
            lookahead_point = np.concatenate((self.waypoints[index_switch, (WP_X_IDX, WP_Y_IDX)], self.waypoints[i, WP_VX_IDX:WP_VX_IDX+1]))

        else:
            if self.pp_use_curvature_correction:
                # LOOKAHEAD_CURVATURE = 5
                LOOKAHEAD_CURVATURE = np.min((i2-i+1, len(self.waypoints)-1))
                curvature = self.waypoints[i: i + LOOKAHEAD_CURVATURE, WP_KAPPA_IDX]
                speeds = self.waypoints[i: i + LOOKAHEAD_CURVATURE, WP_VX_IDX]

                v_max = Settings.PP_NORMING_V_FOR_CURRVATURE
                v_abs_mean = np.mean(np.abs(speeds))
                kappa_abs_mean = np.mean(np.abs(curvature))
                f = np.dot(np.abs(curvature), np.abs(speeds))/(v_max*LOOKAHEAD_CURVATURE)

                f = np.clip(self.hyperbolic_function_for_curvature_factor(f), 0.0, 1.0)

                # print('Lookahead distance: {}'.format(self.lookahead_distance))

                if Settings.PRINTING_ON and Settings.ROS_BRIDGE is False:
                    if self.f_max < f:
                        self.f_max = f
                    elif self.f_min > f:
                        self.f_min = f
                    if self.simulation_index % 20 == 0:
                        print('')
                        print('LOOKAHEAD_CURVATURE: {}'.format(LOOKAHEAD_CURVATURE))
                        print('Mean abs speed: {}'.format(v_abs_mean))
                        print('Mean abs curvature {}'.format(kappa_abs_mean))
                        print('Curvature factor: {}'.format(f))
                        print('Curvature factor max: {}'.format(self.f_max))
                        print('Curvature factor min: {}'.format(self.f_min))
                        print('')
                        pass

                curvature_slowdown_factor = f
                self.lookahead_distance = np.max((self.lookahead_distance * curvature_slowdown_factor, Settings.PP_MINIMAL_LOOKAHEAD_DISTANCE))
                lookahead_point, i, i2 = get_current_waypoint(self.waypoints, self.lookahead_distance, position, pose_theta)

            if lookahead_point is None:
                if Settings.PRINTING_ON and Settings.ROS_BRIDGE is False:
                    print("warning no lookahead point")
                lookahead_point = self.waypoints[Settings.PP_BACKUP_LOOKAHEAD_POINT_INDEX]
                lookahead_point = [lookahead_point[WP_X_IDX],lookahead_point[WP_Y_IDX],lookahead_point[WP_VX_IDX]]
             
        
        # For rendering
        self.lookahead_point = lookahead_point
        
        
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, self.lookahead_distance, self.wheelbase)
        speed = self.waypoint_velocity_factor * speed

        if( abs(steering_angle) > 1.4):
            self.correcting_index+=1
            if(self.correcting_index >= 10):
                steering_angle = -steering_angle
                speed = -1
        else:
            self.correcting_index = 0

        self.speed = speed
        # print(self.speed)

        self.angular_control = steering_angle
        self.translational_control = speed

        self.render_utils.update_pp(
            target_point=lookahead_point,
        )

        self.simulation_index += 1

        return steering_angle, speed



        

