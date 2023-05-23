import sklearn # Don't touch
# sklearn is needed later, need to import it here for nni planner to work on nvidia jetson:
# https://forums.developer.nvidia.com/t/sklearn-skimage-cannot-allocate-memory-in-static-tls-block/236960

from utilities.waypoint_utils import *

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit.Controllers.controller_neural_imitator import controller_neural_imitator

class NeuralNetImitatorPlanner(template_planner):

    def __init__(self, speed_fraction=1, batch_size=1):

        print('Loading NeuralNetImitatorPlanner')

        super().__init__()


        print('Loading NeuralNetImitatorPlanner')

        self.simulation_index = 0

        self.waypoint_utils = None  # Will be overwritten with a WaypointUtils instance from car_system

        self.nni = controller_neural_imitator(
            dt=Settings.TIMESTEP_CONTROL,
            environment_name="Car",
            initial_environment_attributes={},
            control_limits=(control_limits_low, control_limits_high),
        )

        self.nni.configure()

        number_of_next_waypoints_network = len([wp for wp in self.nni.net_info.inputs if wp.startswith("WYPT_REL_X")])
        if number_of_next_waypoints_network != Settings.LOOK_AHEAD_STEPS:
            raise ValueError('Number of waypoints required by network ({}) different than that set in Settings ({})'.format(number_of_next_waypoints_network, Settings.LOOK_AHEAD_STEPS))

    def process_observation(self, ranges=None, ego_odom=None):

        self.LIDAR.load_lidar_measurement(ranges)

        if Settings.LIDAR_CORRUPT:
            self.LIDAR.corrupt_lidar_set_indices()
            self.LIDAR.corrupt_scans()
        self.LIDAR.corrupted_scans_high2zero()

        self.LIDAR.plot_lidar_data()

        # The NNI planner needs relativa waypoints in any case
        waypoints_relative = WaypointUtils.get_relative_positions(self.waypoints, self.car_state)

        #Split up Waypoint Tuples into WYPT_X and WYPT_Y because Network used this format in training from CSV
        waypoints_relative_x = waypoints_relative[:, 0]
        waypoints_relative_y = waypoints_relative[:, 1]

        #Load Waypoint Velocities for next n (defined in Settings) waypoints
        next_waypoint_vx = self.waypoints[:, WP_VX_IDX]
        
        #In training all inputs are ordered alphabetically according to their index -> first LIDAR, then WYPTS, then States (because not capital letters)
        #Example of all possible inputs in correct order:
        # Order has to stay the same: SAME AS IN Config_training
        # If we want to change, look at recording
        #input_data = np.concatenate((ranges, next_waypoints_x, next_waypoints_y,
        #                              [self.car_state[ANGULAR_VEL_Z_IDX], self.car_state[LINEAR_VEL_X_IDX],
        #                              self.car_state[POSE_THETA_COS_IDX], self.car_state[POSE_THETA_SIN_IDX],
        #                              self.car_state[POSE_X_IDX], self.car_state[POSE_Y_IDX]]), axis=0)
        
        #Current Input:
        input_data = np.concatenate((self.LIDAR.processed_scans, waypoints_relative_x, waypoints_relative_y, next_waypoint_vx,
                                      [self.car_state[ANGULAR_VEL_Z_IDX], self.car_state[LINEAR_VEL_X_IDX], self.car_state[SLIP_ANGLE_IDX], self.car_state[STEERING_ANGLE_IDX]]), axis=0)


        # input_data = np.concatenate((self.LIDAR.processed_scans,
        #                               [self.car_state[ANGULAR_VEL_Z_IDX], self.car_state[LINEAR_VEL_X_IDX], self.car_state[SLIP_ANGLE_IDX], self.car_state[STEERING_ANGLE_IDX]]), axis=0)

        net_output = self.nni.step(input_data)

        self.angular_control = net_output[0, 0, 0]
        self.translational_control = net_output[0, 0, 1]

        # Accelerate at the beginning "Schupf" (St model explodes for small velocity) -> must come after loading of waypoints otherwise they aren't saved
        if self.simulation_index < Settings.ACCELERATION_TIME:
            self.simulation_index += 1
            self.translational_control = Settings.ACCELERATION_AMPLITUDE
            self.angular_control = 0
            return self.angular_control, self.translational_control,


        return self.angular_control, self.translational_control
        

