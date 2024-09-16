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
            environment_name="Car",
            initial_environment_attributes={},
            control_limits=(control_limits_low, control_limits_high),
        )

        self.nni.configure()
        self.nn_inputs = self.nni.remaining_inputs
        
        if 'GRU' in self.nni.config_controller['net_name']:
            Settings.ACCELERATION_TIME = 10 # GRU needs a little whashout 
            print("GRU detected... set acceleration time to 10")

        number_of_next_waypoints_network = len([wp for wp in self.nn_inputs if wp.startswith("WYPT_REL_X")])
        if number_of_next_waypoints_network > Settings.LOOK_AHEAD_STEPS:
            raise ValueError('Number of waypoints required by network ({}) different than that set in Settings ({})'.format(number_of_next_waypoints_network, Settings.LOOK_AHEAD_STEPS))

    def process_observation(self, ranges=None, ego_odom=None):

        self.LIDAR.load_lidar_measurement(ranges)

        if Settings.LIDAR_CORRUPT:
            self.LIDAR.corrupt_lidar_set_indices()
            self.LIDAR.corrupt_scans()

        self.LIDAR.corrupted_scans_high2zero()


        # Build a dict data_dict, to store all environment and sensor data that we have access to
        # The NNI will then extract the data it needs from this dict
        # If you have access to more data than waypoints, state, etc, add it to the dict
        
        # About timing: Building the lidar dict takes 1ms, bulding the whole dict takes 1.3ms
        # If you need NNI to be running faster, you dont calculate the dics but build the array by hand.

        # Lidar dict
        lidar_keys = self.LIDAR.get_all_lidar_scans_names()
        lidar_values = self.LIDAR.all_lidar_scans
        lidar_dict = dict(zip(lidar_keys, lidar_values))

        # Waypoint dict
        waypoints_relative = WaypointUtils.get_relative_positions(self.waypoints, self.car_state)
        waypoints_relative_x = waypoints_relative[:, 0]
        waypoints_relative_y = waypoints_relative[:, 1]
        next_waypoint_vx = self.waypoints[:, WP_VX_IDX]
        
        keys_next_x_waypoints = ['WYPT_REL_X_' + str(i).zfill(2) for i in range(len(waypoints_relative_x))]
        keys_next_y_waypoints = ['WYPT_REL_Y_' + str(i).zfill(2) for i in range(len(waypoints_relative_y))]
        keys_next_vx_waypoints = ['WYPT_VX_' + str(i).zfill(2) for i in range(len(next_waypoint_vx))]
        waypoints_dict = {
            **dict(zip(keys_next_x_waypoints, waypoints_relative_x)),
            **dict(zip(keys_next_y_waypoints, waypoints_relative_y)),
            **dict(zip(keys_next_vx_waypoints, next_waypoint_vx))
        }

        # Carstate dict
        state_dict = {state_name: self.car_state[STATE_INDICES[state_name]] for state_name in STATE_VARIABLES}
        
        state_pacejka = full_state_alphabetical_to_original(self.car_state)[4]
        state_dict["v_y"] = state_pacejka

        # Combine all dictionaries into one
        data_dict = {**waypoints_dict, **state_dict, **lidar_dict}
        
        # Check if every key in self.nn_inputs is present in data_dict
        if not all(key in data_dict for key in self.nn_inputs):
            missing_keys = [key for key in self.nn_inputs if key not in data_dict]
            raise Exception(f"Not all data the NN needs for input are present. The following keys are missing from data_dict: {missing_keys}")
        
        # Extract all data from dict that NN needs as input
        input_data = [data_dict[key] for key in self.nn_inputs if key in data_dict]
        
        # NN prediction step 
        net_output = self.nni.step(input_data)
        
        if net_output.shape[2] == 3:
            self.angular_control = net_output[0, 0, 0]
            fricition = net_output[0, 0, 1]
            self.translational_control = net_output[0, 0, 2]
            print("Estimated friction: ", fricition)
        else:
            self.angular_control = net_output[0, 0, 0]
            self.translational_control = net_output[0, 0, 1]
        
        # Accelerate at the beginning "Schupf" (St model explodes for small velocity) -> must come after loading of waypoints otherwise they aren't saved
        if self.simulation_index < Settings.ACCELERATION_TIME:
            self.simulation_index += 1
            self.translational_control = Settings.ACCELERATION_AMPLITUDE
            self.angular_control = 0
            return self.angular_control, self.translational_control,


        return self.angular_control, self.translational_control
        
