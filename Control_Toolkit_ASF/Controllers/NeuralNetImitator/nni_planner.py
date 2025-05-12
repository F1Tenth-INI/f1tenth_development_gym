import sklearn # Don't touch
# sklearn is needed later, need to import it here for nni planner to work on nvidia jetson:
# https://forums.developer.nvidia.com/t/sklearn-skimage-cannot-allocate-memory-in-static-tls-block/236960

from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils
from utilities.Settings import Settings
from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit.Controllers.controller_neural_imitator import controller_neural_imitator
from collections import deque  # Import deque for an efficient rolling buffer

class NeuralNetImitatorPlanner(template_planner):

    def __init__(self, speed_fraction=1, batch_size=1):

        print('Loading NeuralNetImitatorPlanner')

        super().__init__()


        print('Loading NeuralNetImitatorPlanner')

        self.simulation_index = 0
        self.friction_estimate = []
        self.friction = 0
        self.waypoint_utils = None  # Will be overwritten with a WaypointUtils instance from car_system
        self.LIDAR = None  # Will be overwritten with a LidarUtils instance from car_system
        self.render_utils = RenderUtils()
        
        # Initialize a rolling buffer with zeros for control history
        self.control_history_size = 10  # Set the desired buffer size
        self.control_history = deque([(0.0, 0.0)] * self.control_history_size, maxlen=self.control_history_size)


        self.nni = controller_neural_imitator(
            environment_name="Car",
            initial_environment_attributes={},
            control_limits=(control_limits_low, control_limits_high),
        )

        self.nni.configure()
        self.nn_inputs = self.nni.input_mapping.keys()
        
        if 'GRU' in self.nni.config_controller['net_name']:
            Settings.ACCELERATION_TIME = 10 # GRU needs a little whashout 
            print("GRU detected... set acceleration time to 10")

        number_of_next_waypoints_network = len([wp for wp in self.nn_inputs if wp.startswith("WYPT_REL_X")])
        if number_of_next_waypoints_network > Settings.LOOK_AHEAD_STEPS:
            raise ValueError('Number of waypoints required by network ({}) different than that set in Settings ({})'.format(number_of_next_waypoints_network, Settings.LOOK_AHEAD_STEPS))

    def process_observation(self, ranges=None, ego_odom=None):


        # Build a dict data_dict, to store all environment and sensor data that we have access to
        # The NNI will then extract the data it needs from this dict
        # If you have access to more data than waypoints, state, etc, add it to the dict
        
        # About timing: Building the lidar dict takes 1ms, bulding the whole dict takes 1.3ms
        # If you need NNI to be running faster, you dont calculate the dics but build the array by hand.

        # Lidar dict
        # lidar_keys = self.LIDAR.get_all_lidar_ranges_names()
        # lidar_values = self.LIDAR.all_lidar_ranges
        # lidar_dict = dict(zip(lidar_keys, lidar_values))
        lidar_dict = {}
        # Waypoint dict
        waypoints_relative = self.waypoint_utils.get_relative_positions(self.waypoints, self.car_state)
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
        
        env_dict = {
            'mu': Settings.SURFACE_FRICTION,
        }

        control_dict = {
            'angular_control_calculated_-1': self.control_history[-1][0],
            'translational_control_calculated_-1': self.control_history[-1][1],
        }
        # Carstate dict
        state_dict = {state_name: self.car_state[STATE_INDICES[state_name]] for state_name in STATE_VARIABLES}
        # print("angular_vel_z: ", state_dict.get('angular_vel_z', 'Key not found'))        # Combine all dictionaries into one
        data_dict = {**waypoints_dict, **state_dict, **lidar_dict, **env_dict, **control_dict}
        
        # Check if every key in self.nn_inputs is present in data_dict
        if not all(key in data_dict for key in self.nn_inputs):
            missing_keys = [key for key in self.nn_inputs if key not in data_dict]
            raise Exception(f"Not all data the NN needs for input are present. The following keys are missing from data_dict: {missing_keys}")
        
        # Extract all data from dict that NN needs as input
        input_data = [data_dict[key] for key in self.nn_inputs if key in data_dict]
        
        # NN prediction step 
        net_output = self.nni.step(input_data).numpy()
        
        if net_output.shape[2] == 3:
            self.angular_control = net_output[0, 0, 0]
            fricition = net_output[0, 0, 1]
            self.translational_control = net_output[0, 0, 2]

            # Estimate friction over a period of time (AVERAGE_WINDOW)
            if len(self.friction_estimate) < Settings.AVERAGE_WINDOW:
                self.friction_estimate.append(fricition)
                self.friction = np.mean(self.friction_estimate)
            else:
                self.friction_estimate.pop(0)
                self.friction_estimate.append(fricition)
                self.friction = np.mean(self.friction_estimate)
            
            # self.render_utils.set_label_dict({'5: friction_estimated': self.friction,})
            # print("Estimated friction: ", self.friction)
        else:
            self.angular_control = net_output[0, 0, 0]
            self.translational_control = net_output[0, 0, 1]
        
        # Accelerate at the beginning "Schupf" (St model explodes for small velocity) -> must come after loading of waypoints otherwise they aren't saved
        if self.simulation_index < Settings.ACCELERATION_TIME:
            self.simulation_index += 1
            self.translational_control = Settings.ACCELERATION_AMPLITUDE
            self.angular_control = 0

        self.control_history.append((self.angular_control, self.translational_control))

        
        return self.angular_control, self.translational_control
        
