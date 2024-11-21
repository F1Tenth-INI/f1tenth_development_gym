import numpy as np

from SI_Toolkit.General.data_manager import DataManager
from SI_Toolkit.Functions.FunctionalDict import FunctionalDict

from utilities.Settings import Settings
from utilities.state_utilities import STATE_VARIABLES
from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX
from utilities.csv_logger import create_csv_file_name, create_csv_header, create_csv_file

from utilities.path_helper_ros import get_gym_path
gym_path = get_gym_path()  # FIXME: Do we need it here?

rounding_decimals = 5

class Recorder:
    def __init__(self, driver):

        self.driver = driver

        self.dict_data_to_save_basic = FunctionalDict(get_basic_data_dict(driver))

        self.data_to_save_measurement = {}
        self.data_to_save_controller = {}

        self.data_manager = DataManager(create_csv_file, rounding_decimals)

        self.csv_name = None
        self.recording_length = rounding_decimals
        self.start_recording_flag = False  # Gives signal to start recording during the current control iteration, starting recording may take more than one control iteration

        self.recording_mode = Settings.RECORDING_MODE
        self.path_to_experiment_recordings = Settings.RECORDING_FOLDER
        self.time_limited_recording_length = Settings.TIME_LIMITED_RECORDING_LENGTH

        self.dt = Settings.TIMESTEP_CONTROL

        self.controller_info = ''

        self.csv_filepath = None

    def step(self):
        self.csv_recording_step()

    @property
    def recording_running(self):
        return self.data_manager.recording_running

    @property
    def starting_recording(self):
        return self.data_manager.starting_recording

    def start_csv_recording(self, time_limited_recording=False):
        self.recording_on_off(time_limited_recording)
        self.start_csv_recording_if_requested()

    def recording_on_off(self, time_limited_recording=False):
        # (Exclude situation when recording is just being initialized, it may take more than one control iteration)
        if not self.starting_recording:
            if not self.recording_running:

                self.controller_info = self.driver.controller_name

                self.csv_name = create_csv_file_name(Settings)

                if time_limited_recording:
                    self.recording_length = self.time_limited_recording_length
                else:
                    self.recording_length = np.inf

                self.start_recording_flag = True

            else:
                self.finish_csv_recording(wait_till_complete=True)


    def start_csv_recording_if_requested(self):

        if self.start_recording_flag:
            self.start_recording_flag = False
            combined_keys = list(self.dict_data_to_save_basic.keys()) + list(
                self.data_to_save_measurement.keys()) + list(self.data_to_save_controller.keys())

            self.data_manager.start_csv_recording(
                self.csv_name,
                combined_keys,
                None,
                create_csv_header(Settings, self.controller_info, self.dt),
                self.path_to_experiment_recordings,
                mode=self.recording_mode,
                wait_till_complete=True,
                recording_length=self.recording_length
            )
            self.csv_filepath = self.data_manager.csv_filepath

    def csv_recording_step(self):
        if self.recording_running:
            self.data_manager.step([
                self.dict_data_to_save_basic,
                self.data_to_save_measurement,
                self.data_to_save_controller
            ])

    def finish_csv_recording(self, wait_till_complete=True):
        if self.recording_running:
            self.data_manager.finish_experiment(wait_till_complete=wait_till_complete)
        self.recording_length = np.inf



def get_basic_data_dict(driver):
    # The below dict lists variables to be logged with csv file when recording is on
    # Just add new variable here and it will be logged
    time_dict = {
        'time': lambda: driver.time,
    }

    state_dict = {
        state_variable: (lambda index=idx: driver.car_state[index])
        for idx, state_variable in enumerate(STATE_VARIABLES)
    }

    control_input_calculated_dict = {
        'angular_control_calculated': lambda: driver.angular_control,
        'translational_control_calculated': lambda: driver.translational_control,
    }

    # Creating lidar_names based on indices
    lidar_names = ['LIDAR_' + str(i).zfill(4) for i in driver.LIDAR.processed_scan_indices]

    # Creating lidar_ranges_dict with lambda functions that retrieve current lidar values
    lidar_ranges_dict = {
        lidar_name: (lambda index=idx: driver.LIDAR.processed_scans[index])
        for idx, lidar_name in enumerate(lidar_names)
    }

    next_waypoints_dict = get_next_waypoints_dict(driver)
    next_waypoints_relative_dict = get_next_waypoints_relative_dict(driver)

    imu_dict = {
        key: (lambda k=key: driver.current_imu_dict[k])
        for key in driver.current_imu_dict.keys()
    }

    # Combine all dictionaries into one
    combined_dict = {
        **time_dict,
        **state_dict,
        **control_input_calculated_dict,
        **lidar_ranges_dict,
        **next_waypoints_dict,
        **next_waypoints_relative_dict,
        **imu_dict,
    }

    return combined_dict


def get_next_waypoints_dict(driver):

    # Retrieve the next waypoints
    next_waypoints = driver.waypoint_utils.next_waypoints

    # Initialise keys for X, Y, and Velocity waypoints
    num_waypoints = next_waypoints.shape[0]  # Assuming waypoints are rows in a 2D array
    keys_next_x_waypoints = ['WYPT_X_' + str(i).zfill(2) for i in range(num_waypoints)]
    keys_next_y_waypoints = ['WYPT_Y_' + str(i).zfill(2) for i in range(num_waypoints)]
    keys_next_vx_waypoints = ['WYPT_VX_' + str(i).zfill(2) for i in range(num_waypoints)]

    # Create dictionary with lambda functions to dynamically retrieve the X values
    next_waypoints_dict = {
        key: (lambda index=idx: next_waypoints[index, WP_X_IDX])
        for idx, key in enumerate(keys_next_x_waypoints)
    }

    # Update dictionary with lambda functions for Y values
    next_waypoints_dict.update({
        key: (lambda index=idx: driver.waypoint_utils.next_waypoints[index, WP_Y_IDX])
        for idx, key in enumerate(keys_next_y_waypoints)
    })

    # Update dictionary with lambda functions for Velocity (VX) values
    next_waypoints_dict.update({
        key: (lambda index=idx: driver.waypoint_utils.next_waypoints[index, WP_VX_IDX])
        for idx, key in enumerate(keys_next_vx_waypoints)
    })

    return next_waypoints_dict



def get_next_waypoints_relative_dict(driver):

    next_waypoints_relative = driver.waypoint_utils.next_waypoint_positions_relative

    num_waypoints = np.array(next_waypoints_relative[::Settings.INTERPOLATION_STEPS]).shape[0]  # Number of waypoints (rows)

    # Initialise keys for next X and Y waypoints relative using the number of waypoints
    keys_next_x_waypoints_rel = ['WYPT_REL_X_' + str(i).zfill(2) for i in range(num_waypoints)]
    keys_next_y_waypoints_rel = ['WYPT_REL_Y_' + str(i).zfill(2) for i in range(num_waypoints)]
    
     # Create dictionary with lambda functions to dynamically retrieve the X values
    next_waypoints_relative_dict = {
        key: (lambda index=idx: next_waypoints_relative[index, 0])
        for idx, key in enumerate(keys_next_x_waypoints_rel)
    }

    # Update dictionary with lambda functions for Y values
    next_waypoints_relative_dict.update({
        key: (lambda index=idx: next_waypoints_relative[index, 1])
        for idx, key in enumerate(keys_next_y_waypoints_rel)
    })

    return next_waypoints_relative_dict
