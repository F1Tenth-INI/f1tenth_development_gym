"""
This is the class saving data for F1TENTH
"""

import csv
import numpy as np

from utilities.Settings import Settings
from utilities.waypoint_utils import WP_X_IDX, WP_Y_IDX, WP_VX_IDX
from utilities.state_utilities import STATE_VARIABLES, ANGULAR_CONTROL_IDX, TRANSLATIONAL_CONTROL_IDX

from utilities.saving_csv_header import create_csv_header
from utilities.path_helper_ros import get_gym_path
gym_path = get_gym_path()

rounding_decimals = 5


class Recorder:
    def __init__(self, recorder_index=None):

        # Settings
        self.path_to_experiment_recordings = Settings.RECORDING_FOLDER
        self.rounding_decimals = 3

        # Init
        self.headers_already_saved = False
        self.csv_filepath = None

        self.dict_to_save = dict()
        self.dict_buffer = []  # Buffer to save dicts on RAM instead of disk

        controller_name = Settings.CONTROLLER
        controller_name = controller_name if recorder_index is None else controller_name + '-' + str(recorder_index)
        
        if not self.path_to_experiment_recordings.endswith('/'):
            self.path_to_experiment_recordings += '/'
        self.csv_filepath, self.experiment_name = create_csv_header(
                self.path_to_experiment_recordings,
                controller_name,
                Settings.TIMESTEP_CONTROL,
            )
    
    '''
    Pass data to the recorder:
    The recorder builds up a dictionary, which is saved every timestep.
    For some complex or required data types, the creation uf the dict is already implemented ( time, state, next_waypoints, etc)
    For non pre defined data, you can use custom_dict.
    @param custom_dict: dictionary, is appended at the end of the recording
    
    Note: set_data will not save the data. Make sure you call Recorder.push_on_buffer() after set_data
    Note: Also make sure to call save_csv at the end of the simulation
    '''
    def set_data(
            self, 
            time=None,                          # float32
            control_inputs_applied=None,        # Control Input (float32)[2]
            control_inputs_calculated=None,     # Control Input (float32)[2]
            odometry=None,                      # Dict???
            lidar_ranges=None,                  # Lidar Ranges (float32)[0:1080]
            lidar_indices=None,                 # Lidar Indices between 0 and 1080 (int32)[0:1080]
            state=None,                         # Car State (float32)[9]
            next_waypoints=None, 
            next_waypoints_relative=None,
            custom_dict = None):                # Custom Dictionary appended to the recording data
        
        data_dict = dict() 
        
        if(time is not None):
            data_dict['time'] = time
            
        if(state is not None):
            for index, state_variable in enumerate(STATE_VARIABLES):
                data_dict[state_variable] = state[index]
        
        if(control_inputs_applied is not None):
            data_dict['angular_control_applied'] = control_inputs_applied[ANGULAR_CONTROL_IDX]
            data_dict['translational_control_applied'] = control_inputs_applied[TRANSLATIONAL_CONTROL_IDX]
            
        if(control_inputs_calculated is not None):
            data_dict['angular_control_calculated'] = control_inputs_calculated[ANGULAR_CONTROL_IDX]
            data_dict['translational_control_calculated'] = control_inputs_calculated[TRANSLATIONAL_CONTROL_IDX]
        
        if(next_waypoints is not None):
            waypoint_dict = get_next_waypoints_dict(next_waypoints)
            data_dict.update(waypoint_dict)
        
        if(next_waypoints_relative is not None):
            waypoint_relative_dict = get_next_waypoints_relative_dict(next_waypoints_relative)
            data_dict.update(waypoint_relative_dict)

        if(lidar_ranges is not None and lidar_indices is not None):
            lidar_names = ['LIDAR_' + str(i).zfill(4) for i in lidar_indices]
            lidar_dict = dict(zip(lidar_names, lidar_ranges))
            data_dict.update(lidar_dict)
        
        if(custom_dict is not None):
            data_dict.update(custom_dict)

        self.dict_to_save.update(data_dict)

    
    '''
    Push data passed by Recorder.set_data() to the data buffer on RAM (very fast)
    '''
    def push_on_buffer(self,): # Do at every control stel
        self.dict_buffer.append(self.dict_to_save.copy())
    
    '''
    Save data buffer array to CSV
    Please only call once in a while but not every timestep
    Dont forget to call at the end of simulation
    '''
    def save_csv(self):

        # Save this dict
        with open(self.csv_filepath, "a") as outfile:
            writer = csv.writer(outfile)

            if not self.headers_already_saved:
                
                if len(self.dict_buffer) > 0:
                    writer.writerow(self.dict_buffer[-1].keys())
                    self.headers_already_saved = True

            for single_dict in self.dict_buffer:
                single_dict = {key: np.around(value, self.rounding_decimals) for key, value in single_dict.items()}
                writer.writerow([float(x) for x in single_dict.values()])
        self.dict_buffer = []


def get_next_waypoints_dict(next_waypoints):

    waypoints_x_to_save = next_waypoints[:, WP_X_IDX]
    waypoints_y_to_save = next_waypoints[:, WP_Y_IDX]
    waypoints_vel_to_save = next_waypoints[:, WP_VX_IDX]

    # Initialise
    keys_next_x_waypoints = ['WYPT_X_' + str(i).zfill(2) for i in range(len(waypoints_x_to_save))]
    keys_next_y_waypoints = ['WYPT_Y_' + str(i).zfill(2) for i in range(len(waypoints_y_to_save))]
    keys_next_vx_waypoints = ['WYPT_VX_' + str(i).zfill(2) for i in range(len(waypoints_y_to_save))]

    next_waypoints_dict = dict(zip(keys_next_x_waypoints, waypoints_x_to_save))
    next_waypoints_dict.update(zip(keys_next_y_waypoints, waypoints_y_to_save))
    next_waypoints_dict.update(zip(keys_next_vx_waypoints, waypoints_vel_to_save))

    return next_waypoints_dict


def get_next_waypoints_relative_dict(next_waypoints_relative):

    waypoints_to_save = np.array(next_waypoints_relative[::Settings.INTERPOLATION_STEPS])
    waypoints_x_to_save = waypoints_to_save[:, 0]
    waypoints_y_to_save = waypoints_to_save[:, 1]

    # Initialise
    keys_next_x_waypoints_rel = ['WYPT_REL_X_' + str(i).zfill(2) for i in range(len(waypoints_x_to_save))]
    keys_next_y_waypoints_rel = ['WYPT_REL_Y_' + str(i).zfill(2) for i in range(len(waypoints_y_to_save))]

    next_waypoints_dict = dict(zip(keys_next_x_waypoints_rel, waypoints_x_to_save))
    next_waypoints_dict.update(zip(keys_next_y_waypoints_rel, waypoints_y_to_save))

    return next_waypoints_dict

