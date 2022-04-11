"""
This is the class saving data for F1TENTH
"""

import os
import datetime

ranges_decimate = True  # If true, saves only every tenth LIDAR scan
ranges_forward_only = True # Only LIDAR scans in forward direction are saved

path_to_experiment_recordings =

class Recorder:
    def __init__(self):
        self.ranges_decimate = ranges_decimate
        self.ranges_forward_only = ranges_forward_only

        self.keys_ranges = None

        self.ranges_dict = None
        self.odometry_dict = None
        self.control_inputs_dict = None

    def get_ranges(self, ranges):

        ranges_to_save = ranges
        if ranges_forward_only:
            ranges_to_save = ranges_to_save[200:880]

        if self.ranges_decimate:
            ranges_to_save = ranges_to_save[:, :, 10]

        if self.keys_ranges is None:
            #Initialise
            self.keys_ranges = ['LIDAR_'+str(i) for i in range(len(ranges_to_save))]
        else:
            pass

        self.ranges_dict = dict(zip(self.keys_ranges, ranges_to_save))

    def get_odometry(self, odometry_dict):
        self.odometry_dict = odometry_dict

    def get_control_inputs(self, control_inputs):
        speed, steering = control_inputs
        self.control_inputs_dict = {'speed': speed, 'steering': steering}

    def create_csv_path(self, csv_name = None):

        # Make folder to save data (if not yet existing)
        try:
            os.makedirs(self.path_to_experiment_recordings)
        except FileExistsError:
            pass

        # Set path where to save the data
        if csv_name is None or csv_name == '':
            self.csv_filepath = self.path_to_experiment_recordings + 'CP_' + self.controller_name + str(
                datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')) + '.csv'
        else:
            self.csv_filepath = csv_name
            if csv_name[-4:] != '.csv':
                self.csv_filepath += '.csv'

    def reset(self):
        self.keys_ranges = None


