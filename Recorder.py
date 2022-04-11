"""
This is the class saving data for F1TENTH
"""

import os
from datetime import datetime
import csv
import numpy as np

try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

ranges_decimate = True  # If true, saves only every tenth LIDAR scan
ranges_forward_only = True # Only LIDAR scans in forward direction are saved

rounding_decimals = 3

path_to_experiment_recordings = '../ExperimentRecordings/'

class Recorder:
    def __init__(self, controller_name=None, create_header=True):
        self.ranges_decimate = ranges_decimate
        self.ranges_forward_only = ranges_forward_only
        self.path_to_experiment_recordings = path_to_experiment_recordings

        if controller_name is None:
            self.controller_name = ''
        else:
            self.controller_name = controller_name

        self.rounding_decimals = 3

        self.headers_already_saved = False
        self.keys_ranges = None
        self.keys_odometry = None
        self.keys_control_inputs = None
        self.csv_filepath = None
        self.ranges_dict = None
        self.odometry_dict = None
        self.control_inputs_dict = None
        self.dict_to_save = {}

        if create_header:
            self.create_csv_header()

    def get_ranges(self, ranges):

        ranges_to_save = ranges
        if ranges_forward_only:
            ranges_to_save = ranges_to_save[200:880]

        if self.ranges_decimate:
            ranges_to_save = ranges_to_save[::10]

        if self.keys_ranges is None:
            #Initialise
            self.keys_ranges = ['LIDAR_'+str(i) for i in range(len(ranges_to_save))]

        self.ranges_dict = dict(zip(self.keys_ranges, ranges_to_save))

    def get_odometry(self, odometry_dict):
        self.odometry_dict = odometry_dict
        if self.keys_odometry is None:
            self.keys_odometry = self.odometry_dict.keys()

    def get_control_inputs(self, control_inputs):
        speed, steering = control_inputs
        self.control_inputs_dict = {'speed': speed, 'steering': steering}
        if self.keys_control_inputs is None:
            self.keys_control_inputs = self.control_inputs_dict.keys()

    def save_data(self, control_inputs, odometry, ranges):
        self.get_control_inputs(control_inputs=control_inputs)
        self.get_odometry(odometry_dict=odometry)
        self.get_ranges(ranges=ranges)
        self.save_csv()

    def save_csv(self):

        self.dict_to_save.update(self.control_inputs_dict)
        self.dict_to_save.update(self.odometry_dict)
        self.dict_to_save.update(self.ranges_dict)

        # Save this dict
        with open(self.csv_filepath, "a") as outfile:
            writer = csv.writer(outfile)

            if not self.headers_already_saved:
                writer.writerow(self.dict_to_save.keys())
                self.headers_already_saved = True

            if self.rounding_decimals == np.inf:
                pass
            else:
                self.dict_to_save = {key: np.around(value, self.rounding_decimals)
                                     for key, value in self.dict_to_save.items()}

            writer.writerow([float(x) for x in self.dict_to_save.values()])

    def create_csv_header(self, csv_name=None):

        # Make folder to save data (if not yet existing)
        try:
            os.makedirs(self.path_to_experiment_recordings[:-1])
        except FileExistsError:
            pass

        # Set path where to save the data
        if csv_name is None or csv_name == '':
            self.csv_filepath = self.path_to_experiment_recordings + 'F1TENTH_' + self.controller_name + '_' + str(
                datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')) + '.csv'
        else:
            self.csv_filepath = csv_name
            if csv_name[-4:] != '.csv':
                self.csv_filepath += '.csv'

            # If such file exists, append index to the end (do not overwrite)
            net_index = 1
            logpath_new = self.csv_filepath
            while True:
                if os.path.isfile(logpath_new):
                    logpath_new = self.csv_filepath[:-4]
                else:
                    self.csv_filepath = logpath_new
                    break
                logpath_new = logpath_new + '-' + str(net_index) + '.csv'
                net_index += 1

        # Write the header of .csv file
        with open(self.csv_filepath, "a") as outfile:
            writer = csv.writer(outfile)

            writer.writerow(['# ' + 'This is F1TENTH simulation from {} at time {}'
                            .format(datetime.now().strftime('%d.%m.%Y'), datetime.now().strftime('%H:%M:%S'))])
            try:
                repo = Repo('../')
                git_revision = repo.head.object.hexsha
            except:
                git_revision = 'unknown'
            writer.writerow(['# ' + 'Done with git-revision: {}'
                            .format(git_revision)])

            writer.writerow(['#'])

            writer.writerow(['# Controller: {}'.format(self.controller_name)])

            writer.writerow(['#'])

            writer.writerow(['# Data:'])

    def reset(self):
        self.headers_already_saved = False
        self.keys_ranges = None
        self.keys_odometry = None
        self.keys_control_inputs = None
        self.csv_filepath = None
        self.ranges_dict = None
        self.odometry_dict = None
        self.control_inputs_dict = None
        self.dict_to_save = {}


