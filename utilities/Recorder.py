"""
This is the class saving data for F1TENTH
"""

import os
from datetime import datetime
import csv
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import shutil

try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

from utilities.state_utilities import FULL_STATE_VARIABLES

from utilities.path_helper_ros import get_gym_path
gym_path = get_gym_path()

config = yaml.load(open(os.path.join(gym_path, "config.yml"), "r"), Loader=yaml.FullLoader)
waypoint_interpolation_steps = config["waypoints"]["INTERPOLATION_STEPS"]


ranges_decimate = True  # If true, saves only every tenth LIDAR scan
ranges_forward_only = True # Only LIDAR scans in forward direction are saved

rounding_decimals = 5

path_to_experiment_recordings = 'ExperimentRecordings/'


def create_csv_header(path_to_recordings,
                      controller_name,
                      dt,
                      csv_name=None):

    # Make folder to save data (if not yet existing)
    try:
        os.makedirs(path_to_recordings[:-1])
    except FileExistsError:
        pass

    # Set path where to save the data
    if csv_name is None or csv_name == '':
        csv_filepath = path_to_recordings + 'F1TENTH_' + controller_name + '_' + str(
            datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')) + '.csv'
    else:
        csv_filepath = csv_name
        if csv_name[-4:] != '.csv':
            csv_filepath += '.csv'

        # If such file exists, append index to the end (do not overwrite)
        net_index = 1
        logpath_new = csv_filepath
        while True:
            if os.path.isfile(logpath_new):
                logpath_new = csv_filepath[:-4]
            else:
                csv_filepath = logpath_new
                break
            logpath_new = logpath_new + '-' + str(net_index) + '.csv'
            net_index += 1

    # Write the header of .csv file
    with open(csv_filepath, "a") as outfile:
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

        writer.writerow(['# Saving: {} s'.format(dt)])

        writer.writerow(['#'])

        writer.writerow(['# Controller: {}'.format(controller_name)])

        writer.writerow(['#'])

        writer.writerow(['# Data:'])

    return csv_filepath


class Recorder:
    def __init__(self, controller_name=None, create_header=True, dt=None):

        self.dt = dt

        self.ranges_decimate = ranges_decimate
        self.ranges_forward_only = ranges_forward_only
        self.path_to_experiment_recordings = path_to_experiment_recordings

        if controller_name is None:
            self.controller_name = ''
        else:
            self.controller_name = controller_name

        self.rounding_decimals = 3

        self.headers_already_saved = False

        self.keys_time = None
        self.keys_ranges = None
        self.keys_odometry = None
        self.keys_state = None
        self.keys_control_inputs_applied = None
        self.keys_control_inputs_calculated = None
        self.keys_next_x_waypoints = None
        self.keys_next_y_waypoints = None

        self.csv_filepath = None

        self.time_dict = None
        self.ranges_dict = {}
        self.odometry_dict = {}
        self.state_dict = {}
        self.control_inputs_applied_dict = {}
        self.control_inputs_calculated_dict = {}
        self.dict_to_save = {}
        self.next_waypoints_dict = {}

        if create_header:
            self.csv_filepath = create_csv_header(
                self.path_to_experiment_recordings,
                self.controller_name,
                self.dt,
            )


#ToDo safe upper/lower bound of ranges and filter in config.yml
    def get_ranges(self, ranges):

        ranges_to_save = ranges
        if ranges_forward_only:
            ranges_to_save = ranges_to_save[200:880]

        if self.ranges_decimate:
            ranges_to_save = ranges_to_save[::10]

        if self.keys_ranges is None:
            #Initialise
            self.keys_ranges = ['LIDAR_'+str(i).zfill(4) for i in range(len(ranges_to_save))]

        self.ranges_dict = dict(zip(self.keys_ranges, ranges_to_save))

    def get_odometry(self, odometry_dict):
        self.odometry_dict = odometry_dict
        if self.keys_odometry is None:
            self.keys_odometry = self.odometry_dict.keys()

    def get_time(self, time):
        self.time_dict = {'time': time}
        if self.keys_time is None:
            self.keys_time = self.time_dict.keys()

    def get_control_inputs_applied(self, control_inputs):
        translational_control, angular_control = control_inputs
        self.control_inputs_applied_dict = {'translational_control_applied': translational_control, 'angular_control_applied': angular_control}
        if self.keys_control_inputs_applied is None:
            self.keys_control_inputs_applied = self.control_inputs_applied_dict.keys()

    def get_control_inputs_calculated(self, control_inputs):
        translational_control, angular_control = control_inputs
        self.control_inputs_calculated_dict = {'translational_control_calculated': translational_control, 'angular_control_calculated': angular_control}
        if self.keys_control_inputs_calculated is None:
            self.keys_control_inputs_calculated = self.control_inputs_calculated_dict.keys()


    def get_next_waypoints(self, next_waypoints):
        waypoints_to_save = np.array(next_waypoints[::waypoint_interpolation_steps])
        waypoints_x_to_save = waypoints_to_save[:,1]
        waypoints_y_to_save = waypoints_to_save[:, 0]

        if self.keys_next_x_waypoints is None:
            # Initialise
            self.keys_next_x_waypoints = ['WYPT_X_' + str(i) for i in range(len(waypoints_x_to_save))]

        if self.keys_next_y_waypoints is None:
            # Initialise
            self.keys_next_y_waypoints = ['WYPT_Y_' + str(i) for i in range(len(waypoints_y_to_save))]

        self.next_waypoints_dict = dict(zip(self.keys_next_x_waypoints, waypoints_x_to_save))
        self.next_waypoints_dict.update(zip(self.keys_next_y_waypoints, waypoints_y_to_save))

    def get_state(self, state):

        state_to_save = state

        if self.keys_state is None:
            self.keys_state = FULL_STATE_VARIABLES

        self.state_dict = dict(zip(self.keys_state, state_to_save))

    def get_data(self, control_inputs_applied=None, control_inputs_calculated=None, odometry=None, ranges=None, time=None, state=None, next_waypoints=None):
        if time is not None:
            self.get_time(time)
        if control_inputs_applied is not None:
            self.get_control_inputs_applied(control_inputs=control_inputs_applied)
        if control_inputs_calculated is not None:
            self.get_control_inputs_calculated(control_inputs=control_inputs_calculated)
        if odometry is not None:
            self.get_odometry(odometry_dict=odometry)
        if state is not None:
            self.get_state(state=state)
        if ranges is not None:
            self.get_ranges(ranges=ranges)
        if next_waypoints is not None and len(next_waypoints) > 0:
            self.get_next_waypoints(next_waypoints=next_waypoints)

    def save_data(self, control_inputs_applied=None, control_inputs_calculated=None, odometry=None, ranges=None, time=None, state=None, next_waypoints=None):
        self.get_data(control_inputs_applied, control_inputs_calculated, odometry, ranges, time, state, next_waypoints)
        self.save_csv()

    def save_csv(self):

        self.dict_to_save.update(self.time_dict)
        self.dict_to_save.update(self.control_inputs_applied_dict)
        self.dict_to_save.update(self.control_inputs_calculated_dict)
        self.dict_to_save.update(self.odometry_dict)
        self.dict_to_save.update(self.state_dict)
        self.dict_to_save.update(self.ranges_dict)
        self.dict_to_save.update(self.next_waypoints_dict)



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

    '''
    Generate plots from the csv file of the recording
    '''
    def plot_data(self):
        
        save_path = self.csv_filepath[:-4]+"_data"
        os.mkdir(save_path)
        df = pd.read_csv(self.csv_filepath, header = 0, skiprows=range(0,8))

        angular_controls = df['angular_control_applied'].to_numpy()[1:]
        translational_controls = df['translational_control_applied'].to_numpy()[1:]   
        
        # Plot Angular Control
        plt.title("Angular Control")
        plt.plot(angular_controls, color="red")
        plt.savefig(save_path+"/angular_control.png")
        plt.clf()

        # Plot Translational Control
        plt.title("Translational Control")
        plt.plot(translational_controls, color="blue")
        plt.savefig(save_path+"/translational_control.png")
        plt.clf()
        
        
        # Copy Settings and configs
        config_sage_path = os.path.join(save_path, "configs")
        os.mkdir(config_sage_path)
        shutil.copy("Control_Toolkit_ASF/config_controllers.yml", config_sage_path) 
        shutil.copy("Control_Toolkit_ASF/config_cost_function.yml", config_sage_path) 
        shutil.copy("Control_Toolkit_ASF/config_optimizers.yml", config_sage_path) 
        shutil.copy("utilities/Settings.py", config_sage_path) 
        
        
    def reset(self):
        self.headers_already_saved = False
        self.keys_time = None
        self.keys_ranges = None
        self.keys_odometry = None
        self.keys_control_inputs_applied = None
        self.keys_control_inputs_calculated = None
        self.csv_filepath = None
        self.time_dict = None
        self.ranges_dict = None
        self.odometry_dict = None
        self.control_inputs_applied_dict = None
        self.control_inputs_calculated_dict = None
        self.next_waypoints_dict = None
        self.dict_to_save = {}

