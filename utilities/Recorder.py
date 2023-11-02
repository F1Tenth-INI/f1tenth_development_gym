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
import json
from utilities.Settings import Settings

from utilities.waypoint_utils import *


try:
    # Use gitpython to get a current revision number and use it in description of experimental data
    from git import Repo
except:
    pass

from utilities.state_utilities import FULL_STATE_VARIABLES

from utilities.path_helper_ros import get_gym_path
gym_path = get_gym_path()

waypoint_interpolation_steps = Settings.INTERPOLATION_STEPS

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
    def __init__(self, name=""):

        # Settings
        self.name = name
        self.path_to_experiment_recordings = path_to_experiment_recordings
        self.rounding_decimals = 3

        # Init
        self.headers_already_saved = False
        self.csv_filepath = None
        
        self.global_dict= dict()
        self.dict_to_save =dict()
        self.dict_buffer = [] # Buffer to save dicts on RAM instead of disk
        
        self.csv_filepath = create_csv_header(
                self.path_to_experiment_recordings,
                self.name,
                0.1,
            )

    def set_global_data(self, global_dict):
        self.global_dict.update(global_dict)
        self.dict_to_save.update(global_dict)        
    
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
            for index, state_variable in enumerate(FULL_STATE_VARIABLES):
                data_dict[state_variable] = state[index]
        
        if(control_inputs_applied is not None):
            data_dict['angular_control_applied'] = control_inputs_applied[ANGULAR_CONTROL_IDX]
            data_dict['translational_control_applied'] = control_inputs_applied[TRANSLATIONAL_CONTROL_IDX]
            
        if(control_inputs_calculated is not None):
            data_dict['angular_control_calculated'] = control_inputs_calculated[ANGULAR_CONTROL_IDX]
            data_dict['translational_control_calculated'] = control_inputs_calculated[TRANSLATIONAL_CONTROL_IDX]
        
        if(next_waypoints is not None):
            waypoint_dict = self.get_next_waypoints_dict(next_waypoints)
            data_dict.update(waypoint_dict)
        
        if(next_waypoints_relative is not None):
            waypoint_relative_dict = self.get_next_waypoints_relative_dict(next_waypoints_relative)
            data_dict.update(waypoint_relative_dict)

        if(lidar_ranges is not None and lidar_indices is not None):
            lidar_names = ['LIDAR_' + str(i).zfill(4) for i in lidar_indices]
            lidar_dict = dict(zip(lidar_names, lidar_ranges))
            data_dict.update(lidar_dict)
        
        if(custom_dict is not None):
            data_dict.update(custom_dict)
        
        data_dict.update(self.global_dict)        
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
                
                if(len(self.dict_buffer) > 0):
                    writer.writerow(self.dict_buffer[-1].keys())
                    self.headers_already_saved = True

            for dict in self.dict_buffer:
                dict = {key: np.around(value, self.rounding_decimals)
                                        for key, value in dict.items()}
                writer.writerow([float(x) for x in dict.values()])
        self.dict_buffer = []
        
        
    def get_next_waypoints_dict(self, next_waypoints):
        
        waypoints_x_to_save = next_waypoints[:, WP_X_IDX]
        waypoints_y_to_save = next_waypoints[:, WP_Y_IDX]
        waypoints_vel_to_save = next_waypoints[:, WP_VX_IDX]

        # Initialise
        next_waypoints_dict = dict()
        keys_next_x_waypoints = ['WYPT_X_' + str(i).zfill(2) for i in range(len(waypoints_x_to_save))]
        keys_next_y_waypoints = ['WYPT_Y_' + str(i).zfill(2) for i in range(len(waypoints_y_to_save))]
        keys_next_vx_waypoints = ['WYPT_VX_' + str(i).zfill(2) for i in range(len(waypoints_y_to_save))]

        next_waypoints_dict = dict(zip(keys_next_x_waypoints, waypoints_x_to_save))
        next_waypoints_dict.update(zip(keys_next_y_waypoints, waypoints_y_to_save))
        next_waypoints_dict.update(zip(keys_next_vx_waypoints, waypoints_vel_to_save))
        
        return next_waypoints_dict
        
    def get_next_waypoints_relative_dict(self, next_waypoints_relative):
        
        waypoints_to_save = np.array(next_waypoints_relative[::waypoint_interpolation_steps])
        waypoints_x_to_save = waypoints_to_save[:, 0]
        waypoints_y_to_save = waypoints_to_save[:, 1]

        # Initialise
        next_waypoints_dict = dict()
        keys_next_x_waypoints_rel = ['WYPT_REL_X_' + str(i).zfill(2) for i in range(len(waypoints_x_to_save))]
        keys_next_y_waypoints_rel = ['WYPT_REL_Y_' + str(i).zfill(2) for i in range(len(waypoints_y_to_save))]

        next_waypoints_dict = dict(zip(keys_next_x_waypoints_rel, waypoints_x_to_save))
        next_waypoints_dict.update(zip(keys_next_y_waypoints_rel, waypoints_y_to_save))
        
        return next_waypoints_dict
    
   

    '''
    Generate plots from the csv file of the recording
    '''
    def plot_data(self):
        
        save_path = self.csv_filepath[:-4]+"_data"
        os.mkdir(save_path)
        df = pd.read_csv(self.csv_filepath, header = 0, skiprows=range(0,8))

        times = df['time'].to_numpy()[1:]
        pos_xs = df['pose_x'].to_numpy()[1:]
        pos_ys = df['pose_y'].to_numpy()[1:]
        angular_controls = df['angular_control_calculated'].to_numpy()[1:]
        translational_controls = df['translational_control_calculated'].to_numpy()[1:]   
        
        # Plot position
        plt.clf()
        plt.title("Position")
        plt.plot(pos_xs, pos_ys, color="green")
        plt.savefig(save_path+"/position.png")
        plt.clf()
        
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
        
        # Plot time needed for calculation
        dts = []
        for i in range(15, len(times)-1):
            
            last_row = times[i-1]
            row = times[i]
            delta =float(row) - float(last_row) 
            dts.append(delta)


        plt.plot(dts)
        plt.savefig(save_path+"/dt_control.png")
        
        
        # Copy Settings and configs
        config_sage_path = os.path.join(save_path, "configs")
        os.mkdir(config_sage_path)
        shutil.copy("Control_Toolkit_ASF/config_controllers.yml", config_sage_path) 
        shutil.copy("Control_Toolkit_ASF/config_cost_function.yml", config_sage_path) 
        shutil.copy("Control_Toolkit_ASF/config_optimizers.yml", config_sage_path) 
        shutil.copy("utilities/Settings.py", config_sage_path) 
        
        
        shutil.copy("utilities/Settings.py", config_sage_path) 
        
        with open(config_sage_path + '/Settings_applied.json', 'w') as f:
            f.write(str(Settings.__dict__))
    
        shutil.copy(os.path.join(Settings.MAP_PATH, Settings.MAP_NAME+".png"), config_sage_path) 
        shutil.copy(os.path.join(Settings.MAP_PATH, "speed_scaling.yaml"), config_sage_path) 

        
        
    def reset(self):
        self.dict_buffer=[]
        self.dict_to_save = {}

