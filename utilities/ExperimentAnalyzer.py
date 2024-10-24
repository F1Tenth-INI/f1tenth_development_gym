import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString, Point
import os
from typing import Dict

import json

from utilities.Settings import Settings


class ExperimentAnalyzer:
    def __init__(self, experiment_name, experiment_path = Settings.RECORDING_FOLDER, step_start=0, step_end= Settings.EXPERIMENT_LENGTH):
        
        self.step_start = 0
        self.step_end = step_end
        
        self.experiment_path = experiment_path
        self.experiment_name = experiment_name
        self.map_name = Settings.MAP_NAME
        self.controller_name = 'neural'
        self.analyse_folder = 'ExperimentRecordings/Analyse'
        
        csv_path = os.path.join(self.experiment_path, self.experiment_name) 
        self.experiment_data_path = os.path.join(csv_path + "_data")
        self.experiment_configs_path = os.path.join(self.experiment_data_path, "configs")
        
        self.waypoints_file = os.path.join(self.experiment_configs_path, self.map_name + "_wp")
        
        # Waypoints from 
        self.waypoints: pd.DataFrame = pd.read_csv(self.waypoints_file + ".csv")
        self.waypoints.columns = self.waypoints.columns.str.replace(' ', '') # Remove spaces from column names
        
        # Recording from csv file (cut alreay)
        self.recording: pd.DataFrame = pd.read_csv(csv_path + ".csv", skiprows=8)
        self.recording = self.recording.iloc[step_start:step_end]
        
        self.position_errors = self.get_position_error()
        

    def get_position_error(self) -> np.ndarray:
        optimal_x = self.waypoints['x_m'].values
        optimal_y = self.waypoints['y_m'].values

        optimal = list(zip(optimal_x, optimal_y))
        optimal_line = LineString(optimal).coords

        recorded_x = self.recording['pose_x'].values
        recorded_y = self.recording['pose_y'].values

        recorded = list(zip(recorded_x, recorded_y))
        recorded_line = LineString(recorded)

        errors = []

        for i in range(len(recorded_x)):
            errors.append(recorded_line.distance(Point(optimal_line[i%len(optimal_x)])))

        errors = np.array(errors)
        
        return errors

    

    def get_error_stats(self) -> Dict[str, float]:
        
        errors = self.position_errors
        error_stats: Dict[str, float] = {
            'max': float(np.max(errors)),
            'min': float(np.min(errors)),
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'var': float(np.var(errors))
        }
        
        return error_stats
    
    def plot_experiment(self):
        self.plot_errors()
        self.plot_states()
        self.plot_controls()
        self.save_error_stats()
        
        
    def plot_controls(self):
        # Plot States
        state_names = ['angular_vel_z','linear_vel_x','pose_theta','pose_theta_cos','pose_theta_sin','pose_x','pose_y',]
        
        # Create a new figure
        fig = plt.figure(figsize=(15, 20))  # width: 15 inches, height: 20 inches

        for index, state_name in enumerate(state_names):
            # Add subplot for each state
            plt.subplot(7, 1, index+1)  # 7 rows, 1 column, nth plot
            plt.title(state_name)
            plt.plot(self.recording[state_name].to_numpy()[1:], color="red")

    
        plt.savefig(os.path.join(self.experiment_data_path, "state_plots.png" ))
        plt.clf()
        
    def plot_states(self):
        # Plot Control
        
        angular_controls = self.recording['angular_control_calculated'].to_numpy()[1:]
        translational_controls = self.recording['translational_control_calculated'].to_numpy()[1:]   
        
        fig = plt.figure()
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        plt.title("Angular Control")
        plt.plot(angular_controls, color="red")
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
        plt.title("Translational Control")
        plt.plot(translational_controls, color="blue")
        plt.savefig(os.path.join(self.experiment_data_path, "control_plots.png" ))
        plt.clf()
              
    def plot_errors(self):
        
        time_recorded = self.recording['time'].values
        
        controller_name = self.controller_name

        optimal_x = self.waypoints['x_m'].values
        optimal_y = self.waypoints['y_m'].values
        
        recorded_x = self.recording['pose_x'].values
        recorded_y = self.recording['pose_y'].values
        
        plt.clf()
        plt.figure()
        plt.plot(optimal_x, optimal_y, label='Raceline')
        plt.plot(recorded_x, recorded_y, label='Recorded Line')
        plt.legend()

        plt.xlabel('X-Position')
        plt.ylabel('Y-Position')
        # plt.title('Comparison between '+ controller_name +' raceline and recorded line waypoints on '+ map_name)
        plt.savefig(os.path.join(self.experiment_data_path, "position_error_birdview.png" ))


        plt.figure()
        plt.plot(time_recorded[self.step_start:self.step_end], self.position_errors[self.step_start:self.step_end],color='cyan', label='Position error with '+controller_name+' and waypoints')

        plt.xlabel('Time [s]', fontsize=24)
        plt.ylabel('Error [m]', fontsize=24)
        plt.title('Position error with Recording of '+controller_name+' and waypoints on '+ self.map_name, fontsize=24)
        plt.tick_params(axis='both', labelsize=24)
        plt.legend(loc='upper right', fontsize=24)
        plt.grid()
        
        plt.savefig(os.path.join(self.experiment_data_path, "position_error_distance.png" ))
        
        # Boxplot
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_ylabel('Error [m]')
        ax.set_title('Position error with Recording of '+controller_name+' and waypoints on '+ self.map_name)
        ax.boxplot(self.position_errors)    
        plt.savefig(os.path.join(self.experiment_data_path, "position_error_boxplot.png" ))


        # Plot estimated mu along track        
        try:
            
            mus = self.recording['mu'].values
            mus_predicted = self.recording['mu_predicted'].values
            mu_error = mus_predicted - mus
            max_abs_mu_error = np.max(np.abs(mu_error[10:]))

        
            plt.clf()   
            plt.figure(figsize=(8, 6), dpi=150)
            # Cut away first 10 values because they are not accurate and mess up color scale
            sc = plt.scatter(recorded_x[10:], recorded_y[10:], c=mu_error[10:], cmap='seismic', vmin=-max_abs_mu_error, vmax=max_abs_mu_error, label="Position")
            plt.colorbar(sc, label='Mu error')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Car Position with Mu Predicted ({Settings.SURFACE_FRICITON})')
            plt.legend()
            plt.savefig(os.path.join(self.experiment_data_path, "mu_predicted.png" ))
        except Exception as e:
            print(f"Warning: No mu values found in recording: {e}")


    
    def save_error_stats(self):
        error_stats = self.get_error_stats()
        file = os.path.join(self.experiment_data_path, "error_stats.json") 
        with open(file, 'w') as json_file:
            json.dump(error_stats, json_file, indent=4)
    
     

# Test function
if __name__ == "__main__":

    ea = ExperimentAnalyzer("F1TENTH__2024-09-25_10-20-05Recording1_RCA2_nni-lite_50Hz_vel_1.2_noise_c[0.0, 0.0]_mu_0.5") 
    ea.plot_experiment()