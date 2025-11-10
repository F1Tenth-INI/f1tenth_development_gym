import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString, Point
import os
from typing import Dict

import json

from utilities.Settings import Settings


class ExperimentAnalyzer:
    def __init__(self, experiment_name, experiment_path = Settings.RECORDING_FOLDER, step_start=0, step_end= Settings.SIMULATION_LENGTH):
        
        self.step_start = 0
        self.step_end = step_end
        
        self.experiment_path = experiment_path
        self.experiment_name = experiment_name
        self.map_name = Settings.MAP_NAME
        self.map_path = Settings.MAP_PATH
        self.controller_name = 'neural'
        
        csv_path = os.path.join(self.experiment_path, self.experiment_name) 
        self.experiment_data_path = os.path.join(csv_path + "_data")
        self.experiment_configs_path = os.path.join(self.experiment_data_path, "configs")
        
        self.waypoints_file = os.path.join(self.map_path, self.map_name + "_wp")
        # self.waypoints_file = os.path.join(self.experiment_configs_path, self.map_name + "_wp")
        
        # Waypoints from 
        self.waypoints: pd.DataFrame = pd.read_csv(self.waypoints_file + ".csv", comment='#')   
        self.waypoints.columns = self.waypoints.columns.str.replace(' ', '') # Remove spaces from column names
        
        # Recording from csv file (cut alreay)
        self.recording: pd.DataFrame = pd.read_csv(csv_path + ".csv", comment='#')
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
        self.plot_imu_data()
        self.save_error_stats()
        
        
    def plot_controls(self):
        # Plot States
        state_names = ['angular_vel_z','linear_vel_x','linear_vel_y','pose_theta','pose_theta_cos','pose_theta_sin','pose_x','pose_y',]
        
        # Create a new figure
        fig = plt.figure(figsize=(15, 20))  # width: 15 inches, height: 20 inches

        for index, state_name in enumerate(state_names):
            # Add subplot for each state
            plt.subplot(len(state_names), 1, index+1)  # 7 rows, 1 column, nth plot
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
        
        # Check of plot folder exists
        if not os.path.exists(self.experiment_data_path):
            os.makedirs(self.experiment_data_path)
            
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
            plt.title(f'Car Position with Mu Predicted ({Settings.SURFACE_FRICTION})')
            plt.legend()
            plt.savefig(os.path.join(self.experiment_data_path, "mu_predicted.png" ))
        except Exception as e:
            print(f"Warning: No mu values found in recording: {e}")

    def plot_imu_data(self):
        """Plot IMU data including accelerometer, gyroscope, and orientation data."""
        try:
            # Check if IMU data columns exist
            imu_columns = [col for col in self.recording.columns if col.startswith('imu_')]
            if not imu_columns:
                print("Warning: No IMU data found in recording")
                return
            
            # Get time data
            time_data = self.recording['time'].to_numpy()[1:]
            
            # Create figure with subplots for different IMU data types
            fig, axes = plt.subplots(4, 1, figsize=(15, 20))
            fig.suptitle('IMU Sensor Data', fontsize=16)
            
            # Plot 1: Accelerometer data
            ax1 = axes[0]
            accel_cols = [col for col in imu_columns if 'a_' in col and not 'quat' in col]
            for col in accel_cols:
                data = self.recording[col].to_numpy()[1:]
                # Remove gravity from Z-axis accelerometer
                if col == 'imu_a_z':
                    data = data - 9.81  # Remove gravity component
                ax1.plot(time_data, data, label=col.replace('imu_', ''))
            ax1.set_title('Accelerometer Data (m/s²) - Gravity Removed from Z-axis')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Gyroscope data
            ax2 = axes[1]
            gyro_cols = [col for col in imu_columns if 'gyro_' in col]
            for col in gyro_cols:
                ax2.plot(time_data, self.recording[col].to_numpy()[1:], label=col.replace('imu_', ''))
            ax2.set_title('Gyroscope Data (rad/s)')
            ax2.set_ylabel('Angular Velocity (rad/s)')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Euler angles
            ax3 = axes[2]
            euler_cols = [col for col in imu_columns if col in ['imu_roll', 'imu_pitch', 'imu_yaw']]
            for col in euler_cols:
                ax3.plot(time_data, self.recording[col].to_numpy()[1:], label=col.replace('imu_', ''))
            ax3.set_title('Euler Angles (rad)')
            ax3.set_ylabel('Angle (rad)')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Quaternion data
            ax4 = axes[3]
            quat_cols = [col for col in imu_columns if 'quat_' in col]
            for col in quat_cols:
                ax4.plot(time_data, self.recording[col].to_numpy()[1:], label=col.replace('imu_', ''))
            ax4.set_title('Quaternion Data')
            ax4.set_ylabel('Quaternion Component')
            ax4.set_xlabel('Time (s)')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_data_path, "imu_data.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create additional detailed plots for accelerometer and gyroscope
            self._plot_imu_detailed()
            
        except Exception as e:
            print(f"Warning: Error plotting IMU data: {e}")
    
    def _plot_imu_detailed(self):
        """Create detailed IMU plots with magnitude and individual components."""
        try:
            time_data = self.recording['time'].to_numpy()[1:]
            
            # Detailed accelerometer plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Detailed IMU Analysis', fontsize=16)
            
            # Accelerometer magnitude (with gravity removed from Z-axis)
            ax1 = axes[0, 0]
            accel_x = self.recording['imu_a_x'].to_numpy()[1:]
            accel_y = self.recording['imu_a_y'].to_numpy()[1:]
            accel_z = self.recording['imu_a_z'].to_numpy()[1:] - 9.81  # Remove gravity
            accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            ax1.plot(time_data, accel_magnitude, 'k-', linewidth=2, label='Magnitude (Z-gravity removed)')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero reference')
            ax1.set_title('Accelerometer Magnitude (Gravity Removed from Z-axis)')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.legend()
            ax1.grid(True)
            
            # Gyroscope magnitude
            ax2 = axes[0, 1]
            gyro_x = self.recording['imu_gyro_x'].to_numpy()[1:]
            gyro_y = self.recording['imu_gyro_y'].to_numpy()[1:]
            gyro_z = self.recording['imu_gyro_z'].to_numpy()[1:]
            gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            ax2.plot(time_data, gyro_magnitude, 'k-', linewidth=2, label='Magnitude')
            ax2.set_title('Gyroscope Magnitude')
            ax2.set_ylabel('Angular Velocity (rad/s)')
            ax2.legend()
            ax2.grid(True)
            
            # Quaternion magnitude (should be close to 1)
            ax3 = axes[1, 0]
            quat_w = self.recording['imu_quat_w'].to_numpy()[1:]
            quat_x = self.recording['imu_quat_x'].to_numpy()[1:]
            quat_y = self.recording['imu_quat_y'].to_numpy()[1:]
            quat_z = self.recording['imu_quat_z'].to_numpy()[1:]
            quat_magnitude = np.sqrt(quat_w**2 + quat_x**2 + quat_y**2 + quat_z**2)
            ax3.plot(time_data, quat_magnitude, 'k-', linewidth=2, label='Magnitude')
            ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Normalized (1.0)')
            ax3.set_title('Quaternion Magnitude')
            ax3.set_ylabel('Magnitude')
            ax3.set_xlabel('Time (s)')
            ax3.legend()
            ax3.grid(True)
            
            # Yaw angle over time
            ax4 = axes[1, 1]
            yaw_angle = self.recording['imu_yaw'].to_numpy()[1:]
            ax4.plot(time_data, yaw_angle, 'b-', linewidth=2, label='Yaw Angle')
            ax4.set_title('Yaw Angle Over Time')
            ax4.set_ylabel('Yaw Angle (rad)')
            ax4.set_xlabel('Time (s)')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_data_path, "imu_detailed_analysis.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Error creating detailed IMU plots: {e}")

    
    def save_error_stats(self):
        error_stats = self.get_error_stats()
        file = os.path.join(self.experiment_data_path, "error_stats.json") 
        with open(file, 'w') as json_file:
            json.dump(error_stats, json_file, indent=4)
    
     

# Test function
if __name__ == "__main__":
    # experiment_dir = "TrainingLite/Datasets/Custom_IPZ34b/"
    experiment_dir = "ExperimentRecordings/"
    experiment_name = "2025-03-31_12-34-45_Recording1_0_RCA1_neural_50Hz_vel_0.8_noise_c[0.0, 0.0]_mu_0.5_mu_c_None_"
    
    experiment_path = os.path.join(experiment_dir, experiment_name)
    
    ea = ExperimentAnalyzer(experiment_name=experiment_name, experiment_path=experiment_dir) 
    ea.plot_experiment()