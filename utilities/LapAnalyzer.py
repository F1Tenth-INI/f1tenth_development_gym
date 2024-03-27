import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import yaml
import json

import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.Settings import Settings

class LapAnalyzer():
    
    def __init__(self, save_path, experiment_dataframe, map_path, map_name):
        self.save_path = save_path
        self.experiment_dataframe = experiment_dataframe
        self.map_path = map_path
        self.map_name = map_name
        
    def plot_wp(self):
        
        # Load the data
        car_data = self.experiment_dataframe
        
        wp_file = os.path.join(self.map_path, self.map_name + '_wp.csv')    
        wp_data = pd.read_csv(wp_file)
            
        car_x = car_data['pose_x'].to_numpy()
        car_y = car_data['pose_y'].to_numpy()
        
        wp_x = wp_data['x_m'].to_numpy()
        wp_y = wp_data['y_m'].to_numpy()


        img_path = os.path.join(self.map_path, self.map_name)
        with open(img_path + '.yaml', 'r') as file:
            map_data = yaml.safe_load(file)

        # Load the background image
        # print("Image path:", img_path+ '.png')
        img = plt.imread(img_path + '.png')
        # img = plt.imread('./utilities/maps/RCA1/RCA1_wp_min_curve_og.png')
        # Determine the limits based on the origin and resolution
        x_min = map_data['origin'][0]
        y_min = map_data['origin'][1]
        x_max = x_min + img.shape[1] * map_data['resolution']
        y_max = y_min + img.shape[0] * map_data['resolution']

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))

        # Show the image
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')

        # Mark the starting point with a cross
        plt.scatter(wp_x[0], wp_y[0], color='green', marker='o', s=200, label='Starting point waypoints')
        plt.scatter(car_x[0], car_y[0], color='purple', marker='o', s=200, label='Starting point sim')
        
        # Mark the end point with a cross
        plt.scatter(car_x[-1], car_y[-1], color='purple', marker='x', s=200, label='End point sim')
        
        plt.plot(wp_x, wp_y, color='green', label='waypoints')
        # plt.plot(car_x, car_y, color='purple', linestyle='dashdot', label='sim')
    
        # Create a color map
        cmap1 = plt.get_cmap('plasma')
        cmap2 = plt.get_cmap('viridis')
        
        # Calculate the number of segments
        num_segments1 = len(car_x) - 1
        
        # Draw each segment individually
        for i in range(num_segments1):
            # Calculate the color value for this segment
            color = cmap1(i / num_segments1)

            # Draw the segment
            ax.plot(car_x[i:i+2], car_y[i:i+2], color=color, linestyle='dashdot')
            
        # Show the color bar
        sm = plt.cm.ScalarMappable(cmap=cmap1, norm=plt.Normalize(vmin=0, vmax=num_segments1))
        cbar1 = fig.colorbar(sm, ax=ax)
        cbar1.set_label('sim')
            
    
        
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title(map_name + ' Coordinate System')
        title = "NNI on FPGA"
        # Settings.CONTROLLER
    
        plt.title(title)
        plt.legend(loc='upper right')

        
        # Save the plot as a PNG
        plt.savefig(os.path.join(self.save_path, 'result.png'))
        

    def get_position_error(self):
        # Load the data
        car_data = self.experiment_dataframe
        
        wp_file = os.path.join(self.map_path, self.map_name + '_wp.csv')    
        wp_data = pd.read_csv(wp_file)
        wp_data = wp_data[::4]
        
        def point_to_line_segment_dist(x1, y1, x2, y2, x0, y0):
            """Calculate the distance from a point to a line segment."""
            # Calculate the squared distance between the endpoints of the line segment
            l2 = (x2 - x1)**2 + (y2 - y1)**2
            if l2 == 0.0:
                # The line segment is actually a point
                return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

            # Calculate the projection of the point onto the line segment
            t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2))

            # Calculate the coordinates of the projection
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)

            # Calculate the distance between the point and the projection
            return np.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)

        
        car_x = car_data['pose_x'].to_numpy()
        car_y = car_data['pose_y'].to_numpy()
        wp_x = wp_data['x_m'].to_numpy()
        wp_y = wp_data['y_m'].to_numpy()
        
        
        distances = []
        for i in range(len(car_x)):
            min_dist = np.inf
            for j in range(len(wp_x) - 1):
                # dist = point_to_line_dist(wp_x[j], wp_y[j], wp_x[j+1], wp_y[j+1], car_x[i], car_y[i])
                dist = point_to_line_segment_dist(wp_x[j], wp_y[j], wp_x[j+1], wp_y[j+1], car_x[i], car_y[i])
                
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)
            
        distances_stats = {
            "Min": np.min(distances).tolist(),
            "Max": np.max(distances).tolist(),
            "Mean": np.mean(distances).tolist(),
            "Std": np.std(distances).tolist(),
        }

        with open(f"{self.save_path}/accuracy_info.json", 'w') as f:
            json.dump(distances_stats, f,indent=4)
        
        plt.plot(distances)
        plt.savefig(f"{self.save_path}/accuracy.png")
        plt.clf()
        plt.hist(distances, bins=100, edgecolor='blue', histtype='step')
        plt.title('Histogram of Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.savefig(f"{self.save_path}/distances_histogram.png")
        plt.clf()
                
        
    def get_laptime_info(self):
        

        car_data = self.experiment_dataframe
        starting_position = car_data[['pose_x', 'pose_y']].iloc[0]
        
        # Calculate the Euclidean distance from each position to the starting position
        distances = np.sqrt((car_data['pose_x'] - starting_position[0])**2 + (car_data['pose_y'] - starting_position[1])**2)
        times_close_to_starting_point = car_data.loc[distances < 1.0]['time']
        
        time_differences = times_close_to_starting_point.diff() 
        time_differences = time_differences.fillna(0)
        time_differences[0] = 10000


        # Keep only the elements where the difference is greater than 5
        lap_complete_times = times_close_to_starting_point[time_differences.abs() > 5]
        indices_close_to_starting_point = times_close_to_starting_point.index
        
                
        lap_complete_times = lap_complete_times.dropna()
        
        lap_times = lap_complete_times.diff()
        lap_times = lap_times.dropna()

        os.makedirs(self.save_path, exist_ok=True)
        np.savetxt(f"{self.save_path}/laptimes2.csv", lap_times.to_numpy(), delimiter=",", header="laptimes", fmt='%.2f')        
        return lap_times.to_numpy()
      
            
if __name__ == "__main__":
    

    # Load the data
    map_name = 'RCA2'
    exper_folder_path = 'AnalyseData/iros24/data/physical'
    # experiment_recording_name = 'F1TENTH__2024-02-19_18-27-26Recording1_RCA2_neural_50Hz_vel1.0_noise_c[0.0, 0.0]'
    experiment_recording_name = 'F1TENTH__2024-02-19_18-35-58Recording1_RCA2_pp_50Hz_vel1.0_noise_c[0.0, 0.0]'
    map_path = "AnalyseData/iros24/data/physical/" + experiment_recording_name +"_data/configs"


    # experiment_recording_name = 'F1TENTH__2024-03-12_09-33-26Recording1_RCA2_neural_50Hz_vel1.1_noise_c[0.0, 0.0]'
    # experiment_recording_name = 'F1TENTH__2024-03-12_09-56-29Recording1_RCA2_mpc_50Hz_vel1.2_noise_c[0.0, 0.0]'
    # experiment_recording_name = 'F1TENTH__2024-03-13_14-21-33Recording1_RCA2_pp_50Hz_vel0.8_noise_c[0.0, 0.0]'
    # exper_folder_path = 'AnalyseData/iros24/data/sim'
    # map_path = "AnalyseData/iros24/data/sim/" + experiment_recording_name +"_data/configs"
    
    
    
    
    experiment_recording_filename = experiment_recording_name + '.csv'
    experiment_recording_path = os.path.join(exper_folder_path, experiment_recording_filename)
    
    experiment_df = pd.read_csv(experiment_recording_path, comment='#',skipinitialspace=True)
    
    # experiment_df = experiment_df.iloc[:-3000] # NNI experiment
    # experiment_df = experiment_df.iloc[0 :-1000] # Pure pursuit experiment



    lap_analyzer = LapAnalyzer(map_path, experiment_df, map_path, map_name)

    
    lap_analyzer.get_laptime_info()
    lap_analyzer.get_position_error()
    lap_analyzer.plot_wp()