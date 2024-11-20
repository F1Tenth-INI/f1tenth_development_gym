import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import yaml

def plot_wp(map_path, map_name,experiment_recording):
    
    # Load the data
    car_data = experiment_recording
    
    wp_file = os.path.join(map_path, map_name + '_wp.csv')    
    wp_data = pd.read_csv(wp_file, comment='#')
        
    car_x = car_data['pose_x'].to_numpy()
    car_y = car_data['pose_y'].to_numpy()
    
    wp_x = wp_data['x_m'].to_numpy()
    wp_y = wp_data['y_m'].to_numpy()


    img_path = os.path.join(map_path, map_name)
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
    plt.title('Pure Pursuit')
    plt.legend(loc='upper right')

    
    # Save the plot as a PNG
    plt.savefig('result.png')
    plt.show()
    

def get_position_error(map_path, map_name, experiment_recording):
    # Load the data
    car_data = experiment_recording
    
    wp_file = os.path.join(map_path, map_name + '_wp.csv')    
    wp_data = pd.read_csv(wp_file, comment='#')
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
        
    print("distances to raceline")
    print("Min:", np.min(distances))
    print("Max:", np.max(distances))
    print("Mean:", np.mean(distances))
    print("Std:", np.std(distances))
    plt.plot(distances)
    plt.show()
    
def get_laptime_info(experiment_recording):
    

    car_data = experiment_recording
    starting_position = car_data[['pose_x', 'pose_y']].iloc[0]
    
    # Calculate the Euclidean distance from each position to the starting position
    distances = np.sqrt((car_data['pose_x'] - starting_position[0])**2 + (car_data['pose_y'] - starting_position[1])**2)
    times_close_to_starting_point = car_data.loc[distances < 2.0]['time']
    
    time_differences = times_close_to_starting_point.diff() 
    time_differences = time_differences.fillna(0)
    time_differences[0] = 10000


    # Keep only the elements where the difference is greater than 5
    lap_complete_times = times_close_to_starting_point[time_differences.abs() > 5]
    lap_complete_times = lap_complete_times.dropna()
    
    lap_times = lap_complete_times.diff()
    lap_times = lap_times.dropna()
    print("Lap Times: ")
    for time in lap_times.to_numpy():
        print(time)
        
if __name__ == "__main__":
    

    # Load the data
    map_name = 'RCA2'
    # map_path = 'utilities/maps/' + map_name  # if no img, set to None
    # exper_folder_path = 'AnalyseData/iros_paper/sim/02_16/'


    exper_folder_path = 'ExperimentRecordings'

    experiment_recording_name = 'F1TENTH__2024-02-19_13-04-41Recording1_RCA2_neural_50Hz_vel1.0_noise_c[0.0, 0.0]'

    
    
    # Load the data
    experiment_recording_filename = experiment_recording_name + '.csv'
    experiment_recording_path = os.path.join(exper_folder_path, experiment_recording_filename)
    experiment_data_path = os.path.join(exper_folder_path, experiment_recording_name)    
    experiment_recording = pd.read_csv(experiment_recording_path, comment='#',skipinitialspace=True)
    # experiment_recording = experiment_recording[:-1000]

    
    map_path = experiment_data_path+"_data"

    map_path = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/ExperimentRecordings/" + experiment_recording_name +"_data/configs"
    
    get_laptime_info(experiment_recording)
    get_position_error(map_path, map_name, experiment_recording)
    plot_wp(map_path, map_name, experiment_recording)