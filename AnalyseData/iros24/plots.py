import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import yaml
from matplotlib.colors import LinearSegmentedColormap

from scipy.ndimage import rotate

# NN
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = current_dir + '/plots'

def plot_experiment(experiment_name, df, title):
    
    map_name = "RCA2"
    experiment_path = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/AnalyseData/iros24/data/" + experiment_name
    
    
    # Load the data
    car_data = df
    
    map_path = os.path.join(experiment_path+"_data", "configs")
    
    wp_file = os.path.join(experiment_path+"_data","configs",map_name + '_wp.csv') 
    print(wp_file)   
    wp_data = pd.read_csv(wp_file)
        
    car_x = car_data['pose_x'].to_numpy()
    car_y = car_data['pose_y'].to_numpy()
    car_vx = car_data['linear_vel_x'].to_numpy()
    
    wp_x = wp_data['x_m'].to_numpy()
    wp_y = wp_data['y_m'].to_numpy()


    img_path = os.path.join(map_path, map_name)
    with open(img_path + '.yaml', 'r') as file:
        map_data = yaml.safe_load(file)

    # Load the background image
    # print("Image path:", img_path+ '.png')

    img = plt.imread(img_path + '.png')
    # Rotate the image by 10 degrees counterclockwise
    img_rotated = rotate(img, 40, reshape=True)

    x_min = map_data['origin'][0]
    y_min = map_data['origin'][1]
    x_max = x_min + img_rotated.shape[1] * map_data['resolution']
    y_max = y_min + img_rotated.shape[0] * map_data['resolution']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Show the rotated image
    ax.imshow(img_rotated, extent=[x_min, x_max, y_min, y_max], cmap='gray')

    # Rest of the code...
    x_min = map_data['origin'][0]
    y_min = map_data['origin'][1]
    x_max = x_min + img.shape[1] * map_data['resolution']
    y_max = y_min + img.shape[0] * map_data['resolution']

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Show the image
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')
    
    plt.plot(wp_x, wp_y, color='blue', label='waypoints', linewidth=3)
    # plt.plot(car_x, car_y, color='purple', linestyle='dashdot', label='sim')

    # Create a color map
    # cmap1 = plt.get_cmap('plasma')
    cmap1 = LinearSegmentedColormap.from_list(
        name='green_to_red', 
        colors=['green', 'red']
    )
    
    # Calculate the number of segments
    num_segments1 = len(car_x) - 1
    
    # Assuming car_vx is a list of speeds
    car_vx = car_data['linear_vel_x'].to_numpy()

    # Normalize car_vx for color mapping
    # norm = plt.Normalize(vmin=car_vx.min(), vmax=car_vx.max())
    norm = plt.Normalize(vmin=2.0, vmax=12)

    # Draw each segment individually
    for i in range(num_segments1):
        # Calculate the color value for this segment
        color = cmap1(norm(car_vx[i]))
        # Draw the segment
        ax.plot(car_x[i:i+2], car_y[i:i+2], color=color, linestyle='dashdot')

    # Show the color bar
    sm = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
    cbar1 = fig.colorbar(sm, ax=ax)
    cbar1.set_label('Speed [m/s]')

    
    # plt.grid(True)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # Settings.CONTROLLER
    
     
    
    # Draw a line representing 10 meters
    # Choose starting point for the line (x_start, y_start)
    x_start = -5
    y_start = -5
    x_end = x_start + 5
    y_end = y_start
    ax.plot([x_start, x_end], [y_start, y_end], linestyle='-', linewidth=2, color='black')
    ax.text((x_start + x_end)/2, y_start, '5m', ha='center', va='bottom', color='black')

    plt.title(title)
    # plt.legend(loc='upper right')
    
    _, lap_times = get_laptimes(df)
    lap_times_mean = np.mean(lap_times)
    lap_times_mean = round(lap_times_mean, 3)
    
    
    distances = get_position_errors(df, wp_file)
    diestances_mean = round(np.mean(distances), 3)
    distances_std_dev = round(np.std(distances), 3)
    distances_max = round(np.max(distances), 3)
    
    ax.annotate(f"Laptime mean: {lap_times_mean}, distance mean: {diestances_mean}, std_dev: {distances_std_dev}, max: {distances_max}",
                xy=(0.5, 0), xytext=(0.5, -0.1), xycoords='axes fraction', textcoords='axes fraction', ha='center', va='top', color='black')

    
    # Save the plot as a PNG
    plt.axis('off')
    plt.savefig(os.path.join(title + '.png'))
    plt.savefig(f"{save_path}/{title}.png")

    
    # plt.show()


def plot_map_with_waypoints(map_name):
    
    map_path = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/utilities/maps/" + map_name
    wp_file = os.path.join(map_path ,map_name + '_wp.csv') 
    print(wp_file)   
    wp_data = pd.read_csv(wp_file)
        
    wp_x = wp_data['x_m'].to_numpy()
    wp_y = wp_data['y_m'].to_numpy()


    img_path = os.path.join(map_path, map_name)
    with open(img_path + '.yaml', 'r') as file:
        map_data = yaml.safe_load(file)

    # Load the background image
    # print("Image path:", img_path+ '.png')

    img = plt.imread(img_path + '.png')
    # Rotate the image by 10 degrees counterclockwise
    img_rotated = rotate(img, 0, reshape=True)

    x_min = map_data['origin'][0]
    y_min = map_data['origin'][1]
    x_max = x_min + img_rotated.shape[1] * map_data['resolution']
    y_max = y_min + img_rotated.shape[0] * map_data['resolution']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Show the rotated image
    ax.imshow(img_rotated, extent=[x_min, x_max, y_min, y_max], cmap='gray')


    plt.plot(wp_x, wp_y, color='blue', label='waypoints', linewidth=4)
    # plt.plot(car_x, car_y, color='purple', linestyle='dashdot', label='sim')

   
    
    # Draw a line representing 10 meters
    # Choose starting point for the line (x_start, y_start)
    x_start = -5
    y_start = -5
    x_end = x_start + 5
    y_end = y_start
    ax.plot([x_start, x_end], [y_start, y_end], linestyle='-', linewidth=2, color='black')
    ax.text((x_start + x_end)/2, y_start, '5m', ha='center', va='bottom', color='black')


    plt.title(map_name)
    # plt.legend(loc='upper right')

    
    # Save the plot as a PNG
    plt.axis('off')
    plt.savefig(f"{save_path}/{map_name}_wp.png")

    
    # plt.show()

def plot_progress(df1, df2):
    
    df1 = df1.iloc[500:1250]
    df2 = df2.iloc[500:1250]
    
    fig, ax = plt.subplots()
    
    def calculate_dist_integral(df):
        # Calculate the differences between consecutive points
        df['diff_x'] = df['pose_x'].diff()
        df['diff_y'] = df['pose_y'].diff()

        # Calculate the distances between consecutive points
        df['distance'] = np.sqrt(df['diff_x']**2 + df['diff_y']**2)

        # Calculate the cumulative sum of the distances
        df['cumulative_distance'] = df['distance'].cumsum()

        return df['cumulative_distance']

    # Calculate the cumulative sum of the distances
    df1['cumulative_distance'] =  calculate_dist_integral(df1)
    df2['cumulative_distance'] =  calculate_dist_integral(df2)

    # Plot the cumulative distance
    plt.plot(df1['cumulative_distance'], label='NNI')
    plt.plot(df2['cumulative_distance'], label='PP')
    plt.title('Integral over the travelled distance')
    plt.xlabel('Time step')
    plt.ylabel('Cumulative distance')
    plt.show()

    # Show the legend
    # plt.legend()

    # Show the plot
    plt.show()    
    
def get_laptimes(experiment_df):
    

    car_data = experiment_df
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
    lap_indices = lap_times.index.tolist()
    lap_times = lap_times.tolist()
    
    return lap_indices, lap_times

def split_experiment_into_laps(experiment_df):
    lap_indices, lap_times = get_laptimes(experiment_df)
    
    lap_indices.insert(0, 0)
    laps = [experiment_df.iloc[lap_indices[i]:lap_indices[i+1]] for i in range(len(lap_indices)-1)]

    return laps

def get_position_errors(df, wp_path):
        # Load the data
        car_data = df
        
        wp_file = os.path.join(wp_path)    
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

        return np.array(distances)
       
def get_distance_stats(distances):
    distances_stats = {
        "min": np.min(distances),
        "max": np.max(distances),
        "mean": np.mean(distances),
        "std_dev": np.std(distances),
    }
    return distances_stats

map_name = "RCA2"
wp_file = os.path.join("/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/utilities/maps/RCA2", "RCA2_wp.csv")


lap_first = 2
lap_last = 11
number_of_laps = 9


# PLOT EXPERIMENTS FOR SIM CAR
title = "Sim NC"
experiment_name = "sim/F1TENTH__2024-03-12_09-33-26Recording1_RCA2_neural_50Hz_vel1.1_noise_c[0.0, 0.0]"
path= current_dir+'/data/' + experiment_name +'.csv'
df_nn = pd.read_csv(path, comment='#')
dfs_nn = split_experiment_into_laps(df_nn) # Select the first 8 laps
df_nn = pd.concat(dfs_nn[:number_of_laps])
# plot_experiment(experiment_name, df_nn, title)


title = "Sim MPC"
experiment_name = "sim/F1TENTH__2024-03-12_09-56-29Recording1_RCA2_mpc_50Hz_vel1.2_noise_c[0.0, 0.0]"
path= current_dir+'/data/' + experiment_name +'.csv'
df_mpc = pd.read_csv(path, comment='#')
dfs_mpc = split_experiment_into_laps(df_mpc) # Select the first 8 laps
df_mpc = pd.concat(dfs_mpc[:number_of_laps])
# plot_experiment(experiment_name, df_mpc, title)


title = "Sim PP"
experiment_name = "sim/F1TENTH__2024-03-13_14-21-33Recording1_RCA2_pp_50Hz_vel0.8_noise_c[0.0, 0.0]"
path= current_dir+'/data/' + experiment_name +'.csv'
df_pp = pd.read_csv(path, comment='#')
dfs_pp = split_experiment_into_laps(df_pp) # Select the first 8 laps
df_pp = pd.concat(dfs_pp[:number_of_laps])
# plot_experiment(experiment_name, df_pp, title)


# PLOT EXPERIMENTS FOR PHYSICAL CAR
title = "Physical NC"
experiment_name = "physical/F1TENTH__2024-02-19_18-27-26Recording1_RCA2_neural_50Hz_vel1.0_noise_c[0.0, 0.0]"
path= current_dir+'/data/' + experiment_name +'.csv'
df_physical_nn = pd.read_csv(path, comment='#')
dfs_physical_nn = split_experiment_into_laps(df_physical_nn) 
df_physical_nn = pd.concat(dfs_physical_nn[1:11])
# plot_experiment(experiment_name, df_physical_nn, title)

title = "Physical PP"
experiment_name = "physical/F1TENTH__2024-02-19_18-35-58Recording1_RCA2_pp_50Hz_vel1.0_noise_c[0.0, 0.0]"
path= current_dir+'/data/' + experiment_name +'.csv'
df_physical_pp = pd.read_csv(path, comment='#')
dfs_physical_pp = split_experiment_into_laps(df_physical_pp) 
df_physical_pp = pd.concat(dfs_physical_pp[0:10])
# plot_experiment(experiment_name, df_physical_pp, title)




# PLOT THE DISTANCES TO RACELINE SIMULATION
if False:
    distances_nn = get_position_errors(df_nn, wp_file)
    distance_stats_nn = get_distance_stats(distances_nn)
    mean_nn = distance_stats_nn['mean']
    std_dev_nn = distance_stats_nn['std_dev']

    distances_mpc = get_position_errors(df_mpc, wp_file)
    distance_stats_mpc = get_distance_stats(distances_mpc)
    mean_mpc = distance_stats_mpc['mean']
    std_dev_mpc = distance_stats_mpc['std_dev']

    distances_pp = get_position_errors(df_pp, wp_file)
    distance_stats_pp = get_distance_stats(distances_pp)
    mean_pp = distance_stats_pp['mean']
    std_dev_pp = distance_stats_pp['std_dev']

    plt.clf()
    plt.figure(figsize=(6, 2))

    plt.hist(distances_pp, bins=100, label="pp", color='green', histtype='step', linewidth=2, density=True)
    plt.hist(distances_mpc, bins=100, label="mpc", color='red', histtype='step', linewidth=2, density=True)
    plt.hist(distances_nn, bins=100,label="nn",  color='blue', histtype='step', linewidth=2, density=True)

    # Add mean lines
    plt.axvline(mean_pp, color='green', linestyle='dashed', linewidth=2)
    plt.axvline(mean_nn, color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(mean_mpc, color='red', linestyle='dashed', linewidth=2)

    # Add standard deviation fill
    # plt.ylim(0, plt.ylim()[1])
    # plt.fill_betweenx([0, plt.ylim()[1]], mean_pp - std_dev_pp, mean_pp + std_dev_pp, color='green', alpha=0.2)
    # plt.fill_betweenx([0, plt.ylim()[1]], mean_nn - std_dev_nn, mean_nn + std_dev_nn, color='blue', alpha=0.2)
    # plt.fill_betweenx([0, plt.ylim()[1]], mean_mpc - std_dev_mpc, mean_mpc + std_dev_mpc, color='red', alpha=0.2)


    plt.title('Distance to race line');
    plt.xlabel('Distance')
    plt.ylabel('%')
    plt.legend(fontsize='large')  # You can also use numeric values for specific sizes
    plt.savefig(f"{save_path}/distances_histogram_sim.png")
    plt.clf()    


# PLOT THE DISTANCES TO RACELINE PHYSICAL
if True:
    distances_physical_nn = get_position_errors(df_physical_nn, wp_file)
    distances_physical_pp = get_position_errors(df_physical_pp, wp_file)

    distances_stats_nn = get_distance_stats(distances_physical_nn)
    distances_stats_pp = get_distance_stats(distances_physical_pp)

    plt.clf()

    plt.figure(figsize=(6, 2))
    mean_nn = distances_stats_nn['mean']
    std_dev_nn = distances_stats_nn['std_dev']
    mean_pp = distances_stats_pp['mean']
    std_dev_pp = distances_stats_pp['std_dev']

    # Plot the histograms
    # plt.hist(distances_physical_pp, bins=100, label=f"pp (mean={mean_pp:.2f}, std_dev={std_dev_pp:.2f})", color='green', histtype='step', linewidth=2, density=True)
    # plt.hist(distances_physical_nn, bins=100, label=f"nn (mean={mean_nn:.2f}, std_dev={std_dev_nn:.2f})", color='blue', histtype='step', linewidth=2, density=True)
    plt.hist(distances_physical_pp, bins=100, label=f"pp", color='green', histtype='step', linewidth=2, density=True)
    plt.hist(distances_physical_nn, bins=100, label=f"nc", color='blue', histtype='step', linewidth=2, density=True)

    # Add mean lines
    plt.axvline(mean_pp, color='green', linestyle='dashed', linewidth=2)
    plt.axvline(mean_nn, color='blue', linestyle='dashed', linewidth=2)

    # Add standard deviation fill
    plt.ylim(0, plt.ylim()[1])
    # plt.fill_betweenx([0, plt.ylim()[1]], mean_pp - std_dev_pp, mean_pp + std_dev_pp, color='green', alpha=0.2)
    # plt.fill_betweenx([0, plt.ylim()[1]], mean_nn - std_dev_nn, mean_nn + std_dev_nn, color='blue', alpha=0.2)

    plt.title('Distance to race line');
    plt.xlabel('Distance')
    plt.ylabel('%')
    plt.legend()
    plt.savefig(f"{save_path}/distances_histogram_physical.png")
    plt.clf()    






plot_map_with_waypoints("RCA1")
plot_map_with_waypoints("RCA2")

# plot_progress(df_pp, df_nn)


