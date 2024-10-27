import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import yaml
from PIL import Image
import os

def plot_waypoints(map_name, img_path, angle):
    
    # Load the data
    wp_data = pd.read_csv(img_path + '_wp.csv')
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_coords = np.dot(wp_data[[' x_m', ' y_m']], rotation_matrix.T)
    wp_x = rotated_coords[:, 0]
    wp_y = rotated_coords[:, 1]

    with open(img_path + '.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Load the background image
    img1 = Image.open(img_path + '.png')
    img = img1.rotate(angle, expand=True)
    img = np.array(img1)
    print(img)
    print(img.shape)
    
    # Determine the limits based on the origin and resolution
    x_min = data['origin'][0] 
    y_min = data['origin'][1]
    
    x_max = x_min + img.shape[1] * data['resolution'] 
    y_max = y_min + img.shape[0] * data['resolution'] 
    
    # Create the figure and axes
    fig, ax = plt.subplots()

    # Show the image
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')


    plt.plot(wp_x, wp_y, color='green', label='waypoints')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(map_name + ' Coordinate System')
    plt.legend(loc='upper right')

    plt.show()

def plot_wp(map_name, experiment_folder_path, img_path, sim, real):
    
    real_path = experiment_folder_path + real
    sim_path = experiment_folder_path + sim
    
    experiment_data = experiment_folder_path + sim.split('.')[0] + '_data/'
    print(experiment_data)
    
    # Load the data
    wp_data = pd.read_csv(img_path + '_wp.csv')
    if real_path is not None:
        real_data = pd.read_csv(real_path, skiprows=8)
    sim_data = pd.read_csv(sim_path, skiprows=8)
    
    sim_x = sim_data['pose_x'].to_numpy()
    sim_y = sim_data['pose_y'].to_numpy()
    
    wp_x = wp_data['x_m'].to_numpy()
    wp_y = wp_data['y_m'].to_numpy()

    if real_path is not None:
        real_x = real_data['pose_x'].to_numpy()
        real_y = real_data['pose_y'].to_numpy()

    with open(img_path + '.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Load the background image
    # print("Image path:", img_path+ '.png')
    img = plt.imread(img_path + '.png')
    # img = plt.imread('./utilities/maps/RCA1/RCA1_wp_min_curve_og.png')
    # Determine the limits based on the origin and resolution
    x_min = data['origin'][0]
    y_min = data['origin'][1]
    x_max = x_min + img.shape[1] * data['resolution']
    y_max = y_min + img.shape[0] * data['resolution']

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Show the image
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')

    # Mark the starting point with a cross
    plt.scatter(wp_x[0], wp_y[0], color='green', marker='o', s=200, label='Starting point waypoints')
    plt.scatter(sim_x[0], sim_y[0], color='purple', marker='o', s=200, label='Starting point sim')
    
    # Mark the end point with a cross
    plt.scatter(sim_x[-1], sim_y[-1], color='purple', marker='x', s=200, label='End point sim')
    
    plt.plot(wp_x, wp_y, color='green', label='waypoints')
    # plt.plot(sim_x, sim_y, color='purple', linestyle='dashdot', label='sim')
    if real_path is not None:
        plt.scatter(real_x[0], real_y[0], color='blue', marker='o', s=200, label='Starting point real')
        plt.scatter(real_x[-1], real_y[-1], color='blue', marker='x', s=200, label='End point real')
        plt.plot(real_x, real_y, color='blue', linestyle='dashed')
        
    # Create a color map
    cmap1 = plt.get_cmap('plasma')
    cmap2 = plt.get_cmap('viridis')
    
    # Calculate the number of segments
    num_segments1 = len(sim_x) - 1
    
    # Draw each segment individually
    for i in range(num_segments1):
        # Calculate the color value for this segment
        color = cmap1(i / num_segments1)

        # Draw the segment
        ax.plot(sim_x[i:i+2], sim_y[i:i+2], color=color, linestyle='dashdot')
        
    # Show the color bar
    sm = plt.cm.ScalarMappable(cmap=cmap1, norm=plt.Normalize(vmin=0, vmax=num_segments1))
    cbar1 = fig.colorbar(sm, ax=ax)
    cbar1.set_label('sim')
        
    if real_path is not None:
        num_segments2 = len(real_x) - 1     
        for i in range(num_segments2):
            # Calculate the color value for this segment
            color = cmap2(i / num_segments2)

            # Draw the segment
            ax.plot(real_x[i:i+2], real_y[i:i+2], color=color, linestyle='dotted')
        
        sm2 = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=plt.Normalize(vmin=0, vmax=num_segments2))
        cbar2 = plt.colorbar(sm2, ax=ax)
        cbar2.set_label('real')
    
    # Erstellen Sie die zweite Farbleiste
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(map_name + ' Coordinate System')
    plt.legend(loc='upper right')

    # Save the plot as a PNG
    plt.savefig(experiment_data + '_result.png')
    plt.show()
    
def plot_vector_wp(path, map_name, sim_path, path_wp, real_data):
    if real_data is not None:
        real_data = pd.read_csv(real_data, skiprows=8)
    if sim_path is not None:
        sim_data = pd.read_csv(sim_path, skiprows=8)
    if path_wp is not None:
        wp_data = pd.read_csv(path_wp)

    if real_data is not None:
        real_x = real_data['pose_x'].to_numpy()
        real_y = real_data['pose_y'].to_numpy()
        real_kappa = real_data['pose_theta'].to_numpy()
        real_ax = real_data['linear_vel_x'].to_numpy()
    
    if sim_path is not None:
        sim_x = sim_data['pose_x'].to_numpy()
        sim_y = sim_data['pose_y'].to_numpy()
        sim_kappa = sim_data['pose_theta'].to_numpy()
        sim_ax = sim_data['linear_vel_x'].to_numpy()

    if path_wp is not None:
        wp_x = wp_data['x_m'].to_numpy()
        wp_y = wp_data['y_m'].to_numpy()
        wp_kappa = wp_data['kappa_radpm'].to_numpy()
        wp_ax = wp_data['vx_mps'].to_numpy()
    
    # Create the vector plot
    plt.quiver(wp_x, wp_y, wp_ax*np.cos(wp_kappa), wp_ax*np.sin(wp_kappa), color='red', label='RCA1_wp')
    plt.quiver(real_x, real_y, real_ax*np.cos(real_kappa), real_ax*np.sin(real_kappa), color='blue', linestyle='dashed', label='RCA1_real')

    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RCA1 Coordinate System')
    plt.legend(['RCA1'], loc='upper right')

    # Save the plot as a PNG
    plt.savefig(path+map_name+'_test.png')

def plot_rotated_map(img_path, angle):
    # Load the data
    wp_data = pd.read_csv(img_path + '_wp.csv')
    
    # Load the background image
    img1 = Image.open(img_path + '.png')
    img = img1.rotate(angle, expand=True)
    img = np.array(img1)
    
    # Load the metadata (yaml file)
    with open(img_path + '.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Determine the image properties from YAML
    x_min = data['origin'][0] 
    y_min = data['origin'][1]
    resolution = data['resolution']
    
    x_max = x_min + img.shape[1] * resolution
    y_max = y_min + img.shape[0] * resolution

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Show the image
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')

def two_plot_waypoints():
    # Map parameters
    map_name = 'RCA1'
    map_name2 = 'RCA2'
    dir_name = 'utilities/maps'
    map_dir = os.path.join(dir_name, map_name)
    map_dir2 = os.path.join(dir_name, map_name2)
    waypoints_file = os.path.join(map_dir, map_name + '_wp.csv')
    waypoints_file2 = os.path.join(map_dir2, map_name2 + '_wp.csv')
    reverse_waypoints_file = os.path.join(map_dir, map_name + '_wp_reverse.csv')
    image_path = os.path.join(map_dir, map_name )
    image_path2 = os.path.join(map_dir2, map_name2 )
    
    waypoints_df = pd.read_csv(waypoints_file)
    waypoints_df2 = pd.read_csv(waypoints_file2)
    
    # Create a figure
    fig, ax = plt.subplots()


    # Rotate the reverse waypoints by 30 degrees clockwise
    theta = np.radians(-14.9)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_coords = np.dot(waypoints_df2[[' x_m', ' y_m']], rotation_matrix.T)

    # Shift the rotated reverse waypoints by 2 units in the x direction
    shifted_coords = rotated_coords.copy()
    shifted_coords[:, 0] -= 1.5
    shifted_coords[:, 1] += 0.5

    # Plot the original waypoints
    ax.plot(waypoints_df[' x_m'], waypoints_df[' y_m'], color='r', marker='o', linestyle='dashed', label= map_name + ' Waypoints')

    # # Plot the reverse waypoints
    # ax.plot(waypoints_df2[' x_m'], waypoints_df2[' y_m'], color='b', marker='x', linestyle='dotted', label=map_name2+ ' Waypoints')

    # Plot the shifted rotated reverse waypoints
    ax.plot(shifted_coords[:, 0], shifted_coords[:, 1], color='y', marker='x', linestyle='dotted', label=map_name2 + ' Waypoints')

    # Set labels
    ax.set_xlabel('x_m')
    ax.set_ylabel('y_m')
    ax.set_title(map_name + ' and ' + map_name2 + ' Waypoints')
    ax.legend()

    curvature_plot(waypoints_df, 'plasma', map_name, image_path)
    curvature_plot(waypoints_df2, 'viridis', map_name2, image_path2)
    
        
    
def curvature_plot(waypoints_dataframe, color, map_name, img_path):
    #load the data
    x, y = waypoints_dataframe[' x_m'], waypoints_dataframe[' y_m']

    # load the image
    img1 = Image.open(img_path + '.png')
    img = np.array(img1)
    with open(img_path + '.yaml', 'r') as file:
            data = yaml.safe_load(file)

    # Determine the image properties from YAML
    x_min = data['origin'][0] 
    y_min = data['origin'][1]
    resolution = data['resolution']
    
    x_max = x_min + img.shape[1] * resolution
    y_max = y_min + img.shape[0] * resolution
    
    # Compute the gradient of the x and y coordinates
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)

    # Compute the 2nd order gradient of the x and y coordinates
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # Compute the curvature
    # curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    # curvature = gaussian_filter1d(curvature, sigma=1)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # create colour map for the curvature
    norm = plt.Normalize(curvature.min(), curvature.max())
    lc = LineCollection(segments, cmap=color, norm=norm)
    lc.set_array(curvature)
    lc.set_linewidth(2)

    fig, ax = plt.subplots()
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('x_m')
    ax.set_ylabel('y_m')
    ax.set_title('Curvature of map ' + map_name)

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label('curvature')
    
# Load the data
map_name = 'RCA2'
exper_folder_path = './ExperimentRecordings/'
img_path = './utilities/maps/' + map_name + '/' + map_name # if no img, set to None
real_data = 'F1TENTH__2024-08-19_13-23-31Recording1_RCA2_neural_50Hz_vel_0.8_noise_c[0.0, 0.0]_mu_0.8.csv' # if no real data, set to None
sim_data = None # if no sim data, set to None



two_plot_waypoints()

plt.show()