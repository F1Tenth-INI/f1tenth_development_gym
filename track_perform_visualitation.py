import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import yaml

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
    plt.show()

# Load the data
map_name = 'RCA1'
exper_folder_path = './ExperimentRecordings/'
img_path = './utilities/maps/' + map_name + '/' + map_name # if no img, set to None
real_data = 'F1TENTH_ETF1_NNI__2023-11-23_15-54-27.csv' # if no real data, set to None
sim_data = 'F1TENTH_RCA1_neural_50Hz__2023-12-04_15-51-54.csv' # if no sim data, set to None


plot_wp(map_name, exper_folder_path, img_path, sim_data, real_data)
