import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import yaml

def plot_data(map_name, experiment_folder_path, img_path, controller,laptime):
    sim_data = pd.read_csv(experiment_folder_path, skiprows=8)
    wp_data = pd.read_csv(img_path + '_wp.csv')
    
    wp_x = wp_data['x_m'].to_numpy()
    wp_y = wp_data['y_m'].to_numpy()
    sim_x = sim_data['pose_x'].to_numpy()
    sim_y = sim_data['pose_y'].to_numpy()
    
    laptime = str(laptime)
    
    with open(img_path + '.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Load the background image
    img = plt.imread(img_path + '.png')

    # Determine the limits based on the origin and resolution
    x_min = data['origin'][0]
    y_min = data['origin'][1]
    x_max = x_min + img.shape[1] * data['resolution']
    y_max = y_min + img.shape[0] * data['resolution']
    
    fig, ax = plt.subplots()
    
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')
    
    plt.scatter(sim_x[0], sim_y[0], color='red', marker='x', s=100, label='Starting point Race car')
    
    plt.plot(sim_x, sim_y, color='red', label='position Race car')
    plt.plot(wp_x, wp_y, color='green', marker='o', markersize=2,linestyle='dotted', label='waypoints')
    plt.grid(True)
    plt.xlabel('x [m]', fontsize=24)
    plt.xlim(-24, 8)
    plt.ylim(-4, 12)
    plt.ylabel('y [m]', fontsize=24)
    plt.title(map_name + ' Coordinate System with controller: ' + controller, fontsize=24)
    # if laptime != 'crashed':
    #     plt.suptitle('2-Laptime: ' + laptime+' s', fontsize=24, fontweight='bold')
    # else:
    #     plt.suptitle('Crashed', fontsize=24, fontweight='bold')
    plt.tick_params(axis='both', labelsize=24)
    plt.legend(loc='upper right', fontsize=24)

    # Save the plot as a PNG
    plt.show()

def plot_wp(map_name,img_path):
    wp_data = pd.read_csv(img_path + '_wp.csv')
    
    wp_x = wp_data['x_m'].to_numpy()
    wp_y = wp_data['y_m'].to_numpy()
    
    with open(img_path + '.yaml', 'r') as file:
        data = yaml.safe_load(file)

    # Load the background image
    img = plt.imread(img_path + '.png')

    # Determine the limits based on the origin and resolution
    x_min = data['origin'][0]
    y_min = data['origin'][1]
    x_max = x_min + img.shape[1] * data['resolution']
    y_max = y_min + img.shape[0] * data['resolution']
    
    fig, ax = plt.subplots()
    
    ax.imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')
    
    plt.plot(wp_x, wp_y, color='green', label='waypoints')
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(map_name + ' Coordinate System')
    plt.legend(loc='upper right')

    # Save the plot as a PNG
    plt.show()

def plot_wp_in(map_name, experiment_folder_path, img_path, sim, real):
    
    if real is None:
        real_path = None
    else:
        real_path = experiment_folder_path + real
        
    sim_path = experiment_folder_path + sim
    experiment_data = experiment_folder_path + sim.split('.')[0] + '_data/'
    
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
    img = plt.imread(img_path + '.png')

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
    plt.savefig(experiment_data + 'Plot_' + map_name + '_wp.png')
    print("Plot saved as: ", experiment_data + sim.split('.')[0] +'Plot' + map_name + "_wp.png")
    plt.show()
    
def plot_vector_wp(map_name, experiment_folder_path, img_path, sim, real):
    if real is None:
        real_path = None
    else:
        real_path = experiment_folder_path + real
            
    sim_path = experiment_folder_path + sim
    experiment_data = experiment_folder_path + sim.split('.')[0] + '_data/'
    
    # Load the data
    wp_data = pd.read_csv(img_path + '_wp.csv')
    if real_path is not None:
        real_data = pd.read_csv(real_path, skiprows=8)
    sim_data = pd.read_csv(sim_path, skiprows=8)

    if real_path is not None:
        real_time = real_data['time'].to_numpy()
        real_x = real_data['pose_x'].to_numpy()
        real_y = real_data['pose_y'].to_numpy()
        real_kappa = real_data['pose_theta'].to_numpy()
        real_vx = real_data['linear_vel_x'].to_numpy()
        real_translational_con_cal = real_data['translational_control_calculated'].to_numpy()
        real_angular_con_cal = real_data['angular_control_calculated'].to_numpy()
    
    if sim_path is not None:
        sim_time = sim_data['time'].to_numpy()
        sim_x = sim_data['pose_x'].to_numpy()
        sim_y = sim_data['pose_y'].to_numpy()
        sim_kappa = sim_data['pose_theta'].to_numpy()
        sim_vx = sim_data['linear_vel_x'].to_numpy()
        sim_translational_con_applied = sim_data['translational_control_applied'].to_numpy()
        sim_translational_con_cal = sim_data['translational_control_calculated'].to_numpy()
        sim_angular_con_applied = sim_data['angular_control_applied'].to_numpy()
        sim_angular_con_cal = sim_data['angular_control_calculated'].to_numpy()

    wp_x = wp_data['x_m'].to_numpy()
    wp_y = wp_data['y_m'].to_numpy()
    wp_kappa = wp_data['kappa_radpm'].to_numpy()
    wp_vx = wp_data['vx_mps'].to_numpy()
    
    # Matching the length of the arrays to compare
    if real_path is not None and sim_path is not None:
        min_time = min(len(sim_time), len(real_time))
        min_length = min(len(sim_x), len(real_x))
        sim_time = sim_time[:min_time]
        real_time = real_time[:min_time]
        sim_x = sim_x[:min_length]
        real_x = real_x[:min_length]
        sim_y = sim_y[:min_length]
        real_y = real_y[:min_length]
        sim_kappa = sim_kappa[:min_length]
        real_kappa = real_kappa[:min_length]
        sim_vx = sim_vx[:min_length]
        real_vx = real_vx[:min_length]
        sim_angular_con_applied = sim_angular_con_applied[:min_length]
        real_angular_con_cal = real_angular_con_cal[:min_length]
        sim_angular_con_cal = sim_angular_con_cal[:min_length]
        sim_translational_con_applied = sim_translational_con_applied[:min_length]
        real_translational_con_cal = real_translational_con_cal[:min_length]
        sim_translational_con_cal = sim_translational_con_cal[:min_length]
    
    # Calculate the error
    error_abs_pos_x = np.abs(sim_x - real_x)
    error_abs_pos_y = np.abs(sim_y - real_y)
    error_abs_kappa = np.abs(sim_kappa - real_kappa)
    error_abs_vx = np.abs(sim_vx - real_vx)
    error_pos = np.sqrt(error_abs_pos_x**2 + error_abs_pos_y**2)
    
    error_translation_sim = np.abs(sim_translational_con_applied - sim_translational_con_cal)
    error_angular_sim = np.abs(sim_angular_con_applied - sim_angular_con_cal)
    error_translational_con_cal = np.abs(sim_translational_con_cal - real_translational_con_cal)
    normalized_error_translational_con_cal = error_translational_con_cal / np.max(real_translational_con_cal)
    error_angular_con_cal = np.abs(sim_angular_con_cal - real_angular_con_cal)    
    
    # Load the background image
    with open(img_path + '.yaml', 'r') as file:
        data = yaml.safe_load(file)
    img = plt.imread(img_path + '.png')
    
    # Determine the limits based on the origin and resolution
    x_min = data['origin'][0]
    y_min = data['origin'][1]
    x_max = x_min + img.shape[1] * data['resolution']
    y_max = y_min + img.shape[0] * data['resolution']

    #TODO: Calculate one laptime for real and sim
    
    
    # Create the figure and axes    
    fig, ax = plt.subplots(2)
    ax[0].imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')
    ax[1].imshow(img, extent=[x_min, x_max, y_min, y_max], cmap='gray')
    
    # Create the vector plot
    # Subplot 1 with real data
    if real_path is not None:
        ax[0].quiver(wp_x, wp_y, wp_vx*np.cos(wp_kappa), wp_vx*np.sin(wp_kappa), color='red', label=map_name + '_wp')
        ax[0].quiver(real_x, real_y, real_vx*np.cos(real_kappa), real_vx*np.sin(real_kappa), color='blue', label= map_name + '_real')
        ax[0].grid(True)
        ax[0].set_title(map_name + ' Coordinate System 1 real vs wp')
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].legend()
        
        # Fügen Sie Zeitmarkierungen hinzu
        for i in range(0, len(sim_time), 250): # last is 250 for 5s since 50Hz sampling rate
            ax[0].annotate(str(real_time[i]), (real_x[i], real_y[i]))
            ax[1].annotate(str(sim_time[i]), (sim_x[i], sim_y[i]))
        

    # Subplot 2 with sim data   
    ax[1].quiver(wp_x, wp_y, wp_vx*np.cos(wp_kappa), wp_vx*np.sin(wp_kappa), color='red', label=map_name +'_wp')
    ax[1].quiver(sim_x, sim_y, sim_vx*np.cos(sim_kappa), sim_vx*np.sin(sim_kappa), color='purple', label=map_name + '_sim')
    ax[1].grid(True)
    ax[1].set_title(map_name + ' Coordinate System 2 sim vs wp')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].legend()
    
    fig, bx = plt.subplots(2)
    
    # Subplot 3 with error and time
    bx[0].plot(sim_time, error_pos, color='green', label='error between sim and real position')
    bx[0].plot(sim_time, error_abs_kappa, color='blue', label='error between sim and real kappa')
    bx[0].plot(sim_time, error_abs_vx, color='red', label='error between sim and real vx')
    bx[0].grid(True)
    bx[0].set_title('Error over time')
    bx[0].set_xlabel('time')
    bx[0].set_ylabel('error')
    bx[0].legend()

    # Subplot 4 with error and time
    # ax[3].plot(sim_time, error_translation_sim, color='green', linestyle='dashed', label='error between applied and calculated translational control sim')
    # ax[3].plot(sim_time, error_angular_sim, color='red', label='error between applied and calculated angular control sim')
    bx[1].plot(sim_time, error_angular_con_cal, color='blue', label='error between calculated angular control sim and real')
    bx[1].plot(sim_time, normalized_error_translational_con_cal, color='orange', label='normalized error between calculated translational control sim and real')
    # bx[1].plot(sim_time, error_translational_real, color='orange', label='error between applied and calculated translational control real') #TODO: get data from real to plot this
    # bx[1].plot(sim_time, error_angular_real, color='blue', label='error between applied and calculated angular control real') #TODO: get data from real to plot this
    bx[1].grid(True)
    bx[1].set_title('Error over time')
    bx[1].set_xlabel('time')
    bx[1].set_ylabel('error')
    bx[1].legend()

    # Save the plot as a PNG
    # plt.savefig(experiment_data + sim.split('.')[0] + '_Plot_' + map_name + '_wp_vector.png')
    # plt.savefig(exper_folder_path + 'Performance_tracking/Plot_' + sim.split('.')[0] + '_wp_vector.png')
    plt.show()

# # Set the data to be plotted paths

map_name = 'TEST'
exper_folder_path = './ExperimentRecordings/'
img_path = './utilities/maps/' + map_name + '/' + map_name # if no img, set to None
real_data = None # if no real data, set to None 'F1TENTH_ETF1_NNI__2023-11-23_15-54-27.csv'
sim_data = 'F1TENTH__2024-05-01_17-58-46_TEST_pajecka_50Hz.csv' # if no sim data, set to None

plot_wp_in(map_name, exper_folder_path, img_path, sim_data, real_data)

# # Plot the data with the desired settings
# ex = 'ExperimentRecordings/Physical_car/F1TENTH_ETF1_NNI__2023-12-18_14-11-50.csv'
# # controller = 'PP'
# # laptime = 67.61 #'crashed'
# # plot_wp_in(map_name, exper_folder_path, img_path, sim_data, real_data)
# # plot_vector_wp(map_name, exper_folder_path, img_path, sim_data, real_data)
# plot_wp(map_name, img_path)
# # plot_data(map_name, ex, img_path, controller, laptime)

# map = "TEST"
# path = "utilities/maps/TEST/TEST"

# plot_wp(map, path)