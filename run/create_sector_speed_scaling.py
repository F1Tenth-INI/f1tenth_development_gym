import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import yaml

# get current working directory
def path_dir(file_path):
    current_working_dir = os.getcwd()

    # remove the last folder from the path to get the base path
    if 'run' in current_working_dir:
        base_path = current_working_dir.split('run')[0]  
    else:
        base_path = current_working_dir  

    # joining the base path with the file path
    dir_path = os.path.join(base_path, file_path)

    # Check if the directory exists
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
        
    return dir_path

# Function to plot extrema points
def get_extrema(waypoints_df, x_col, y_col):
    x = waypoints_df[x_col]
    y = waypoints_df[y_col]

    # Find peaks (maxima)
    peaks, _ = find_peaks(y)
    # Find valleys (minima) by inverting the signal
    valleys, _ = find_peaks(-y)
    
    # Create a list of extrema points sorted by index
    extrema_list = [[peaks, y[peaks].to_numpy()], [valleys, y[valleys].to_numpy()]]
    extrema_list_sorted = sorted([(index, value) for extrema in extrema_list for index, value in zip(extrema[0], extrema[1])], key=lambda x: x[0])
    minima_list = [(index, value) for index, value in extrema_list_sorted if index in valleys]
    maxima_list = [(index, value) for index, value in extrema_list_sorted if index in peaks]

    return extrema_list_sorted, minima_list, maxima_list

def save_extrema_to_yaml(extrema_list_sorted, min_list, max_list, output_dir, default_global_speed=0.5):
    
    modified_list = []
    
    # go through the list and check if the difference between the values is greater than 0.6
    if extrema_list_sorted[0][1] < extrema_list_sorted[1][1]:
        i = 0
        while True:
            if i == len(extrema_list_sorted):
                modified_list.append(list(extrema_list_sorted[i-1]))
                break
            if abs(extrema_list_sorted[i][1] - extrema_list_sorted[i+1][1]) > 0.6:
                if not min_list:
                    modified_list.append(list(extrema_list_sorted[i]))
                    break
                modified_list.append(list(min_list[0]))
                min_list.pop(0)
                i += 2  # Erhöht i um 2
            else:
                min_list.pop(0)
                i += 2  # Erhöht i um 1, wenn die Bedingung nicht erfüllt ist
            
    sector_dic = {}
    sector_dic['Sector0'] = {'start': int(0), 'end': int(modified_list[0][0]), 'scaling': default_global_speed}
    
    for i in range(len(modified_list)-1):
        sector_dic['Sector' + str(i+1)] = {'start': int(modified_list[i][0]), 'end': int(modified_list[i+1][0]), 'scaling': default_global_speed}
    
    # split the extrema list into maxima and minima
    num_sector = {'n_sectors': int(len(modified_list))}
    default_speed = {'global_limit': default_global_speed}
    output_file = output_dir+ '/' + 'speed_scaling.yaml'
    
    # Save the dictionary to a YAML file
    with open(output_file, 'w') as file:
        yaml.dump(sector_dic, file, default_flow_style=False, sort_keys=False)
        yaml.dump(default_speed, file, default_flow_style=False)
        yaml.dump(num_sector, file, default_flow_style=False)

# Map parameters
map_name = 'RCA1'
dir_name = 'utilities/maps'

# File paths
map_dir = os.path.join(dir_name, map_name)
waypoints_file = os.path.join(map_dir, map_name + '_wp.csv')
reverse_waypoints_file = os.path.join(map_dir, map_name + '_wp_reverse.csv')
image_path = os.path.join(map_dir, map_name + '.png')

# Load the waypoints
waypoints_df = pd.read_csv(path_dir(waypoints_file))

# Get extrema points
extrema, min, max = get_extrema(waypoints_df, '# s_m', ' vx_mps')

dir_path = path_dir(map_dir)
save_extrema_to_yaml(extrema, min, max, dir_path)

