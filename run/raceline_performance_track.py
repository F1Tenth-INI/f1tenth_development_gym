# DEPRECATED
# The functionality of this script is now integrated in the utilities/ExperimentAnalysis.py script
# And is automatically called when running the experiment


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString, Point
import os
# Annahme: 'optimal_line.csv' enthält die optimale Linie, 'recorded_line.csv' enthält die aufgezeichnete Linie
def error_func(waypoints_file, recorded_file, steps, map_name, controller_name):
    # Lade die Daten aus den CSV-Dateien
    optimal_data = pd.read_csv(waypoints_file)
    recorded_data = pd.read_csv(recorded_file, skiprows=8)


    # Annahme: Die Spalten 'x' und 'y' enthalten die Positionen der Fahrspurlinien
    optimal_x = optimal_data[' x_m'].values
    optimal_y = optimal_data[' y_m'].values

    optimal = list(zip(optimal_x, optimal_y))
    optimal_line = LineString(optimal).coords

    recorded_x = recorded_data['pose_x'].values
    recorded_y = recorded_data['pose_y'].values
    time_recorded = recorded_data['time'].values

    recorded = list(zip(recorded_x, recorded_y))
    recorded_line = LineString(recorded)

    error = []

    for i in range(len(recorded_x)):
        error.append(recorded_line.distance(Point(optimal_line[i%len(optimal_x)])))

    error_data = [max(error), min(error), np.mean(error), np.std(error), np.var(error)]

    # Plot der optimalen Linie und der aufgezeichneten Linie
    plt.figure()
    plt.plot(optimal_x, optimal_y, label='Raceline')
    plt.plot(recorded_x, recorded_y, label='Recorded Line')
    plt.legend()

    plt.xlabel('X-Position')
    plt.ylabel('Y-Position')
    plt.title('Comparison between '+ controller_name +' raceline and recorded line waypoints on '+ map_name)
    
    plt.figure()
    plt.plot(time_recorded[:steps], error[:steps],color='cyan', label='Position error with '+controller_name+' and waypoints')

    plt.xlabel('Time [s]', fontsize=24)
    plt.ylabel('Error [m]', fontsize=24)
    plt.title('Position error with Recording of '+controller_name+' and waypoints on '+ map_name, fontsize=24)
    plt.tick_params(axis='both', labelsize=24)
    plt.legend(loc='upper right', fontsize=24)
    plt.grid()
        
    return time_recorded, error, error_data

def boxplot_error(error_datas):
    
    fig, ax = plt.subplots()
    ax.set_ylabel('Error [m]')
    ax.set_title('Position error with Recording of '+controller_name+' and waypoints on '+ map_name_RCA2)
    ax.boxplot(error_datas)
    
# intialize the map name and controller name
map_name_RCA2 = 'RCA2'

controller_name = 'neural'

Analyse_folder = 'ExperimentRecordings/Analyse'
waypoints_file_RCA2 = 'utilities/maps/'+map_name_RCA2+'/RCA2_wp.csv'
real_data = 'ExperimentRecordings/F1TENTH__2024-08-30_11-51-34Recording1_RCA2_neural_50Hz_vel_0.8_noise_c[0.0, 0.0]_mu_0.7.csv'
filename = os.path.basename(real_data).split('.')[0]
defining_steps_for_1_lap = 3000

t1, error1, ed1 = error_func(waypoints_file_RCA2, real_data, defining_steps_for_1_lap, map_name_RCA2, controller_name)
t2, error2, ed2 = error_func(waypoints_file_RCA2, real_data, defining_steps_for_1_lap, map_name_RCA2, controller_name)


error_data = [ed1, ed2]
boxplot_error([error1])
error_datas = [ed1]

#round to 3 decimal places
for i in range(len(error_datas)):
    error_datas[i] = [round(x, 4) for x in error_datas[i]]

# if there is no folder with this name, create one
if not os.path.exists(Analyse_folder):
    os.makedirs(Analyse_folder)

path_to_save = os.path.join(Analyse_folder, f'analysis_{filename}.csv')


pd.DataFrame(error_datas, columns=['max', 'min', 'mean', 'std', 'var']).to_csv(path_to_save)

plt.show()
