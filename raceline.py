import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString, Point

# Annahme: 'optimal_line.csv' enthält die optimale Linie, 'recorded_line.csv' enthält die aufgezeichnete Linie
def error_func(waypoints_file, recorded_file):
    # Lade die Daten aus den CSV-Dateien
    optimal_data = pd.read_csv(waypoints_file)
    recorded_data = pd.read_csv(recorded_file, skiprows=8)


    # Annahme: Die Spalten 'x' und 'y' enthalten die Positionen der Fahrspurlinien
    optimal_x = optimal_data['x_m'].values
    optimal_y = optimal_data['y_m'].values

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
    plt.title('Comparison between RCA2 raceline and recorded line '+recorded_file.split('-')[-1].split('_')[-2])
        
    return time_recorded, error, error_data

waypoints_file_RCA2 = 'utilities/maps/RCA2/RCA2_wp.csv'
waypoints_file_RCA1 = 'utilities/maps/RCA1/RCA1_wp.csv'
real1 = 'ExperimentRecordings/Physical_car/F1TENTH__2023-11-23_15-54-27_RCA1_neural_LSTM3_50Hz.csv'
real2 = 'ExperimentRecordings/Physical_car/F1TENTH__2023-12-18_14-03-54_RCA2_neural_LSTM4_50Hz.csv'
real3 = 'ExperimentRecordings/Physical_car/F1TENTH__2023-12-18_14-11-50_RCA2_pp_50Hz.csv'

defining_steps_for_1_lap = 3000

t1, error1, ed1 = error_func(waypoints_file_RCA1, real1)
t2, error2, ed2 = error_func(waypoints_file_RCA2, real2)
t3, error3, ed3 = error_func(waypoints_file_RCA2, real3)

plt.figure()
plt.plot(t1[:defining_steps_for_1_lap], error1[:defining_steps_for_1_lap],color='cyan', label='Position error with LSTM3 and waypoints')

plt.xlabel('Time [s]', fontsize=24)
plt.ylabel('Error [m]', fontsize=24)
plt.title('Position error with Recording of LSTM3 and waypoints on RCA1', fontsize=24)
plt.tick_params(axis='both', labelsize=24)
plt.legend(loc='upper right', fontsize=24)
plt.grid()


plt.figure()
plt.plot(t2[:defining_steps_for_1_lap], error2[:defining_steps_for_1_lap],color='cyan', label='Position error with LSTM4 and waypoints')
plt.plot(t3[:defining_steps_for_1_lap], error3[:defining_steps_for_1_lap],color='grey', label='Position error with PP and waypoints')

plt.xlabel('Time [s]', fontsize=24)
plt.ylabel('Error [m]', fontsize=24)
plt.title('Position error with Recording of LSTM4, PP and waypoints on RCA2', fontsize=24)
plt.tick_params(axis='both', labelsize=24)
plt.legend(loc='upper right', fontsize=24)
plt.grid()


error_datas = [ed1, ed2, ed3]

#round to 3 decimal places
for i in range(len(error_datas)):
    error_datas[i] = [round(x, 4) for x in error_datas[i]]

# pd.DataFrame(error_datas).to_csv('error_datas.csv')
# insert column names
pd.DataFrame(error_datas, columns=['max', 'min', 'mean', 'std', 'var']).to_csv('error_datas_3.csv')

plt.show()

# r1 = 'ExperimentRecordings/F1TENTH__2023-12-27_20-08-01_RCA2_neural_lstm1_50Hz.csv'
# r2 = 'ExperimentRecordings/F1TENTH__2023-12-28_14-14-05_RCA2_neural_lstm2_50Hz.csv'
# r3 = 'ExperimentRecordings/F1TENTH__2023-12-28_14-14-49_RCA2_neural_lstm3_50Hz.csv'
# r4 = 'ExperimentRecordings/F1TENTH__2023-12-28_14-16-10_RCA2_neural_lstm4_50Hz.csv'
# r5 = 'ExperimentRecordings/F1TENTH__2023-12-28_14-17-08_RCA2_neural_lstm4_delayed_50Hz.csv'

# mpc_e = 'ExperimentRecordings/F1TENTH__2023-12-27_16-27-29_RCA2_mpc_50Hz.csv'
# pp_e = 'ExperimentRecordings/F1TENTH__2023-12-27_16-35-49_RCA2_pp_50Hz.csv'

# g1 = 'ExperimentRecordings/F1TENTH__2023-12-27_20-06-35_RCA2_neural_gru1_50Hz.csv'
# g2 = 'ExperimentRecordings/F1TENTH__2023-12-28_10-02-18_RCA2_neural_gru2_50Hz.csv'
# g3 = 'ExperimentRecordings/F1TENTH__2023-12-28_10-02-55_RCA2_neural_gru3_50Hz.csv'

# d0 = 'ExperimentRecordings/F1TENTH__2023-12-27_20-26-55_RCA2_neural_dense0_50Hz.csv'
# d1 = 'ExperimentRecordings/F1TENTH__2023-12-27_20-14-14_RCA2_neural_dense1_50Hz.csv'
# d2 = 'ExperimentRecordings/F1TENTH__2023-12-28_09-53-50_RCA2_neural_dense2_50Hz.csv'
# d3 = 'ExperimentRecordings/F1TENTH__2023-12-28_09-54-15_RCA2_neural_dense3_50Hz.csv'



# t1, error1, ed1 = error_func(waypoints_file, r1)
# t2, error2, ed2 = error_func(waypoints_file, r2)
# t3, error3, ed3 = error_func(waypoints_file, r3)
# t4, error4, ed4 = error_func(waypoints_file, r4)
# t5, error5, ed5 = error_func(waypoints_file, r5)

# t6, error6, ed6 = error_func(waypoints_file, mpc_e)
# t7, error7, ed7 = error_func(waypoints_file, pp_e)

# t8, error8, ed8 = error_func(waypoints_file, g1)
# t9, error9, ed9 = error_func(waypoints_file, g2)
# t10, error10, ed10 = error_func(waypoints_file, g3)

# t11, error11, ed11 = error_func(waypoints_file, d0)
# t12, error12, ed12 = error_func(waypoints_file, d1)
# t13, error13, ed13 = error_func(waypoints_file, d2)
# t14, error14, ed14 = error_func(waypoints_file, d3)

# error_datas = [ed1, ed2, ed3, ed4, ed5, ed6, ed7, ed8, ed9, ed10, ed11, ed12, ed13, ed14]

# #round to 3 decimal places
# for i in range(len(error_datas)):
#     error_datas[i] = [round(x, 4) for x in error_datas[i]]




# # Plot LSTM errors with waypoints on RCA2 
# plt.figure()
# plt.plot(t1[:defining_steps_for_1_lap], error1[:defining_steps_for_1_lap],color='blue', label='Position error with LSTM1 and waypoints')
# plt.plot(t2[:defining_steps_for_1_lap], error2[:defining_steps_for_1_lap],color='green', label='Position error with LSTM2 and waypoints')
# plt.plot(t3[:defining_steps_for_1_lap], error3[:defining_steps_for_1_lap],color='red', label='Position error with LSTM3 and waypoints')
# plt.plot(t4[:defining_steps_for_1_lap], error4[:defining_steps_for_1_lap],color='orange', label='Position error with LSTM4 and waypoints')
# plt.plot(t5[:defining_steps_for_1_lap], error5[:defining_steps_for_1_lap],color='purple', label='Position error with delayed control inputs (0.1 s) with LSTM4 and waypoints')
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')
# plt.plot(t7[:defining_steps_for_1_lap], error7[:defining_steps_for_1_lap],color='grey', label='Position error with PP and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different LSTMs and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()

# # Plot PP and MPC errors with waypoints on RCA2
# plt.figure()
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')
# plt.plot(t7[:defining_steps_for_1_lap], error7[:defining_steps_for_1_lap],color='grey', label='Position error with PP and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different MPC, PP and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()

# # Plot Dense0, MPC and PP errors with waypoints on RCA2
# plt.figure()
# plt.plot(t11[:defining_steps_for_1_lap], error11[:defining_steps_for_1_lap],color='blue', label='Position error with Dense0 and waypoints')
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')
# plt.plot(t7[:defining_steps_for_1_lap], error7[:defining_steps_for_1_lap],color='grey', label='Position error with PP and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different MPC, PP and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()


# plt.figure()
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')
# plt.plot(t12[:defining_steps_for_1_lap], error12[:defining_steps_for_1_lap],color='green', label='Position error with Dense1 and waypoints')
# plt.plot(t8[:defining_steps_for_1_lap], error8[:defining_steps_for_1_lap],color='blue', label='Position error with GRU1 and waypoints')
# plt.plot(t1[:defining_steps_for_1_lap], error1[:defining_steps_for_1_lap],color='red', label='Position error with LSTM1 and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different Dense1, GRU1, LSTM1, MPC and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()


# plt.figure()
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')
# plt.plot(t13[:defining_steps_for_1_lap], error13[:defining_steps_for_1_lap],color='red', label='Position error with Dense2 and waypoints')
# plt.plot(t9[:defining_steps_for_1_lap], error9[:defining_steps_for_1_lap],color='green', label='Position error with GRU2 and waypoints')
# plt.plot(t2[:defining_steps_for_1_lap], error2[:defining_steps_for_1_lap],color='blue', label='Position error with LSTM2 and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different Dense2, GRU2, LSTM2, MPC and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()

# plt.figure()
# plt.plot(t10, error10,color='blue', label='Position error with GRU3 and waypoints')
# plt.plot(t14, error14,color='orange', label='Position error with Dense3 and waypoints')
# plt.plot(t3, error3,color='red', label='Position error with LSTM3 and waypoints')
# plt.plot(t7, error7,color='grey', label='Position error with PP and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different Dense3, GRU3, LSTM3, PP and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()

# plt.figure()
# plt.plot(t4[:defining_steps_for_1_lap], error4[:defining_steps_for_1_lap],color='orange', label='Position error with LSTM4 and waypoints')
# plt.plot(t5[:defining_steps_for_1_lap], error5[:defining_steps_for_1_lap],color='purple', label='Position error with delayed control inputs (0.1 s) with LSTM4 and waypoints')
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different LSTM4s, MPC and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()


# plt.figure()
# plt.plot(t8[:defining_steps_for_1_lap], error8[:defining_steps_for_1_lap],color='blue', label='Position error with GRU1 and waypoints')
# plt.plot(t9[:defining_steps_for_1_lap], error9[:defining_steps_for_1_lap],color='green', label='Position error with GRU2 and waypoints')
# plt.plot(t10[:defining_steps_for_1_lap], error10[:defining_steps_for_1_lap],color='red', label='Position error with GRU3 and waypoints')
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')
# plt.plot(t7[:defining_steps_for_1_lap], error7[:defining_steps_for_1_lap],color='grey', label='Position error with PP and waypoints')

# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different GRUs and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()


# plt.figure()
# plt.plot(t11[:defining_steps_for_1_lap], error11[:defining_steps_for_1_lap],color='blue', label='Position error with Dense0 and waypoints')
# plt.plot(t12[:defining_steps_for_1_lap], error12[:defining_steps_for_1_lap],color='green', label='Position error with Dense1 and waypoints')
# plt.plot(t13[:defining_steps_for_1_lap], error13[:defining_steps_for_1_lap],color='red', label='Position error with Dense2 and waypoints')
# plt.plot(t14[:defining_steps_for_1_lap], error14[:defining_steps_for_1_lap],color='orange', label='Position error with Dense3 and waypoints')
# plt.plot(t6[:defining_steps_for_1_lap], error6[:defining_steps_for_1_lap],color='cyan', label='Position error with MPC and waypoints')
# plt.plot(t7[:defining_steps_for_1_lap], error7[:defining_steps_for_1_lap],color='grey', label='Position error with PP and waypoints')


# plt.xlabel('Time [s]', fontsize=24)
# plt.ylabel('Error [m]', fontsize=24)
# plt.title('Position error with different Denses and waypoints on RCA2', fontsize=24)
# plt.tick_params(axis='both', labelsize=24)
# plt.legend(loc='upper right', fontsize=24)
# plt.grid()

# print(error_datas)

# # pd.DataFrame(error_datas).to_csv('error_datas.csv')
# # insert column names
# pd.DataFrame(error_datas, columns=['max', 'min', 'mean', 'std', 'var']).to_csv('error_datas_2.csv')
# plt.show()
