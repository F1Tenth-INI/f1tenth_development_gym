from SI_Toolkit.load_and_normalize import get_paths_to_datafiles, load_data

folder_with_data_to_calculate_norm_info = 'SI_Toolkit_ASF/Experiments/Experiment-1/Recordings/Train'

list_of_paths_to_datafiles = get_paths_to_datafiles(folder_with_data_to_calculate_norm_info)

# region Load data
df = load_data(list_of_paths_to_datafiles=list_of_paths_to_datafiles)
# endregion

for i in range(len(df)):
    if df[i]['LIDAR_3'].max() > 20.0:
        print('Problem:')
        print(list_of_paths_to_datafiles[i])
    elif df[i]['pose_theta'].max() > 20.0:
        print('Problem:')
        print(list_of_paths_to_datafiles[i])

print('Check_finished')