from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_reversed/Recordings/'
save_files_to = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_reversed/Recordings/'


transform_dataset(get_files_from, save_files_to, transformation='time_reverse',)

