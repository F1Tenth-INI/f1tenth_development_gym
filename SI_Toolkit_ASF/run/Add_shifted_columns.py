from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_pid_reversed/Recordings/'
save_files_to = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_pid_reversed/Recordings/'


variables_to_shift = ['translational_control_calculated', 'angular_control_calculated', 'angular_control_pid_constr', 'translational_control_pid_constr']
indices_by_which_to_shift = [1]

transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns',
                    split_on_column='experiment_index',
                    variables_to_shift=variables_to_shift, indices_by_which_to_shift=indices_by_which_to_shift)
 
