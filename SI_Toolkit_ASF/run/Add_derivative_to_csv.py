from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/Experiments_19_11_2025_BackToFront/Recordings/'
save_files_to = get_files_from
variables_for_derivative = ['pose_x',
                            'pose_y',
                            'pose_theta',
                            'pose_theta_sin',
                            'pose_theta_cos',
                            'linear_vel_x',
                            'angular_vel_z',
                            'slip_angle',
                            'steering_angle']
derivative_algorithm = "single_difference"

transform_dataset(get_files_from, save_files_to, transformation='append_derivatives',
                  split_on_column='experiment_index',
                  variables_for_derivative=variables_for_derivative, derivative_algorithm=derivative_algorithm)
