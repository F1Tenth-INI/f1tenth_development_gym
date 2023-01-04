from SI_Toolkit.load_and_normalize import add_derivatives_to_csv_files

get_files_from = 'SI_Toolkit_ASF/Experiments/MPPI_DATASET_221224_MODEL_LEARNING/Recordings-basic/Test'
save_files_to = 'SI_Toolkit_ASF/Experiments/MPPI_DATASET_221224_MODEL_LEARNING/Recordings/Test'
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

add_derivatives_to_csv_files(get_files_from, save_files_to, variables_for_derivative, derivative_algorithm)