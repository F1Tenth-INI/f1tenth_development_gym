from SI_Toolkit.load_and_normalize import add_derivatives_to_csv_files

for folder in ['Test', 'Validate', 'Train']:
    experiment_folder = '0_AL'
    data_folder = 'Recordings_dungeon'
    get_files_from = f'SI_Toolkit_ASF/Experiments/{experiment_folder}/{data_folder}/{folder}'
    save_files_to = f'SI_Toolkit_ASF/Experiments/{experiment_folder}/{data_folder}_delta/{folder}'
    variables_for_derivative = ['pose_x',
                                'pose_y',
                                'pose_theta',
                                'pose_theta_sin',
                                'pose_theta_cos',
                                'linear_vel_x',
                                'angular_vel_z',
                                'slip_angle',
                                'steering_angle']
    derivative_algorithm = "backward_difference"

    add_derivatives_to_csv_files(get_files_from, save_files_to, variables_for_derivative, derivative_algorithm)
