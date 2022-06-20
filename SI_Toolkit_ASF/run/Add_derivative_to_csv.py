from SI_Toolkit.load_and_normalize import add_derivatives_to_csv_files

get_files_from = 'ExperimentRecordings'
save_files_to = 'SI_Toolkit_ASF/Experiments/Experiment-2/Recordings_with_derivative'
variables_for_derivative = ['pose_x', 'pose_y', 'pose_theta_sin', 'pose_theta_cos', 'linear_vel_x', 'linear_vel_y', 'angular_vel_z']
derivative_algorithm = "single_difference"

add_derivatives_to_csv_files(get_files_from, save_files_to, variables_for_derivative, derivative_algorithm)