from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = '/Users/marcinpaluch/PycharmProjects/f1tenth_development_gym/SI_Toolkit_ASF/Experiments/Experiments_19_11_2025_BackToFront/Recordings'
save_files_to = '/Users/marcinpaluch/PycharmProjects/f1tenth_development_gym/SI_Toolkit_ASF/Experiments/Experiments_19_11_2025_BackToFront/Recordings'


variables_to_shift = variables_for_derivative = \
['translational_control', 'angular_control',
    'D_pose_x',
    'D_pose_y',
    'D_pose_theta',
    'D_pose_theta_sin',
    'D_pose_theta_cos',
    'D_linear_vel_x',
    'D_angular_vel_z',
    'D_slip_angle',
    'D_steering_angle']

indices_by_which_to_shift = [-1, 1]

transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns',
                    split_on_column='experiment_index',
                    variables_to_shift=variables_to_shift, indices_by_which_to_shift=indices_by_which_to_shift)
 
