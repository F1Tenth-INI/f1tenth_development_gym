from SI_Toolkit.data_preprocessing import transform_dataset
import os

get_files_from = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_reversed/Recordings/'
save_files_to = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_reversed/Recordings/'
variables_to_flip = [
    'D_pose_x',
    'D_pose_y',
    'D_pose_theta',
    'D_pose_theta_sin',
    'D_pose_theta_cos',
    'D_linear_vel_x',
    'D_angular_vel_z',
    'D_slip_angle',
    'D_steering_angle',
]


transform_dataset(get_files_from, save_files_to, transformation='flip_column_signs',
                  variables_to_flip=variables_to_flip,)

