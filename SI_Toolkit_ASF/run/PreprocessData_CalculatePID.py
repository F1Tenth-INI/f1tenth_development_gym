import numpy as np

from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise/Recordings/'
save_files_to = 'SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_pid/Recordings/'

STATE_VARIABLES = np.sort([
    'angular_vel_z',  # x5: yaw rate
    'linear_vel_x',   # x3: velocity in x direction
    'linear_vel_y',   # x6: velocity in y direction
    'pose_theta',  # x4: yaw angle
    'pose_theta_cos',
    'pose_theta_sin',
    'pose_x',  # x0: x position in global coordinates
    'pose_y',  # x1: y position in global coordinates
    'slip_angle',  # [DEPRECATED] x6: slip angle at vehicle center
    'steering_angle'  # x2: steering angle of front wheels
])

CONTROL_INPUTS = np.sort(['angular_control_calculated', 'translational_control_calculated'])


ODE_MODEL_OF_CAR_DYNAMICS = 'ODE:ks_pacejka'
ENV_CAR_PARAMETER_FILE = "gym_car_parameters.yml"
TIMESTEP_SIM = 0.01


transform_dataset(get_files_from, save_files_to, transformation='calculate_pid',
                  state_variables_pid_function=STATE_VARIABLES, control_inputs_pid_function=CONTROL_INPUTS,
                  ode_model_of_car_dynamics=ODE_MODEL_OF_CAR_DYNAMICS, env_car_parameter_file=ENV_CAR_PARAMETER_FILE,
                  timestep_sim=TIMESTEP_SIM)





