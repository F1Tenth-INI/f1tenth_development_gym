import numpy as np
import tensorflow as tf

from SI_Toolkit_ASF.car_model import car_model
from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid


def check_step_dynamics(state, inputs):
    # Predict ASF
    model = car_model('ODE:st', True, 1, "utilities/car_files/ini_car_parameters.yml",
                      dt)

    state_ASF = tf.constant(state[np.newaxis, ...], dtype=tf.float32)
    for next_input in inputs:
        state_ASF = model._step_dynamics_st(state_ASF, tf.constant(next_input[np.newaxis, ...], dtype=tf.float32), None)
    state_ASF = state_ASF.numpy()[0]

    # Predict F1tenth
    mu = model.car_parameters['mu']  # friction coefficient  [-]
    C_Sf = model.car_parameters['C_Sf']  # cornering stiffness front [1/rad]
    C_Sr = model.car_parameters['C_Sr']  # cornering stiffness rear [1/rad]
    lf = model.car_parameters['lf']  # distance from venter of gracity to front axle [m]
    lr = model.car_parameters['lr']  # distance from venter of gracity to rear axle [m]
    h = model.car_parameters['h']  # center of gravity height of toal mass [m]
    m = model.car_parameters['m']  # Total Mass of car [kg]
    I_car = model.car_parameters['I']  # Moment of inertia for entire mass about z axis  [kgm^2]
    # g = model.car_parameters['g']

    state_f1tenth = np.array([pos[0], pos[1], steering_angle, vel, yaw, yaw_dot, slip_angle])
    for next_input in inputs:
        delta_state_f1tenth = vehicle_dynamics_st(state_f1tenth, next_input, mu, C_Sf, C_Sr, lf, lr, h, m, I_car,
                                                  -0.4189, 0.4189, -3.2, 3.2, 7.319, 9.01, -5.0, 20)
        state_f1tenth = state_f1tenth + dt * delta_state_f1tenth

    # Print outputs
    state_ASF = state_ASF.round(4)
    state_f1tenth = state_f1tenth.round(4)
    print('      | ASF | f1tenth')
    print('Position x:', state_ASF[5], state_f1tenth[0])
    print('Position y:', state_ASF[6], state_f1tenth[1])
    print('Steering angle:', state_ASF[8], state_f1tenth[2])
    print('Velocity:', state_ASF[1], state_f1tenth[3])
    print('Yaw:', state_ASF[2], state_f1tenth[4])
    print('Angular Velocity:', state_ASF[0], state_f1tenth[5])
    print('Slip Angle:', state_ASF[7], state_f1tenth[6])


def check_dynamics_with_pid(state, inputs):
    # Predict ASF
    model = car_model('ODE:st', True, 1, "utilities/car_files/ini_car_parameters.yml",
                      dt, intermediate_steps=intermediate_steps)

    state_ASF = tf.constant(state[np.newaxis, ...], dtype=tf.float32)
    for next_input in inputs:
        state_ASF = model.step_dynamics(state_ASF, tf.constant(next_input[np.newaxis, ...], dtype=tf.float32), None)
    state_ASF = state_ASF.numpy()[0]

    # Predict F1tenth
    mu = model.car_parameters['mu']  # friction coefficient  [-]
    C_Sf = model.car_parameters['C_Sf']  # cornering stiffness front [1/rad]
    C_Sr = model.car_parameters['C_Sr']  # cornering stiffness rear [1/rad]
    lf = model.car_parameters['lf']  # distance from venter of gracity to front axle [m]
    lr = model.car_parameters['lr']  # distance from venter of gracity to rear axle [m]
    h = model.car_parameters['h']  # center of gravity height of toal mass [m]
    m = model.car_parameters['m']  # Total Mass of car [kg]
    I_car = model.car_parameters['I']  # Moment of inertia for entire mass about z axis  [kgm^2]
    # g = model.car_parameters['g']
    max_sv = model.car_parameters['sv_max']
    max_a = model.car_parameters['a_max']
    max_v = model.car_parameters['v_max']
    min_v = model.car_parameters['v_min']

    state_f1tenth = np.array([pos[0], pos[1], steering_angle, vel, yaw, yaw_dot, slip_angle])
    for next_input in inputs:
        accl, s_v = pid(next_input[1], next_input[0], state_f1tenth[3], state_f1tenth[2], max_sv, max_a, max_v, min_v)
        next_input = [s_v, accl]
        for _ in range(intermediate_steps):
            delta_state_f1tenth = vehicle_dynamics_st(state_f1tenth, next_input, mu, C_Sf, C_Sr, lf, lr, h, m, I_car,
                                                      -0.4189, 0.4189, -3.2, 3.2, 7.319, 9.01, -5.0, 20)
            state_f1tenth = state_f1tenth + (dt / intermediate_steps) * delta_state_f1tenth

    # Print outputs
    state_ASF = state_ASF.round(4)
    state_f1tenth = state_f1tenth.round(4)
    print('      | ASF | f1tenth')
    print('Position x:', state_ASF[5], state_f1tenth[0])
    print('Position y:', state_ASF[6], state_f1tenth[1])
    print('Velocity:', state_ASF[1], state_f1tenth[3])
    print('Yaw:', state_ASF[2], state_f1tenth[4])
    print('Angular Velocity:', state_ASF[0], state_f1tenth[5])
    print('Slip Angle:', state_ASF[7], state_f1tenth[6])
    print('Steering angle:', state_ASF[8], state_f1tenth[2])
    


if __name__ == '__main__':
    # Starting from 
    dt = 0.04
    intermediate_steps = 4
    # Setup state
    pos = (0.0, 0.0)
    steering_angle = -0.00753363
    vel = 0.5406
    yaw = -0.00013362
    yaw_dot = -0.010278
    slip_angle = -0.00753363

    # Setup inputs
    # angular_control = [0.85, 1.0, -0.08, -0.62, -0.75, -1.06]
    # translational_control = [6.51, 14.9, 11.57, 3.18, 2.21, 10.11]
    angular_control = [0.25482783]
    translational_control = [4.691723]

    state = np.array([yaw_dot, vel, yaw, 0, 0, pos[0], pos[1], slip_angle, steering_angle])
    inputs = np.transpose(np.array([angular_control, translational_control], dtype=np.double))

    # check_step_dynamics(state, inputs)
    check_dynamics_with_pid(state, inputs)
