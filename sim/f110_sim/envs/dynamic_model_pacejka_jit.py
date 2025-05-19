import numpy as np
from numba import njit, prange


class StateIndices:
    yaw_rate = 0
    v_x = 1
    v_y = 2
    yaw_angle = 3
    yaw_angle_cos = 4
    yaw_angle_sin = 5
    pose_x = 6
    pose_y = 7
    slip_angle = 8
    steering_angle = 9


    number_of_states = 10
    
    
    
class ControlIndices:
    desired_steering_angle = 0
    acceleration = 1





@njit(fastmath=True)
def car_dynamics_pacejka_jit(s, Q, car_params, t_step):
    """Advance car dynamics using optimized Pacejka model with JIT, fully matching batch version.
    
     Args:
        s (np.ndarray): Car state as deinfed in state_utilities,
        
        Q (np.ndarray): Control input vector with the following elements:
            - Q[0]: desired_steering_angle (desired steering angle input)
            - Q[1]: acceleration (translational control input)

        car_params (np.ndarray): Array of car parameters as defined in VehicleParameters.to_np_array().

        t_step (float): Time step for integration.

    Returns:
        np.ndarray: updated carstate
          
    """
    
    # Unpack car parameters from NumPy array
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
    servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch = car_params

    # Unpack state
    psi_dot, v_x, v_y ,psi, _,  _,s_x, s_y,  _, delta= s

    # Unpack control inputs
    desired_steering_angle, translational_control = Q

    # Apply Servo PID Control for Steering (Matches Batch Model)
    steering_angle_difference = desired_steering_angle - delta
    delta_dot = (steering_angle_difference * servo_p) if abs(steering_angle_difference) > 0.0001 else 0.0
    delta_dot = max(min(delta_dot, sv_max), sv_min)  # Apply steering velocity constraints

    # Apply Motor PID Control for Acceleration (Matches Batch Model)
  
    v_x_dot = translational_control
        
    # v_x_dot = max(min(v_x_dot, a_max), a_min)  # Apply acceleration constraints
    # Limit due to velocity (motor power)
    if v_x > v_switch:
        pos_limit = a_max * v_switch / v_x
    else:
        pos_limit = a_max
    v_x_dot = min(v_x_dot, pos_limit)

    # Limit due to tire friction
    max_a_friction = mu * g_
    v_x_dot = min(max(v_x_dot, -max_a_friction), max_a_friction)


    # Euler integration with optimized calculations
    for _ in range(1):  # Keep intermediate_steps = 1 for simplicity
        # v_x = max(v_x, 1e-5)  # Prevent division by zero
        if v_x == 0:
            v_x = 1e-5

        alpha_f = -np.arctan((v_y + psi_dot * lf) / v_x) + delta
        alpha_r = -np.arctan((v_y - psi_dot * lr) / v_x)

        # Compute vertical tire forces
        F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
        F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

        # Compute lateral forces using Pacejka's formula
        F_yf = mu * F_zf * D_f * np.sin(C_f * np.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - np.arctan(B_f * alpha_f))))
        F_yr = mu * F_zr * D_r * np.sin(C_r * np.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - np.arctan(B_r * alpha_r))))
                
        # Compute state derivatives
        d_s_x = v_x * np.cos(psi) - v_y * np.sin(psi)
        d_s_y = v_x * np.sin(psi) + v_y * np.cos(psi)
        d_psi = psi_dot
        d_v_x = v_x_dot
        d_v_y = (F_yr + F_yf) / m - v_x * psi_dot
        d_psi_dot = (-lr * F_yr + lf * F_yf) / I_z

        # Integrate using Euler's method
        s_x += t_step * d_s_x
        s_y += t_step * d_s_y
        delta = max(min(delta + t_step * delta_dot, s_max), s_min)
        v_x += t_step * d_v_x
        v_y += t_step * d_v_y
        psi += t_step * d_psi
        psi_dot += t_step * d_psi_dot


    # Kinematic-to-Pacejka Blending (Restored from Batch Model)
    low_speed_threshold, high_speed_threshold = 0.5, 3.0
    weight = (v_x - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
    weight = max(min(weight, 1), 0)  # Ensure within [0,1]

    # Simple kinematic model for low speeds
    s_x_ks = s_x + t_step * (v_x * np.cos(psi))
    s_y_ks = s_y + t_step * (v_x * np.sin(psi))
    psi_ks = psi + t_step * (v_x / lf * np.tan(delta))
    v_y_ks = 0  # No lateral velocity in kinematic model

    # Weighted interpolation
    s_x = (1 - weight) * s_x_ks + weight * s_x
    s_y = (1 - weight) * s_y_ks + weight * s_y
    psi = (1 - weight) * psi_ks + weight * psi
    v_y = (1 - weight) * v_y_ks + weight * v_y

    psi_sin = np.sin(psi)
    psi_cos = np.cos(psi)
    
    # Return the updated state (8 elements)
    return np.array([psi_dot, v_x, v_y,psi,psi_cos, psi_sin, s_x ,s_y ,  0, delta], dtype=np.float32)



@njit(fastmath=True)
def car_steps_sequential(s, Q_sequence, car_params, t_step, num_steps):
    """
    Runs car_step for single car sequentially.

    Inputs:
    - states: (10) array of states
    - Qs: (2, H) Sequence of H Controlls applied to the car
    - car_params: (fixed-size) array of car parameters
    - t_step: time step
    - num_steps: number of steps to run

    Output:
    - state_trajectory: Trajetory of states during apng aplying the H controlls
    """
    
    state_trajectory = np.zeros((num_steps, 10), dtype=np.float32)
    
    for i in range(num_steps):
        s = car_dynamics_pacejka_jit(s, Q_sequence[i], car_params, t_step)
        state_trajectory[i] = s
    return state_trajectory

        
        

@njit(parallel=True, fastmath=True)
def car_step_parallel(states, Qs, car_params, t_step):
    """
    Runs car_step for multiple cars in parallel using prange (multi-core CPU execution).
    
    Inputs:
    - states: (N, 7) array where N is number of cars
    - Qs: (N, 2) array where N is number of cars
    - car_params: (fixed-size) array of car parameters
    - t_step: time step

    Output:
    - Updated states array (N, 8)
    """
    num_cars = states.shape[0]  # Number of cars
    new_states = np.empty_like(states)  # Store updated states

    # Run each car's dynamics in parallel
    for i in prange(num_cars):  # 🔥 Parallel execution across cars
        new_states[i] = car_dynamics_pacejka_jit(states[i], Qs[i], car_params, t_step)

    return new_states  # Return all updated states
