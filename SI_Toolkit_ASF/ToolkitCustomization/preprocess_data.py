import numpy as np


def calculate_pid(df, state_variables_pid_function, control_inputs_pid_function, ode_model_of_car_dynamics, env_car_parameter_file, timestep_sim, **kwargs):

    from SI_Toolkit_ASF.car_model import car_model
    from SI_Toolkit.computation_library import NumpyLibrary

    n = len(df)

    # Vectorised state and control arrays
    s = df[state_variables_pid_function].to_numpy(dtype=np.float32)
    u_des = df[control_inputs_pid_function].to_numpy(dtype=np.float32)

    # Batch-size car model
    cm = car_model(
        model_of_car_dynamics=ode_model_of_car_dynamics,
        batch_size=n,
        car_parameter_file=env_car_parameter_file,
        dt=timestep_sim,
        intermediate_steps=1,
        computation_lib=NumpyLibrary(),
        **kwargs
    )

    # PID + constraints in batch
    u_pid = cm.pid(s, u_des)
    u_pid_constrained = cm.apply_constrains(s, u_pid)  # shape (n, 2)

    # Extract command components directly
    steering_speed_cmd = u_pid_constrained[:, 0]
    acceleration_x_cmd = u_pid_constrained[:, 1]

    # Augment DataFrame
    df_processed = df.copy()
    df_processed['angular_control_pid_constr'] = steering_speed_cmd
    df_processed['translational_control_pid_constr'] = acceleration_x_cmd

    return df_processed