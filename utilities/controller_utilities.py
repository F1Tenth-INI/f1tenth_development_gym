from utilities.Settings import Settings
from utilities.car_files.vehicle_parameters import VehicleParameters
from numba import jit

class ControllerUtilities:
    def __init__(self):
        self.vehicle_parameters = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
        
    def motor_pid(self, desired_speed, current_speed):
        
        a_max = self.vehicle_parameters.a_max
        v_max = self.vehicle_parameters.v_max
        v_min = self.vehicle_parameters.v_min
        
        return motor_pid_with_speed_difference(desired_speed, current_speed, a_max, v_max, v_min)


@jit
def motor_pid_with_speed_difference(translational_control, v_x, a_max, v_max, v_min):
    """
    PID controller implementation using speed difference logic.
    
    Args:
        translational_control (float): Desired translational control speed.
        v_x (float): Current velocity.
        a_max (float): Maximum acceleration.
        v_max (float): Maximum velocity.
        v_min (float): Minimum velocity (negative for reverse).
    
    Returns:
        float: PID control output (v_x_dot).
    """
    # Speed difference
    speed_difference = translational_control - v_x

    p_gain = 3.0
    if v_x > 0:  # Forward
        v_x_dot =p_gain * (a_max / v_max * speed_difference) if speed_difference > 0 else p_gain * (a_max / (-v_min) * speed_difference)
    else:  # Backward
        v_x_dot = 2.0 * (a_max / v_max * speed_difference) if speed_difference > 0 else 2.0 * (a_max / (-v_min) * speed_difference)

    return v_x_dot