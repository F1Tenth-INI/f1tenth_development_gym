from typing import List
import yaml
import os
import numpy as np

from utilities.Settings import Settings

class VehicleParameters:
    mu: float # Surface friction
    C_Sf: float # Front tire cornering stiffness [N/rad]
    C_Sr: float # Rear tire cornering stiffness [N/rad]
    lf: float # Distance from the center of mass to the front axle [m]
    lr: float # Distance from the center of mass to the rear axle [m]
    l_wb: float # Wheelbase [m]
    h: float # Height of the center of mass [m]
    m: float  # mass of the car [kg]
    I_z: float # Moment of inertia around the z-axis [kg*m^2]
    g: float # Gravitational acceleration [m/s^2]
    width: float # Width of the car [m]
    length: float # Length of the car [m]
    s_min: float # Minimum steering angle [rad]
    s_max: float # Maximum steering angle [rad]
    sv_min: float # Minimum steering velocity [rad/s]
    sv_max: float # Maximum steering velocity [rad/s]
    a_max: float # Maximum acceleration [m/s^2]
    a_min: float # Minimum acceleration [m/s^2]
    v_min: float # Minimum velocity [m/s]
    v_max: float # Maximum velocity [m/s]
    v_switch: float # Switching velocity [m/s]: From here accelerating is harder
    servo_p: float # Servo proportional gain
    steering_diff_low: float
    min_speed_st: float
    
    # Pacejka parameters
    C_0d: float 
    C_Pf: List[float] # Pacejka parameters for the front tires [B, C, D, E]
    C_Pr: List[float] # Pacejka parameters for the rear tires [B, C, D, E]


    """
    Initializes a new instance of the CarParameters class.

    This method sets the class variables based on the parameters
    defined in the specified YAML file. It also allows for the overwriting
    of the surface friction value if it is specified in the Settings.

    :param param_file_name: The name of the YAML file containing car parameters.
                            Defaults to 'gym_car_parameters.yaml'.
    """
    def __init__(self, param_file_name = 'gym_car_parameters.yml'):
      
        class_variable_names = list(VehicleParameters.__annotations__.keys())
        current_dir = os.path.dirname(__file__)
        yaml_file_path = os.path.join(current_dir, param_file_name)

        with open(yaml_file_path, 'r') as file:
          params = yaml.safe_load(file)
          for class_variable_name in class_variable_names: 
            if class_variable_name not in params:
              raise ValueError(f"Parameter '{class_variable_name}' not found in the YAML file.")
            setattr(self, class_variable_name, params[class_variable_name])
        
        # Overwrite Sufrace friction
        if Settings.SURFACE_FRICTION is not None:
            self.mu = Settings.SURFACE_FRICTION


    def to_np_array(self):
        return np.array([
            # Simulator engine Car parameters
            self.mu,  # mu (friction coefficient)
            self.lf,  # lf (distance from center of gravity to front axle)
            self.lr,  # lr (distance from center of gravity to rear axle)
            self.h,  # h_cg (center of gravity height of sprung mass)
            self.m,  # m (Total Mass of car)
            self.I_z,  # I_z (Moment of inertia about z-axis)
            self.g,  # g (Gravitation Constant)
            
            # Pacejka Magic Formula Parameters (Front Tire)
            self.C_Pf[0],  # B_f
            self.C_Pf[1],  # C_f
            self.C_Pf[2],  # D_f
            self.C_Pf[3],  # E_f

            # Pacejka Magic Formula Parameters (Rear Tire)
            self.C_Pr[0],  # B_r
            self.C_Pr[1],  # C_r
            self.C_Pr[2],  # D_r
            self.C_Pr[3],  # E_r

            # Steering Constraints
            self.servo_p,  # servo_p (proportional factor of servo PID)
            self.s_min,  # s_min (min steering angle)
            self.s_max,  # s_max (max steering angle)
            self.sv_min,  # sv_min (min steering velocity)
            self.sv_max,  # sv_max (max steering velocity)

            # Acceleration Constraints
            self.a_min,  # a_min (min acceleration)
            self.a_max,  # a_max (max acceleration)
            self.v_min,  # v_min (min velocity)
            self.v_max,  # v_max (max velocity)
            self.v_switch,  # v_switch (velocity threshold for model transition)
        ], dtype=np.float32)