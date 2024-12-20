from typing import List
import yaml
import os

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
        if Settings.SURFACE_FRICITON is not None:
            self.mu = Settings.SURFACE_FRICITON

  