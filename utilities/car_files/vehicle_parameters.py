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
    I_z: float
    g: float
    width: float
    length: float
    s_min: float
    s_max: float
    sv_min: float
    sv_max: float
    a_max: float
    a_min: float
    v_min: float
    v_max: float
    v_switch: float
    servo_p: float
    steering_diff_low: float
    min_speed_st: float
    
    # Pacejka parameters
    C_0d: float
    C_Pf: List[float]
    C_Pr: List[float]


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

  