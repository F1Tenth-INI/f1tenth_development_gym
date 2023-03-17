import casadi
import yaml
import os
from Control_Toolkit_ASF.Cost_Functions.Car.racing_forces import racing_forces
from SI_Toolkit.computation_library import NumpyLibrary
"""
Forces requires a function of the cost in the form
objective = f(z,p)
to derive equality constraints
"""

config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)

def f1tenth_stage_cost(z, p):
    racing_forces_instance = racing_forces(None, NumpyLibrary)
    return racing_forces_instance.get_stage_cost(z[2:], z[:2], p)

def f1tenth_distance_to_waypoint(z, p):
    v = z[2:4] - p[17:19]
    return v.T @ v + 100*z[7]

