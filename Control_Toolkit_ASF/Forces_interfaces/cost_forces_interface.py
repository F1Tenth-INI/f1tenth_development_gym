import casadi
import yaml
import os
from Control_Toolkit_ASF.Cost_Functions.Car.racing_forces import racing_forces
"""
Forces requires a function of the cost in the form
objective = f(z,p)
to derive equality constraints
"""

config = yaml.load(
    open(os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"), "r"),
    Loader=yaml.FullLoader,
)

def f1tenth(z, p):
    return racing_forces.get_stage_cost(z[2:], z[:2])