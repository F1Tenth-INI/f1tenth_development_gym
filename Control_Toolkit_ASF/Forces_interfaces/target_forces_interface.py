################################################
# ForcesPRO requires the cost function to be
# fixed at compile time. In order to change
# target at runtime it is necessary to provide it
# as parameter
################################################

import numpy as np

def f1tenth_target(controller, parameters_map):
    target = np.hstack((parameters_map['previous_input'], controller.next_waypoints.numpy().flatten()))[np.newaxis,:]
    return target

