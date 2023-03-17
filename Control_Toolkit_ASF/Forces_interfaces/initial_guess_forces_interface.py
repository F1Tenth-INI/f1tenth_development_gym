################################################
# ForcesPRO requires an initial guess for the
# optimal trajectory. Using a reasonable one
# might help the solver avoid unfeasibility
################################################

import numpy as np

def PD(x, x_f):
    P_gain = 1.0
    D_gain = 0.5
    u = [0.0, 0.0]
    u[0] = P_gain*(x_f[0,6] - x[4])
    u[1] = P_gain*np.linalg.norm(x_f[0,17:19] - x[0:2]) + D_gain*(x_f[0,5] - x[3])
    return u

def no_action(x, x_f):
    return np.array([0.0, 0.0])