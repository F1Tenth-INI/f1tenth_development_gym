import numpy as np
import casadi
from f110_gym.envs.dynamic_models import vehicle_dynamics_st

"""
Forces requires a function of the dynamics in the form
dx/dt = f(x,u,p)
to derive equality constraints
"""


def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    steering_velocity = \
        casadi.if_else(
            casadi.logic_or(casadi.logic_and(steering_angle <= s_min,
                                             steering_velocity <= 0),
                            casadi.logic_and(steering_angle >= s_max,
                                             steering_velocity >= 0)),
            0,
            casadi.if_else(steering_velocity <= sv_min,
                           sv_min,
                           casadi.if_else(steering_velocity >= sv_max,
                                          sv_max,
                                          steering_velocity))
    )

    return steering_velocity

def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    pos_limit = casadi.if_else(vel > v_switch, a_max * v_switch / vel, a_max)

    # accl limit reached?
    accl = casadi.if_else(casadi.logic_or(casadi.logic_and(vel <= v_min, accl <= 0),
                                          casadi.logic_and(vel >= v_max, accl >= 0)), 0,
                          casadi.if_else(accl <= -a_max, -a_max,
                                         casadi.if_else(accl >= pos_limit, pos_limit, accl)))

    return accl

def vehicle_dynamics_ks(x, u_unclipped, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min,
                        v_max):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u0: steering angle velocity of front wheels
                u1: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # # constraints
    # u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])
    u = casadi.SX((2,1))
    u[0] = steering_constraint(x[2], u_unclipped[0], s_min, s_max, sv_min, sv_max)
    u[1] = accl_constraints(x[3], u_unclipped[1], v_switch, a_max, v_min, v_max)

    # system dynamics
    f = casadi.vcat([x[3] * np.cos(x[4]),
                     x[3] * np.sin(x[4]),
                     u[0],
                     u[1],
                     x[3] / lwb * np.tan(x[2])])
    return f


def vehicle_dynamics_st_forces(x, u_unclipped, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max,
                               v_min,
                               v_max):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: velocity in x direction
                x4: yaw angle
                x5: yaw rate
                x6: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u0: steering angle velocity of front wheels
                u1: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # # constraints
    # u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])
    u = casadi.SX((2, 1))
    u[0] = steering_constraint(x[2], u_unclipped[0], s_min, s_max, sv_min, sv_max)
    u[1] = accl_constraints(x[3], u_unclipped[1], v_switch, a_max, v_min, v_max)

    # kinematic model for small velocities
    # wheelbase
    lwb = lf + lr

    # system dynamics
    x_ks = x[0:5]
    f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max,
                               v_min, v_max)
    f_small_vels = casadi.vcat(
        (f_ks, casadi.vcat([u[1] / lwb * np.tan(x[2]) + x[3] / (lwb * np.cos(x[2]) ** 2) * u[0], 0])))

    # kinematic model for big velocities
    # system dynamics
    f = casadi.vcat([x[3] * np.cos(x[6] + x[4]),
                     x[3] * np.sin(x[6] + x[4]),
                     u[0],
                     u[1],
                     x[5],
                     -mu * m / (x[3] * I * (lr + lf)) * (
                             lf ** 2 * C_Sf * (g * lr - u[1] * h) + lr ** 2 * C_Sr * (g * lf + u[1] * h)) * x[5] \
                     + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h)) *
                     x[6] \
                     + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u[1] * h) * x[2],
                     (mu / (x[3] ** 2 * (lr + lf)) * (
                             C_Sr * (g * lf + u[1] * h) * lr - C_Sf * (g * lr - u[1] * h) * lf) - 1) * x[5] \
                     - mu / (x[3] * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h)) * x[6] \
                     + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * x[2]])

    f = casadi.if_else(casadi.fabs(x[3]) < 0.1, f_small_vels, f)
    return f

params = {
    'mu': 1.0489,  # friction coefficient  [-]
    'C_Sf': 4.718,  # cornering stiffness front [1/rad]
    'C_Sr': 5.4562,  # cornering stiffness rear [1/rad]
    'lf': 0.15875,  # distance from venter of gracity to front axle [m]
    'lr': 0.17145,  # distance from venter of gracity to rear axle [m]
    'h': 0.074,  # center of gravity height of toal mass [m]
    'm': 3.74,  # Total Mass of car [kg]
    'I': 0.04712,  # Moment of inertia for entire mass about z axis  [kgm^2]
    's_min': -0.4189,  # Min steering angle [rad]
    's_max': 0.4189,  # Max steering angle [rad]
    'sv_min': -3.2,  # Min steering velocity [rad/s]
    'sv_max': 3.2,  # Max steering velocity [rad/s]
    'v_switch': 7.319,  # switching velocity [m/s]
    'a_max': 9.51,  # Max acceleration [m/s^2]
    'v_min': -5.0,  # Min velocity [m/s]
    'v_max': 20.0,  # Max velocity [m/s]
    # 'width': 0.31,  # Width of car [m]
    # 'length': 0.58  # Length of car [m]
}


def f1tenth_dynamics(s, u, p):
    sD = vehicle_dynamics_st_forces(
        s,
        u,
        **params)
    return sD


# dynamics of the environment only for DEBUG purposes
# should be equal to the one in this file
def Car_env(s, u, p):
    sD = vehicle_dynamics_st(
        s,
        u,
        **params)
    return sD

def casadi_to_numpy(sx):
    return np.array([float(sx[i]) for i in range(sx.shape[0])])

def test_dynamics(M: int):
    for i in range(M):
        s = (np.random.random_sample((7,))-0.5)*30
        u = (np.random.random_sample((2,))-0.5)*10

        sD_env = Car_env(s, u, None)
        sD_forces = casadi_to_numpy(f1tenth_dynamics(s, u, None))
        error = np.linalg.norm(sD_env - sD_forces)
        pass
    return


if __name__ == '__main__':
    test_dynamics(200)
