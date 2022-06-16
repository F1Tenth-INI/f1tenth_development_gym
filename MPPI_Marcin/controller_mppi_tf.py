import scipy
import numpy as np
from numpy.random import SFC64, Generator
from datetime import datetime
from numba import jit, prange
import tensorflow as tf


from MPPI_Marcin.template_controller import template_controller

import yaml

from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf

from SI_Toolkit.TF.TF_Functions.Compile import Compile
from SI_Toolkit.TF.TF_Functions.Interpolation import interpolate_tf

#load constants from config file
config = yaml.load(open("MPPI_Marcin/config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = 2  # specific to a system

q, phi = None, None
cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_cmd = 'from MPPI_Marcin.cost_functions.'+cost_function+' import q, phi'
exec(cost_function_cmd)

dt = config["controller"]["mppi"]["dt"]
INTERPOLATION_STEP = config["controller"]["mppi"]["INTERPOLATION_STEP"]
mppi_horizon = config["controller"]["mppi"]["mpc_horizon"]
num_rollouts = config["controller"]["mppi"]["num_rollouts"]

cc_weight = config["controller"]["mppi"]["cc_weight"]

NET_NAME = config["controller"]["mppi"]["NET_NAME"]
predictor_type = config["controller"]["mppi"]["predictor_type"]

mppi_samples = int(mppi_horizon / dt)  # Number of steps in MPC horizon

R = tf.convert_to_tensor(config["controller"]["mppi"]["R"])
LBD = config["controller"]["mppi"]["LBD"]
NU = tf.convert_to_tensor(config["controller"]["mppi"]["NU"])
SQRTRHODTINV = tf.convert_to_tensor(config["controller"]["mppi"]["SQRTRHOINV"]) * tf.convert_to_tensor((1 / np.math.sqrt(dt)))
GAMMA = config["controller"]["mppi"]["GAMMA"]
SAMPLING_TYPE = config["controller"]["mppi"]["SAMPLING_TYPE"]

clip_control_input = config["controller"]["mppi"]["CLIP_CONTROL_INPUT"]
if isinstance(clip_control_input[0], list):
    clip_control_input_low = tf.constant(clip_control_input[0], dtype=tf.float32)
    clip_control_input_high = tf.constant(clip_control_input[1], dtype=tf.float32)
else:
    clip_control_input_high = tf.constant(clip_control_input, dtype=tf.float32)
    clip_control_input_low = -clip_control_input_high

#create predictor
predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf(horizon=mppi_samples, dt=dt, intermediate_steps=1, disable_individual_compilation=True)
    predictor_single_trajectory = predictor
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)
    predictor_single_trajectory = predictor
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=num_rollouts, net_name=NET_NAME, disable_individual_compilation=True
    )
    predictor_single_trajectory = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=1, net_name=NET_NAME, disable_individual_compilation=True
    )

GET_ROLLOUTS_FROM_MPPI = True
# GET_ROLLOUTS_FROM_MPPI = False

GET_OPTIMAL_TRAJECTORY = True

def check_dimensions_s(s):
    # Make sure the input is at least 2d
    if tf.rank(s) == 1:
        s = s[tf.newaxis, :]

    return s

#mppi correction
def mppi_correction_cost(u, delta_u):
    return tf.math.reduce_sum(cc_weight * (0.5 * (1 - 1.0 / NU) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2)), axis=-1)

def cost(s_hor ,u, target, u_prev, delta_u):
    '''
    total cost of the trajectory
    @param s_hor: All rollout results (trajectories) for the whole horizon
    @param u: all control inputs for rollouts s_hor
    @param target: (number_of_lidar_points, 2), target point (largest gap) and sensor data
    @param u_prev: (len(control_input)) prevoius control input
    @param delta_u: (batch_size, horizon, len(control_input)) perturbation of previous best control sequence
    '''
    stage_cost = q(s_hor[:,1:,:],u,target, u_prev) # (batch_size,horizon), all costs for every step in the trajectory
    stage_cost = stage_cost + mppi_correction_cost(u, delta_u)
    total_cost = tf.math.reduce_sum(stage_cost,axis=1)  # (batch_size) Ads up the stage costs to the total cost
    total_cost = total_cost + phi(s_hor, target)  # phi is the terminal state cost, which is at the moment the angle to the target at the terminal state
    return total_cost


def reward_weighted_average(S, delta_u):
    '''
    @param S: (batch_size), costs for tracectories
    @param delta_u: (batch_size, horizon, len(control_input)): Perturbation of optimal trajectory to be weighted
    '''
    rho = tf.math.reduce_min(S)
    exp_s = tf.exp(-1.0/LBD * (S-rho))
    a = tf.math.reduce_sum(exp_s)
    b = tf.math.reduce_sum(exp_s[:, tf.newaxis, tf.newaxis]*delta_u, axis=0)/a
    return b

def inizialize_pertubation(random_gen, stdev = SQRTRHODTINV, sampling_type = SAMPLING_TYPE, interpolation_step=INTERPOLATION_STEP):
    if sampling_type == "interpolated":
        independent_samples = int(tf.math.ceil(mppi_samples / interpolation_step)) + 1
        delta_u = random_gen.normal([num_rollouts, independent_samples, num_control_inputs], dtype=tf.float32) * stdev
        interp = interpolate_tf(delta_u, interpolation_step, axis=1)
        delta_u = interp[:, :mppi_samples, :]
    else:
        delta_u = random_gen.normal([num_rollouts, mppi_samples, num_control_inputs], dtype=tf.float32) * stdev
    return delta_u



#cem class
class controller_mppi_tf(template_controller):
    def __init__(self):
        #First configure random sampler
        SEED = config["controller"]["mppi"]["SEED"]
        if SEED == "None":
            SEED = int((datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0)
        self.rng_cem = tf.random.Generator.from_seed(SEED)

        self.u_nom = tf.ones([1, mppi_samples, num_control_inputs], dtype=tf.float32)*tf.constant([6.0, 0.0], dtype=tf.float32)
        self.u = tf.convert_to_tensor([6.0, 0.0], dtype=tf.float32)

        self.rollout_trajectory = None
        self.traj_cost = None

        self.optimal_trajectory = None

        # Defining function - the compiled part must not have if-else statements with changing output dimensions
        if predictor_type ==  'NeuralNet':
            self.update_internal_state = self.update_internal_state_of_RNN
        else:
            self.update_internal_state = lambda s, u_nom: ...

        if GET_ROLLOUTS_FROM_MPPI:
            self.mppi_output = self.return_all
        else:
            self.mppi_output = self.return_restricted

    def return_all(self, u, u_nom, rollout_trajectory, traj_cost):
        return u, u_nom, rollout_trajectory, traj_cost

    def return_restricted(self, u, u_nom, rollout_trajectory, traj_cost):
        return u, u_nom, None, None

    @Compile
    def predict_and_cost(self, s, target, u_nom, random_gen, u_old):
        """
        Part of MPPI which can be XLS (with Tensorflow) compiled.
        @param: s: current state of the car
        @param: target: Target position of the car and lidat scans stacked on each other
        @param: u_nom: Last optimal control sequence (Array of control inputs)
        @param: random_gen: Tensoflow random generator
        @param: u_old: Last optimal control input
        """
        s = tf.tile(s, tf.constant([num_rollouts, 1]))
        u_nom = tf.concat([u_nom[:, 1:, :], u_nom[:, -1, tf.newaxis, :]], axis=1)
        delta_u = inizialize_pertubation(random_gen)  #(batch_size, horizon, len(control_input)) perturbation of the last control input for rollouts
        u_run = tf.tile(u_nom, [num_rollouts, 1, 1])+delta_u  #(batch_size, horizon, len(control_input)) Control inputs for MPPI rollouts (last optimal + perturbation)
        u_run = tf.clip_by_value(u_run, clip_control_input_low, clip_control_input_high)  # (batch_size, horizon, len(control_input)) Clip control input based on system parameters
        rollout_trajectory = predictor.predict_tf(s, u_run)
        traj_cost = cost(rollout_trajectory, u_run, target, u_old, delta_u)  # (batch_size,) Cost for each trajectory
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), clip_control_input_low, clip_control_input_high)  # (1, horizon, len(control_input)) Find optimal control sequence by weighted average of trajectory costs and clip the result
        u = u_nom[0, 0, :]  # (number of control inputs e.g. 2 for speed and steering,) Returns only the first step of the optimal control sequence
        self.update_internal_state(s, u_nom)
        return self.mppi_output(u, u_nom, rollout_trajectory, traj_cost)

    @Compile
    def predict_optimal_trajectory(self, s, u_nom):
        optimal_trajectory = predictor_single_trajectory.predict_tf(s, u_nom)
        if predictor_type ==  'NeuralNet':
            predictor_single_trajectory.update_internal_state_tf(s=s, Q0=u_nom[:, :1, :])
        return optimal_trajectory

    #step function to find control
    def step(self, s: np.ndarray, target: np.ndarray, time=None):
        """
        Execute one full step of the MPPI contol based on the available information about car state and returns the control input
        @param: s: current state of the car [x,y,theta]
        @param: target: Target state of the car and lidar scans stacked to form one matrix
        @param: time:
        """
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        s = check_dimensions_s(s)
        target = tf.convert_to_tensor(target, dtype=tf.float32)

        self.u, self.u_nom, rollout_trajectory, traj_cost = self.predict_and_cost(s, target, self.u_nom, self.rng_cem,
                                                                                  self.u)
        if GET_ROLLOUTS_FROM_MPPI:
            self.rollout_trajectory = rollout_trajectory.numpy()
            self.traj_cost = traj_cost.numpy()

        if GET_OPTIMAL_TRAJECTORY:
            self.optimal_trajectory = self.predict_optimal_trajectory(s, self.u_nom).numpy()

        return self.u.numpy()

    def controller_reset(self):
        self.u_nom = tf.zeros([1, mppi_samples, num_control_inputs], dtype=tf.float32)
        self.u = 0.0