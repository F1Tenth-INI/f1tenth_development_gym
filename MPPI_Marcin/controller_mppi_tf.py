import scipy
import numpy as np
from numpy.random import SFC64, Generator
from datetime import datetime
from numba import jit, prange
import tensorflow as tf
import tensorflow_probability as tfp

from MPPI_Marcin.template_controller import template_controller

import yaml

from  SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
from  SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
from  SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf

from  SI_Toolkit.TF.TF_Functions.Compile import Compile

#load constants from config file
config = yaml.load(open("MPPI_Marcin/config.yml", "r"), Loader=yaml.FullLoader)

num_control_inputs = 2  # specific to a system

q, phi = None, None
cost_function = config["controller"]["general"]["cost_function"]
cost_function = cost_function.replace('-', '_')
cost_function_cmd = 'from MPPI_Marcin.cost_functions.'+cost_function+' import q, phi'
exec(cost_function_cmd)

dt = config["controller"]["mppi"]["dt"]
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

clip_control_input = tf.constant(config["controller"]["mppi"]["CLIP_CONTROL_INPUT"], dtype=tf.float32)

#create predictor
predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)

"""Define Predictor"""
if predictor_type == "EulerTF":
    predictor = predictor_ODE_tf(horizon=mppi_samples, dt=dt, intermediate_steps=10, disable_individual_compilation=True)
elif predictor_type == "Euler":
    predictor = predictor_ODE(horizon=mppi_samples, dt=dt, intermediate_steps=10)
elif predictor_type == "NeuralNet":
    predictor = predictor_autoregressive_tf(
        horizon=mppi_samples, batch_size=num_rollouts, net_name=NET_NAME
    )

GET_ROLLOUTS_FROM_MPPI = True
# GET_ROLLOUTS_FROM_MPPI = False

GET_OPTIMAL_TRAJECTORY = True

#mppi correction
def mppi_correction_cost(u, delta_u):
    return tf.math.reduce_sum(cc_weight * (0.5 * (1 - 1.0 / NU) * R * (delta_u ** 2) + R * u * delta_u + 0.5 * R * (u ** 2)), axis=-1)

def cost(s_hor ,u, target, u_prev, delta_u):
    '''
    total cost of the trajectory
    @param s_hor: All rollout results (trajectories) for the whole horizon
    @param u: all control inputs for rollouts s_hor
    @param target: (109, 2), target point (largest gap) and sensor data
    @param u_prev: (2,) prevoius control input 
    @param delta_u: (2000, 10, 2) perturbation of previous best control sequence
    '''
    stage_cost = q(s_hor[:,1:,:],u,target, u_prev) # (2000,10), all costs for every step in the trajectory
    stage_cost = stage_cost + mppi_correction_cost(u, delta_u)
    total_cost = tf.math.reduce_sum(stage_cost,axis=1) # (2000) Ads up the stage costs to the total cost
    total_cost = total_cost + phi(s_hor, target) # phi is the terminal state cost, which is at the moment the angle to the target at the terminal state
    # print(stage_cost.numpy())
    return total_cost


def reward_weighted_average(S, delta_u):
    '''
    @param S: (2000), costs for tracectories
    @param delta_u: (2000, 10, 2): Perturbation of optimal trajectory to be weighted 
    '''
    rho = tf.math.reduce_min(S)
    exp_s = tf.exp(-1.0/LBD * (S-rho))
    a = tf.math.reduce_sum(exp_s)
    b = tf.math.reduce_sum(exp_s[:, tf.newaxis, tf.newaxis]*delta_u, axis=0)/a
    return b

def inizialize_pertubation(random_gen, stdev = SQRTRHODTINV, sampling_type = SAMPLING_TYPE):
    if sampling_type == "interpolated":
        step = 10
        range_stop = int(tf.math.ceil(mppi_samples / step)*step) + 1
        t = tf.range(range_stop, delta = step)
        t_interp = tf.cast(tf.range(range_stop), tf.float32)
        delta_u = random_gen.normal([num_rollouts, t.shape[0], num_control_inputs], dtype=tf.float32) * stdev
        interp = tfp.math.interp_regular_1d_grid(t_interp, t_interp[0], t_interp[-1], delta_u, axis=1)
        delta_u = interp[:,:mppi_samples, :]
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
        #Random generator (from Tensorflow)
        self.rng_cem = tf.random.Generator.from_seed(SEED)
        
        #Last control input ?
        self.u_nom = tf.ones([1, mppi_samples, num_control_inputs], dtype=tf.float32)*tf.constant([6.0, 0.0], dtype=tf.float32)
        self.u = tf.convert_to_tensor([6.0, 0.0], dtype=tf.float32)

        self.rollout_trajectory = None
        self.traj_cost = None

        self.optimal_trajectory = None

    # @Compile
    def predict_and_cost(self, s, target, u_nom, random_gen, u_old):
        """
        Generate random input sequence and clip to control limits
        @param: s: current state of the car [x,y,theta]
        @param: target: Target state of the car and lidat scans stacked on each other
        @param: u_nom: Last optimal control sequence (Array of control inputs)
        @param: random_gen: Tensoflow random generator 
        @param: u_old: Last optimal control input
        """
        delta_u = inizialize_pertubation(random_gen) #(2000, 10, 2) perturbation of the last control input for rollouts
        u_run = tf.tile(u_nom, [num_rollouts, 1, 1])+delta_u #(2000, 10, 2) Hostiry based control inputs for rollouts (last optimal + perturbation)
        u_run = tf.clip_by_value(u_run, -clip_control_input, clip_control_input) # (2000, 10, 2) Clip control input based on model parameters
        rollout_trajectory = predictor.predict_tf(s, u_run) # (2000, 11, 3) All trajectories for the state distribution
        traj_cost = cost(rollout_trajectory, u_run, target, u_old, delta_u)  # (2000,) Cost for each trajectory
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -clip_control_input, clip_control_input) # (1, 10, 2) Find optimal control sequence by weighted average of trajectory costs
        u = u_nom[0, 0, :] # (2,) Return only the first step of the optimal control sequence
        u_nom = tf.concat([u_nom[:, 1:, :], u_nom[:, -1, tf.newaxis, :]], axis=1)
        if GET_ROLLOUTS_FROM_MPPI:
            return u, u_nom, rollout_trajectory, traj_cost
        else:
            return u, u_nom, None, None

    @Compile
    def predict_optimal_trajectory(self, s, u_nom):
        return predictor.predict_tf(s, u_nom)

    #step function to find control
    def step(self, s: np.ndarray, target: np.ndarray, time=None):
        """
        Execute one full step of the MPPI contol based on the sensor measurements and returns the control input
        @param: s: current state of the car [x,y,theta]
        @param: target: Target state of the car and lidat scans stacked on each other
        @param: time: 
        """
        s = np.tile(s, tf.constant([num_rollouts, 1]))
        s = tf.convert_to_tensor(s, dtype=tf.float32)
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