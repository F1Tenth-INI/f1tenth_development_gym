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
control_interpolation_steps = config["controller"]["mppi"]["control_interpolation_steps"]
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
    predictor = predictor_ODE_tf(horizon=mppi_samples, dt=dt, intermediate_steps=1, disable_individual_compilation=True)
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
    @param target: (number_of_lidar_points, 2), target point (largest gap) and sensor data
    @param u_prev: (len(control_input)) prevoius control input 
    @param delta_u: (batch_size, horizon, len(control_input)) perturbation of previous best control sequence
    '''
    stage_cost = q(s_hor[:,1:,:],u,target, u_prev) # (batch_size,horizon), all costs for every step in the trajectory
    stage_cost = stage_cost + mppi_correction_cost(u, delta_u)
    
    total_cost = tf.math.reduce_sum(stage_cost,axis=1) # (batch_size) Ads up the stage costs to the total cost
    
    total_cost = total_cost + phi(s_hor, target) # phi is the terminal state cost, which is at the moment the angle to the target at the terminal state
    # print(stage_cost.numpy())
    # sc = stage_cost.numpy()[:10]
    # tc1 = total_cost.numpy()[:10]
    # tc2 = total_cost.numpy()[:10]
    # print("Tc", tc)
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




def interpolate_mean(a, number_of_steps):
    '''Interpolate control inputs
    @param a: list of control inputs (number_of_rollouts, horizon, len(control_input))
    @param number_of_steps: number of steps between each control input 
    
    We want to increase the resolution of the rollouts without the need of a larger batch size:
    For every consecutive control input, we calculate number_of_steps values in between (linear interpolation), with the following trick:
    
    For example we take number_of_steps = 3 and a control input of [1,2,3,4,5]
    We calculate the interpolation by adding shifted repeated versions of the control input
    
        1 1 1 2 2 2 3 3 3 4 4 4 5 
    +   1 1 2 2 2 3 3 3 4 4 4 5 5 
    +   1 2 2 2 3 3 3 4 4 4 5 5 5
    / 3
    =   1.0 1.333 1.666 2.0 ....        ...4.666 5.0
    
    returns result: list of control inputs (number_of_rollouts, horizon * number_of_steps,  len(control_input))
    '''


    index_steps = number_of_steps - 1

    a = tf.repeat(a, number_of_steps, axis=1)
    interpolation = tf.zeros([tf.shape(a)[0], tf.shape(a)[1] - index_steps, tf.shape(a)[2]], dtype= tf.float32)

    i = tf.constant(0)
    while_condition = lambda i, interpolation, a: tf.less(i, number_of_steps)
    def body(i, interpolation, a):
        lenght = tf.shape(a)[1] - index_steps
        a_sliced = tf.slice(a, [0, i, 0], [a.shape[0], lenght ,a.shape[2]])
        interpolation = tf.add(interpolation, a_sliced)
        return [tf.add(i, 1), interpolation, a]

    # do the loop:
    index, result, a = tf.while_loop(while_condition, body, [i, interpolation, a])
    result = result/number_of_steps
    
    return result



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

    @Compile
    def predict_and_cost(self, s, target, u_nom, random_gen, u_old):
        """
        Generate random input sequence and clip to control limits
        @param: s: current state of the car [x,y,theta]
        @param: target: Target state of the car and lidat scans stacked on each other
        @param: u_nom: Last optimal control sequence (Array of control inputs)
        @param: random_gen: Tensoflow random generator 
        @param: u_old: Last optimal control input
        """
        delta_u = inizialize_pertubation(random_gen) #(batch_size, horizon, len(control_input)) perturbation of the last control input for rollouts
        u_run = tf.tile(u_nom, [num_rollouts, 1, 1])+delta_u #(batch_size, horizon, len(control_input)) Hostiry based control inputs for rollouts (last optimal + perturbation)
        u_run = tf.clip_by_value(u_run, -clip_control_input, clip_control_input) # (batch_size, horizon, len(control_input)) Clip control input based on model parameters
        u_run = interpolate_mean(u_run, control_interpolation_steps) # Increase resolution for control inputs (same horizon but smaller timestep)
        delta_u_ext = tf.repeat(delta_u, control_interpolation_steps, axis=1)[:, :-1, :] # Fit dimensions with interpolated version
        
        rollout_trajectory = predictor.predict_tf(s, u_run) # (batch_size, 11, 3) All trajectories for the state distribution
        traj_cost = cost(rollout_trajectory, u_run, target, u_old, delta_u_ext)  # (batch_size,) Cost for each trajectory
        u_nom = tf.clip_by_value(u_nom + reward_weighted_average(traj_cost, delta_u), -clip_control_input, clip_control_input) # (1, horizon, len(control_input)) Find optimal control sequence by weighted average of trajectory costs
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