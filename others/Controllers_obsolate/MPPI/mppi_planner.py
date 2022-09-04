import sys
from matplotlib.font_manager import json_dump

from numba.core import compiler_machinery

from scipy.spatial.distance import cdist
# sys.path.insert(0, "./pkg/src/pkg/mppi")

from MPPI.mppi_settings import MppiSettings
# from util import *

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import shapely.geometry as geom
import matplotlib.pyplot as plt

import time
import math
import csv
from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid

from pyglet.gl import GL_POINTS
import pyglet

def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    
    x = x0
    #Calculate dt vector
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    return history

    


class MppiPlanner:
    """
    Implementation of a neural MPPI MPC controller
    """

    def __init__(self, track=None, predictor="euler", model_name=None):
        """
        Initilalize the MPPI MPC car controller
        @param track{Track}: The track which the car should complete 
        @param predictor{enum:("euler"|"odeint"|"nn") }: The type of prediction the controller uses 
        @param model_name {String} (optional): Required if prediction = "nn", defines the name of the neural network that solves the trajectory prediction 
        @return: None
        """

        self.lidar_scan_angles = np.linspace(-2.35,2.35, 1080)

        # Global
        self.tControlSequence = 0.05  # [s] How long is a control input applied

        # Racing objects
        #     x1: x position in global coordinates
        #     x2: y position in global coordinates
        #     x3: steering angle of front wheels
        #     x4: velocity in x direction
        #     x5: yaw angle
        #     x6: yaw rate
        #     x7: slip angle at vehicle center
        self.car_state = MppiSettings.INITIAL_STATE
      

        self.track = track  # Track waypoints for drawing

        # Control with common-road models
        self.predictior = predictor
        # self.parameters = parameters_vehicle1()
        
        self.params = {
                'mu': 1.0489,       # friction coefficient  [-]
                'C_Sf': 4.718,      # cornering stiffness front [1/rad]
                'C_Sr': 5.4562,     # cornering stiffness rear [1/rad]
                'lf': 0.15875,      # distance from venter of gracity to front axle [m]
                'lr': 0.17145,      # distance from venter of gracity to rear axle [m]
                'h': 0.074,         # center of gravity height of toal mass [m]
                'm': 3.74,          # Total Mass of car [kg]
                'I': 0.04712,       # Moment of inertia for entire mass about z axis  [kgm^2]
                's_min': -0.4189,   # Min steering angle [rad]
                's_max': 0.4189,    # Max steering angle [rad]
                'sv_min': -3.2,     # Min steering velocity [rad/s]
                'sv_max': 3.2,      # Max steering velocity [rad/s]
                'v_switch': 7.319,  # switching velocity [m/s]
                'a_max': 9.51,      # Max acceleration [m/s^2]
                'v_min':-5.0,       # Min velocity [m/s]
                'v_max': 20.0,      # Max velocity [m/s]
                'width': 0.31,      # Width of car [m]
                'length': 0.58      # Length of car [m]
                }

        self.tEulerStep = 0.01 # [s] One step of solving the ODEINT or EULER

        # Control with neural network
        # if predictor == "nn":
        #     print("Setting up controller with neural network")
        #     if model_name is not None:
        #         from nn_prediction.prediction import NeuralNetworkPredictor

        #         self.nn_predictor = NeuralNetworkPredictor(model_name=model_name)

        # MPPI data
        self.simulated_history = [
        [[0,0,0,0,0,0,0]]
        ]  # Hostory of simulated car states
       
        self.simulated_costs = []  # Hostory of simulated car states
        self.last_control_input = [0, 0]
        self.best_control_sequenct = []

        # Data collection
        self.collect_distance_costs = []
        self.collect_acceleration_costs = []

        # Goal point for new cost function
        self.goal_point = (0,0)

        self.collision_corse = False
        # Get track ready
        self.update_trackline()

        self.vertex_list = pyglet.graphics.vertex_list(2,
                                                       ('v2i', (10, 15, 30, 35)),
                                                       ('c3B', (0, 0, 255, 0, 255, 0))
                                                       )
        
        self.next_state_estimation = MppiSettings.INITIAL_STATE
        self.simulation_index = 0

    def set_state(self, state):
        """
        Overwrite the controller's car_state
        Maps the repetitive variables of the car state onto the trained space
        Fot the L2Race deployment the states have to be transformed into [m] and [deg] instead of [pixel] and [rad]
        @param state{array<float>[7]}: the current state of the car
        """
        self.car_state = state

        # Convert yaw angle to trained space -pi, pi
        self.car_state[4] = self.car_state[4] % 6.28
        if self.car_state[4] > 3.14:
            self.car_state[4] = self.car_state[4] - 6.28

    def update_trackline(self):
        return
        """
        Update the the next points of the trackline due to the current car state
        Those waypoints are used for the cost function and drawing the simulated history
        """

        if self.track == None:
            return


        # save only Next NUMBER_OF_NEXT_WAYPOINTS points of the track
        waypoint_modulus = self.track.waypoints.copy()
        waypoint_modulus.extend(waypoint_modulus[:NUMBER_OF_NEXT_WAYPOINTS])

        closest_to_car_position = self.track.get_closest_index(self.car_state[:2])
        first_waypoint = closest_to_car_position + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS
        last_waypoint = (
            closest_to_car_position
            + NUMBER_OF_NEXT_WAYPOINTS
            + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS
        )

        waypoint_modulus = waypoint_modulus[first_waypoint:last_waypoint]

        self.trackline = geom.LineString(waypoint_modulus)

    def car_dynamics(self,x, t, u, p):
        """
        Dynamics of the simulated car from common road
        To use other car dynamics than the defailt ones, comment out here
        @param x: The cat's state
        @param t: array of times where the state has to be evaluated
        @param u: Control input that is applied on the car
        @param p: The car's physical parameters
        @returns: the commonroad car dynamics function, that can be integrated for calculating the state evolution
        """

        f = vehicle_dynamics_st(
            x,
            u,
            self.params['mu'],
            self.params['C_Sf'],
            self.params['C_Sr'],
            self.params['lf'],
            self.params['lr'],
            self.params['h'],
            self.params['m'],
            self.params['I'],
            self.params['s_min'],
            self.params['s_max'],
            self.params['sv_min'],
            self.params['sv_max'],
            self.params['v_switch'],
            self.params['a_max'],
            self.params['v_min'],
            self.params['v_max'])
        return f

    
    
    def simulate_step(self, state, control_input):
        """
        Calculate the next system state due to a given state and control input
        for one time step
        @state{array<float>[7]}: the system's state
        @control_input{array<float>[7]}: the applied control input 
        returns: the simulated car state {array<float>[7]} after tControlSequence [s]
        """

        # print("Simulate_step", state, control_input)
        t = np.arange(0, self.tControlSequence, self.tEulerStep) 

        x_next = solveEuler(self.car_dynamics, state, t, args=(control_input, self.params))
        x_next = x_next[-1]
        

        x_next[4] = x_next[4] % 6.28
        if x_next[4] > 3.14:
            x_next[4] = x_next[4] - 6.28

        return x_next


    def simulate_trajectory(self, control_inputs):
        """
        Simulates a hypothetical trajectory of the car due to a list of control inputs
        @control_inputs: list<control_input> The list of apllied control inputs over time
        returns: simulated_trajectory{list<state>}: The simulated trajectory due to the given control inputs, cost{float}: the cost of the whole trajectory
        """

        simulated_state = self.car_state
        simulated_trajectory = []
        cost = 0
        index = 0


        start = time.time()
        for control_input in control_inputs:
            if cost > MppiSettings.MAX_COST:
                cost = MppiSettings.MAX_COST
                # continue

            simulated_state = self.simulate_step(simulated_state, control_input)

            simulated_trajectory.append(simulated_state)

            index += 1
        end = time.time()
        # print("Simulate all trajectories", end - start)


        start = time.time()
        cost = self.point_cloud_cost_function(simulated_trajectory, control_inputs)
        end = time.time()
        # print("Get distance from cost function", end - start)

        return simulated_trajectory, cost

    def simulate_trajectory_distribution(self, control_inputs_distrubution):
        """
        Simulate and rage a distribution of hypothetical trajectories of the car due to multiple control sequences
        @control_inputs_distrubution: list<control sequence> A distribution of control sequences
        returns: results{list<state_evolutions>}: The simulated trajectories due to the given control sequences, costs{array<float>}: the cost for each trajectory
        """

        # if we predict the trajectory distribution with a neural network, we have to swap the axes for speedup.
        # if self.predictior == "nn":
        #     return self.simulate_trajectory_distribution_nn(control_inputs_distrubution)

  
        results = []
        costs = np.zeros(len(control_inputs_distrubution))

        i = 0
        for control_input in control_inputs_distrubution:
            simulated_trajectory, cost = self.simulate_trajectory(control_input)

            results.append(simulated_trajectory)
            costs[i] = cost
            i += 1

        self.simulated_history = results
        self.simulated_costs = costs

        return results, costs

    # Not yet used: First train Neural networks
    def simulate_trajectory_distribution_nn(self, control_inputs_distrubution):

        """
        Simulate and rage a distribution of hypothetical trajectories of the car due to multiple control sequences
        This method does the same like simulate_trajectory_distribution but it is optimized for a fast use for neural networks. 
        Since the keras model can process much data at once, we have to swap axes to parallelly compute the rollouts.
        @control_inputs_distrubution: list<control sequence> A distribution of control sequences
        returns: results{list<state_evolutions>}: The simulated trajectories due to the given control sequences, costs{array<float>}: the cost for each trajectory
        """

        control_inputs_distrubution = np.swapaxes(control_inputs_distrubution, 0, 1)
        results = []
        states = np.array(len(control_inputs_distrubution[0]) * [self.car_state])
        for control_inputs in control_inputs_distrubution:

            states = self.nn_predictor.predict_multiple_states(states, control_inputs)
            results.append(states)

        results = np.array(results)
        results = np.swapaxes(results, 0, 1)

        costs = []
        for result in results:
            cost = self.border_segments_cost_function(result)
            costs.append(cost)

        self.simulated_history = results
        self.simulated_costs = costs

        return results, costs

    def static_control_inputs(self):

        """
        Sample primitive hand crafted control sequences
        This method was only for demonstration purposes and is no longer used.
        @returns: results{list<control sequence>}: A primitive distribution of control sequences
        """
        control_inputs = [
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0., 0.]],  # No input
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[-0.2, 0.]],  # little left
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[-1., 0.]],  # hard left
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0.2, 0.]],  # little right
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[1., 0.]],  # hard right
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0., -1.]],  # brake
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[0., 5.]],  # accelerate
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[-1., 5.]],  # accelerate and left
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY * [[1., 5.]],  # accelerate and right
        ]
        # print ("control_input_sequences",control_inputs)

        return control_inputs

    def sample_control_inputs(self):
        """
        Sample zero mean gaussian control sequences
        @returns: results{list<control sequence>}: A zero mean gaussian distribution of control sequences
        """

        steering = np.random.normal(
            0,
            MppiSettings.INITIAL_STEERING_VARIANCE,
            MppiSettings.NUMBER_OF_INITIAL_TRAJECTORIES * MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY,
        )
        # acceleration = np.random.normal(
        #     INITIAL_ACCELERATION_MEAN,
        #     INITIAL_ACCELERATION_VARIANCE,
        #     MppiSettings.NUMBER_OF_INITIAL_TRAJECTORIES * MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY,
        # )
        acceleration = np.random.uniform(
            -10,
            10,
            MppiSettings.NUMBER_OF_INITIAL_TRAJECTORIES * MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY,
        )

        control_input_sequences = np.column_stack((steering, acceleration))
        control_input_sequences = np.reshape(
            control_input_sequences,
            (MppiSettings.NUMBER_OF_INITIAL_TRAJECTORIES, MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY, 2),
        )
        return control_input_sequences

    def sample_control_inputs_history_based(self, last_control_sequence, ):
        """
        Sample history based control sequences (simmilar to the last "perfect" strategy)
        @returns: results{list<control sequence>}: A history based small variance distribution of control sequences
        """

        # Chose sampling method by uncommenting
        # return self.sample_control_inputs()
        # return self.static_control_inputs()

        # Not initialized
        if len(last_control_sequence) == 0:
            return self.sample_control_inputs()
        if self.collision_corse:
            return self.sample_control_inputs()

        # Delete the first step of the last control sequence because it is already done
        # To keep the length of the control sequence add one to the end
        last_control_sequence = last_control_sequence[1:]
        last_control_sequence = np.append(last_control_sequence, [0, 0]).reshape(
            MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY, 2
        )

        last_steerings = last_control_sequence[:, 0]
        last_accelerations = last_control_sequence[:, 1]

        control_input_sequences = np.zeros(
            [MppiSettings.NUMBER_OF_TRAJECTORIES, MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY, 2]
        )

        for i in range(MppiSettings.NUMBER_OF_TRAJECTORIES):

            steering_noise = np.random.normal(
                0, MppiSettings.STEP_STEERING_VARIANCE, MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY
            )
            acceleration_noise = np.random.normal(
                0, MppiSettings.STEP_ACCELERATION_VARIANCE, MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY
            )

            next_steerings = last_steerings + steering_noise
            next_accelerations = last_accelerations + acceleration_noise

            # Optional: Filter for smoother control
            # next_steerings = signal.medfilt(next_steerings, 3)
            # next_accelerations = signal.medfilt(next_accelerations, 3)

            next_control_inputs = np.vstack((next_steerings, next_accelerations)).T
            control_input_sequences[i] = next_control_inputs

        return control_input_sequences

    def point_cloud_cost_function(self, trajectory, control_inputs, log = False):

        def dist(p1, p2):
            return (p1[0] - p2[0]) **2 + (p1[1] - p2[1]) **2

        def dist_to_array(point, points2):
            points1 = len(points2) * [point]

            points1 = np.array(points1)
            points2 = np.array(points2)

            diff_x = points1[:,0] - points2[:,0]
            diff_y = points1[:,1] - points2[:,1]
            squared_distances = np.square(diff_x) + np.square(diff_y)

            return squared_distances


        executed_control_input = control_inputs[0]        
        executed_acceleration = executed_control_input[1]

        acceleration_cost =  10 - executed_acceleration

        initial_state = trajectory[0] 
        initial_position = initial_state[:2]

        terminal_state = trajectory[-1]
    
        distance_cost = 0

        steer_sum = 0
        for control_input in control_inputs:
            steer_sum += abs(control_input[0])
        
        trajectory = np.array(trajectory)


        for state in trajectory:
            distances = dist_to_array(state, self.lidar_points)
            if(np.any(distances < 0.15) ):
                return 10000

        speed_cost = 10.4 - terminal_state[3]
        steer_cost = steer_sum

        

        cost = 0 * distance_cost + 1 * speed_cost + 2 * steer_cost + 1 * acceleration_cost

        if(False):
            print("distance_cost", distance_cost)
            print("speed_cost", speed_cost)
            print("steer_cost", steer_cost)
            print("acceleration_cost", acceleration_cost)
            print("cost", cost)


        return cost

    # Deprecated: Cost function on comparison to the border line segments of the track
    # Not useable if there are obstacles 
    def border_segments_cost_function(self, trajectory):
        return 0
        """
        calculate the cost of a trajectory
        @param: trajectory {list<state>} The trajectory of states that needs to be evaluated
        @returns: cost {float}: the scalar cost of the trajectory
        """
        distance_cost_weight = 1
        terminal_speed_cost_weight = 0
        terminal_position_cost_weight = 1

        distance_cost = 0
        terminal_speed_cost = 0
        terminal_position_cost = 0
       

        number_of_states = len(trajectory)
        index = 0


        # Don't come too close to border
        for state in trajectory:

            simulated_position = geom.Point(state[0], state[1])

            for border_line in self.track.line_strings:
                distance_to_border = simulated_position.distance(border_line)
                distance_cost += 1 / math.exp(distance_to_border)
                # if(distance_to_border < 0.15):
                #     distance_cost += 1000

  
        # Initial State
        initial_state = trajectory[0]
        terminal_state = trajectory[-1]

        # Terminal Speed cost
        # terminal_speed = terminal_state[3]
        # if(terminal_speed != 0):
        #     terminal_speed_cost += abs(1 / terminal_speed)
        # else:
        #     terminal_speed_cost = 1

        # if terminal_state[3] < 5:  # Min speed  = 5
        #     terminal_speed_cost += 3 * abs(5 - terminal_speed)

        # Terminal Position cost
        terminal_position = geom.Point(terminal_state[0], terminal_state[1])
        terminal_distance_to_initial_state = terminal_position.distance(geom.Point(initial_state))
        terminal_position_cost += abs(terminal_distance_to_initial_state)

        # Don't cross the border
        trajectory_line = geom.LineString([initial_state[:2], terminal_state[:2]])
        border_line = self.track.line_strings[0]
        for border_line in self.track.line_strings:
            if(border_line.crosses(trajectory_line)):
                terminal_position_cost = 1000

    

        # print("distance_cost", distance_cost)
        # print("terminal_position_cost", terminal_position_cost)

        # Total cost
        cost = (
            distance_cost_weight * distance_cost
            + terminal_speed_cost_weight * terminal_speed_cost
            + terminal_position_cost_weight * terminal_position_cost
        )
        # print("cost", cost)
        return cost

       

    # Deprecated: Old costfunction from l2race
    # Can be used again when we know the optimal trackline
    def waypoint_cost_function(self, trajectory):
        """
        calculate the cost of a trajectory
        @param: trajectory {list<state>} The trajectory of states that needs to be evaluated
        @returns: cost {float}: the scalar cost of the trajectory
        """

        distance_cost_weight = 1
        terminal_speed_cost_weight = 0
        terminal_position_cost_weight = 0
        angle_cost_weight = 0

        distance_cost = 0
        angle_cost = 0
        terminal_speed_cost = 0
        terminal_position_cost = 0

        number_of_states = len(trajectory)
        index = 0

        

        # angles = np.absolute(self.track.AngleNextCheckpointRelative)
        # waypoint_index = 0 #self.track.get_closest_index(self.car_state[:2])
        # angles = angles[
        #     waypoint_index
        #     + ANGLE_COST_INDEX_START : waypoint_index
        #     + ANGLE_COST_INDEX_STOP
        # ]
        # angles_squared = np.absolute(angles)  # np.square(angles)
        # angle_sum = np.sum(angles_squared)
        angle_sum = 0

        for state in trajectory:
            discount = (number_of_states  - 0.05 * index) / number_of_states

            simulated_position = geom.Point(state[0], state[1])
            distance_to_track = simulated_position.distance(self.trackline)
            distance_to_track = discount * distance_to_track

            # for line in self.track.line_strings:

            #     distance_to_border = simulated_position.distance(line)
            #     distance_cost += 1 + 1/distance_to_border

            # distance_to_track = distance_to_track ** 2
            distance_cost += distance_to_track

            # print("Car position: ", state[:2])
            # print("Closest trackline: ",self.trackline)
            # print("Distance to track: ", distance_to_track )

            # Don't leave track!
            # if distance_to_track > TRACK_WIDTH:
            #     distance_cost += 1000
            # index += 1

        # Terminal Speed cost
        terminal_state = trajectory[-1]
        terminal_speed = terminal_state[3]
        if(terminal_speed != 0):
            terminal_speed_cost += abs(1 / terminal_speed)
        else:
            terminal_speed_cost = 1

        if terminal_state[3] < 5:  # Min speed  = 5
            terminal_speed_cost += 3 * abs(5 - terminal_speed)

        # Terminal Position cost
        terminal_position = geom.Point(terminal_state[0], terminal_state[1])
        terminal_distance_to_track = terminal_position.distance(self.trackline)
        terminal_position_cost += abs(terminal_distance_to_track)

        # Angle cost
        angle_cost = angle_sum * terminal_state[3]

        # Total cost
        cost = (
            distance_cost_weight * distance_cost
            + terminal_speed_cost_weight * terminal_speed_cost
            + terminal_position_cost_weight * terminal_position_cost
            + angle_cost_weight * angle_cost
        )
        return cost

    def process_observation(self, ranges=None, ego_odom=None):

        """
        gives actuation given observation
        @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
        @ ego_odom: A dict with following indices:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
            'linear_vel_x': float,
            'linear_vel_y': float,
            'angular_vel_z': float,
        }
        """
        
        if  self.simulation_index < 10:
            self.simulation_index += 1
            return 100, 0
             
        pose_x = ego_odom['pose_x']
        pose_y = ego_odom['pose_y']
        pose_theta = ego_odom['pose_theta']
        linear_vel_x = ego_odom['linear_vel_x']
        linear_vel_y = ego_odom['linear_vel_y']

        speed = math.sqrt(linear_vel_x ** 2 + linear_vel_y ** 2)
        angle = linear_vel_x * math.sin(linear_vel_y)

        angular_vel_z = ego_odom['angular_vel_z']

        slipping_angle = angle - pose_theta
        # print("angle", angle)
        print("pose_theta", pose_theta)
        # print("linear_vel_x", linear_vel_x)
        # print("linear_vel_y", linear_vel_y)
        print("angle", angle)

        # State of TUM dynamical models
        #     x1: x position in global coordinates
        #     x2: y position in global coordinates
        #     x3: steering angle of front wheels
        #     x4: velocity in front direction
        #     x5: yaw angle
        #     x6: yaw rate
        #     x7: slip angle at vehicle center
        # self.car_state = [
        #     pose_x, pose_y, 0. , linear_vel_x, pose_theta, angular_vel_z, 0.
        # ]
        
        # self.car_state = self.next_state_estimation
        # self.car_state[0] = pose_x
        # self.car_state[1] = pose_y
        # self.car_state[2] = pose_theta
        # self.car_state[3] = linear_vel_x
        # self.car_state[4] = pose_theta
        # self.car_state[5] = angular_vel_z
        
        # print("mppi state estimation", self.car_state)
        # print("mppi state estimation", self.next_state_estimation)

        scans = ranges
        points = []
        for i in range(108):
            max_distance = 15
            index = 10*i
            if(scans[index] > max_distance): continue
            p1 = self.car_state[0] + scans[index] * math.cos(self.lidar_scan_angles[index] + self.car_state[4])
            p2 = self.car_state[1] + scans[index] * math.sin(self.lidar_scan_angles[index] + self.car_state[4])
            points.append((p1,p2))
            
            # print("points", points)
            self.lidar_points = points
    

        # Take into account size of car
        mppi_accel, mppi_steer = self.plan()
        
        next_state = self.simulate_step(self.car_state, [mppi_accel, mppi_steer])
        
      
        print("Next state", next_state)
        
        # print ("Accel, steer", mppi_accel, mppi_steer)
        self.simulation_index += 1
        
        return mppi_accel, mppi_steer
        
    def plan(self):
        dist = self.sample_control_inputs()
        
        # dist = self.static_control_inputs()
        # dist = self.sample_control_inputs_history_based(self.best_control_sequenct)
        # dist = self.sample_control_inputs()

        self.update_trackline()
        
        trajectories, costs = self.simulate_trajectory_distribution(dist)

        # print("Costs", costs)

        
        if(math.isnan(costs[0])):
            min_index = 0
            mppi_steering = dist[min_index][0][0]
            mppi_speed = dist[min_index][0][1]

        else:
            min_index = np.argmin(costs)

            mppi_steering =  dist[min_index][0][0]

            weights = np.zeros(len(dist))
            best_index = 0
            for i in range(len(dist)):
                lowest_cost = 100000

                cost = costs[i]
                # find best
                if cost < lowest_cost:
                    best_index = i
                    lowest_cost = cost

                if(cost < 9999):
                    weight = math.exp((-1 / MppiSettings.INVERSE_TEMP) * cost)
                else:
                    weight = 0
                
                weights[i] = weight

            if not np.all((weights == 0)):
                next_control_sequence = np.average(dist, axis=0, weights=weights)
                self.collision_corse = False

            else:
                next_control_sequence = [[0,-5]] * MppiSettings.NUMBER_OF_STEPS_PER_TRAJECTORY
                mppi_steering = 0
                mppi_speed = -5
                print("No good trajectory")
                self.collision_corse = True
                # next_control_sequence = dist[best_index]

            # next_control_sequence = dist[best_index]
            
            best_route, best_cost = np.array(self.simulate_trajectory(next_control_sequence))
            # executed_cost = self.point_cloud_cost_function(best_route, next_control_sequence, log = True)
            self.next_state_estimation = best_route[0]
            # print("lowest_cost",lowest_cost)

            mppi_steering = next_control_sequence[0][0]
            mppi_speed = next_control_sequence[0][1]

            self.best_control_sequenct = next_control_sequence

        if MppiSettings.DRAW_LIVE_ROLLOUTS:
            best_route, best_cost = np.array(self.simulate_trajectory(next_control_sequence))
            self.draw_simulated_history(0, best_route)


        # print("Planning in Car Controller", mppi_speed, mppi_steering )

        return mppi_speed, mppi_steering


  

    """
    draws the simulated history (position and speed) of the car into a plot for a trajectory distribution resp. the history of all trajectory distributions
    """
    def render(self,renderer):
        points = np.array(self.simulated_history)
        chosen_trajectory = points[0][:][:]
        chosen_trajectory_positions = chosen_trajectory[:][:]
        c = 0
        

        if(points.shape[0] != 1):
            self.vertex_list.delete()
            points = points.reshape((points.shape[0] * points.shape[1],7))
            trajectory = points[:,:2]

            scaled_points = 50.*trajectory

            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()

            c = c + 140
            self.vertex_list = renderer.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                        ('c3B', (0, c, 255 -c) * howmany ))


        self.simulated_history  = [[0,0,0,0,0,0,0]]
        
    def draw_simulated_history(self, waypoint_index=0, chosen_trajectory=[]):
        
        
        plt.clf()

        plt.xlim(-6, 6)
        plt.ylim(-2, 10)

        fig, position_ax = plt.subplots()

        # for segment in self.track.segments:
        #     x_val = [x[0] for x in segment]
        #     y_val = [x[1] for x in segment]

        #     plt.plot(x_val,y_val,'o')

        plt.title("History based random control")
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

        plt.savefig("mppi/live_rollouts_without_optimal.png")



        s_x = []
        s_y = []
        costs = []
        i = 0
        ind = 0
        indices = []
        for trajectory in self.simulated_history:
            cost = self.simulated_costs[i]
            if cost < 9999:
                for state in trajectory:
                    # if state[0] > 1:
                    s_x.append(state[0])
                    s_y.append(state[1])
                    costs.append(cost)
                    indices.append(cost)
                    ind += 1
                i += 1

        trajectory_costs = position_ax.scatter(s_x, s_y, c=indices)
        colorbar = fig.colorbar(trajectory_costs)
        colorbar.set_label("Trajectory costs")

        # Draw car position
        p_x = self.car_state[0]
        p_y = self.car_state[1]
        position_ax.scatter(p_x, p_y, c="#FF0000", label="Current car position")

        # Draw waypoints
        # waypoint_index = self.track.get_closest_index(self.car_state[:2])

        # waypoints = np.array(self.track.waypoints)
        # w_x = waypoints[
        #     waypoint_index
        #     + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS : waypoint_index
        #     + NUMBER_OF_NEXT_WAYPOINTS
        #     + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS,
        #     0,
        # ]
        # w_y = waypoints[
        #     waypoint_index
        #     + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS : waypoint_index
        #     + NUMBER_OF_NEXT_WAYPOINTS
        #     + NUMBER_OF_IGNORED_CLOSEST_WAYPOINTS,
        #     1,
        # ]
        # position_ax.scatter(w_x, w_y, c="#000000", label="Next waypoints")

        # Draw Borders

        # segment = np.array(self.track.segments[0])
        # print("Segment", segment)
        # w_x = segment[:,0]
        # w_y = segment[:,1]
        # print("w_x", w_x)
        # print("w_y", w_y)
        # position_ax.scatter(w_x, w_y, c="#008800", label="Next waypoints")
        # Plot Chosen Trajectory
        t_x = []
        t_y = []
        plt.savefig("live_rollouts_without_optimal.png")
        for state in chosen_trajectory:
            t_x.append(state[0])
            t_y.append(state[1])

        plt.scatter(t_x, t_y, c="#D94496", label="Chosen control")

        plt.savefig("live_rollouts.png")
        plt.legend(fancybox=True, shadow=True, loc="best")
        return plt

