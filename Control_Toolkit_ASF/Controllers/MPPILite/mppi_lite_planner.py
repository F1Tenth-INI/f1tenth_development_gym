from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils
from Control_Toolkit_ASF.Controllers import template_planner
from f110_sim.envs.dynamic_model_pacejka_jit import *
from utilities.car_files.vehicle_parameters import VehicleParameters
import time
from scipy.special import softmax
import os
os.environ["NUMBA_NUM_THREADS"] = '8'


class MPPILitePlanner(template_planner):

    def __init__(self):

        print('Loading MPPILitePlanner')

        super().__init__()

        self.simulation_index = 0
        self.car_state = None
        self.waypoints = None
        self.imu_data = None
        
        self.car_state_history = []
        
        self.render_utils = RenderUtils()
        self.send_to_recorder = None

        
        self.waypoint_utils = None  # Will be overwritten with a WaypointUtils instance from car_system
        
        self.angular_control = 0
        self.translational_control = 0
        
        self.control_index = 0
        
        self.mu_predicted = 0
        self.predicted_frictions = []
        
        
        # Precompute variables
        self.batch_size = 2001
        self.horizon = 50    
        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.car_params_array = VehicleParameters().to_np_array()

        s_batch = np.zeros(( self.batch_size, 7), dtype=np.float32)
        Q_batch_sequence = np.random.rand(self.horizon,  self.batch_size, 2).astype(np.float32)
        total_cost_batch = np.random.rand(self.batch_size).astype(np.float32)

        # Call JIT functions once to force compilation
        _ = car_step_parallel(s_batch, Q_batch_sequence[0], self.car_params_array, 0.01)
        _ = batch_sequence(Q_batch_sequence, s_batch, self.car_params_array)
        # _ = cost_function_batch_sequence(Q_batch_sequence, Q_batch_sequence, self.waypoint_utils.next_waypoints)
        _ = exponential_weighting(Q_batch_sequence, total_cost_batch)
        _ = compute_waypoint_distance(np.zeros(7), np.zeros((30, 8)))
        print("JIT compilation completed")
        

        

    
    def process_observation(self, ranges=None, ego_odom=None):
        # Convert self variables into local variables
        batch_size = self.batch_size    
        horizon = self.horizon
        car_params_array = self.car_params_array
        last_Q_sq = self.last_Q_sq

        s = np.array(self.car_state)
        
        # Call JIT-compiled function
        next_control, Q_sequence = process_observation_jit(
            s, last_Q_sq, batch_size, horizon, car_params_array, self.waypoint_utils.next_waypoints
        )

        # Store updated Q sequence
        self.last_Q_sq = Q_sequence

        # Extract final control
        self.angular_control, self.translational_control = next_control

        self.control_index += 1
        return self.angular_control, self.translational_control

@njit(fastmath=True, parallel=True)
def process_observation_jit(s, last_Q_sq, batch_size, horizon, car_params_array, next_waypoints):
    # Initialize s_batch manually
    s_batch = np.zeros((batch_size, s.shape[0]), dtype=np.float32)
    for i in prange(batch_size):  # Parallelize batch expansion
        s_batch[i] = s

    # Manually shift last_Q_sq instead of using np.roll()
    print(f"Before shift: last_Q_sq.shape = {last_Q_sq.shape}")

    last_Q_sq[:-1] = last_Q_sq[1:]  # Shift all rows forward
    last_Q_sq[-1] = np.random.rand(2).astype(np.float32)  # Update last row
    print(f"After shift: last_Q_sq.shape = {last_Q_sq.shape}")


    # Expand last_Q_sq to batch size using a loop (Numba-compatible)
    Q_batch_sequence = np.zeros((horizon, batch_size, 2), dtype=np.float32)
    for i in prange(batch_size):  # Parallelize over batch
        Q_batch_sequence[:, i, :] = last_Q_sq


    # Perturb Qs by adding random noise
    random_perturbation = np.zeros((horizon, batch_size, 2), dtype=np.float32)
    random_perturbation[:, :, 0] = np.random.uniform(-0.2, 0.2, size=(horizon, batch_size))
    random_perturbation[:, :, 1] = np.random.uniform(-1, 1, size=(horizon, batch_size))

    
    Q_batch_sequence += random_perturbation

    # Clip control values (steering angle -0.4 - 0.4 and throttle -10, 10)
    # Define control limits
    steering_min, steering_max = -0.4, 0.4  # Example range for steering
    throttle_min, throttle_max = -5, 10  # Example range for throttle

    # Clip the control values to ensure they stay within limits
    Q_batch_sequence[:, :, 0] = np.clip(Q_batch_sequence[:, :, 0], steering_min, steering_max)  # Clip steering
    Q_batch_sequence[:, :, 1] = np.clip(Q_batch_sequence[:, :, 1], throttle_min, throttle_max)  # Clip throttle

        
    # Run batch rollout
    state_batch_rollout = batch_sequence(Q_batch_sequence, s_batch, car_params_array)

    # Compute total cost
    waypoints = next_waypoints
    total_cost_batch = cost_function_batch_sequence(state_batch_rollout, Q_batch_sequence, waypoints)
    # Compute optimal Q sequence
    Q_sequence = exponential_weighting(Q_batch_sequence, total_cost_batch)

    # Extract the first control decision
    next_control = Q_sequence[0]

    return next_control, Q_sequence




@njit(fastmath=True)
def cost_function(state, control, waypoints):
    """
    Computes cost based on speed, control effort, and distance to the next waypoint.
    """
    speed_cost = state[3] * -0.08  # Penalize low speed
    angular_cost = abs(control[0]) * 0.0  # Penalize sharp steering

    # Compute waypoint distance cost
    waypoint_dist_sq = compute_waypoint_distance(state, waypoints)
    waypoint_cost = waypoint_dist_sq * 0.1  # Scale weight for balancing

    total_cost = speed_cost + angular_cost + waypoint_cost
    return total_cost


@njit(fastmath=True, parallel=True)
def cost_function_batch(state_batch, control_batch, waypoints):
    """
    Computes cost for a batch of states and controls.
    """
    batch_size = state_batch.shape[0]
    cost_batch = np.zeros(batch_size, dtype=np.float32)

    for i in prange(batch_size):  # Parallel execution
        cost_batch[i] = cost_function(state_batch[i], control_batch[i], waypoints)

    return cost_batch


@njit(fastmath=True,)
def cost_function_batch_sequence(state_batch_rollout, Q_batch_sequence, waypoints):
    """
    Computes total cost for an entire batch sequence using waypoint distances.
    """
    horizon, batch_size = state_batch_rollout.shape[:2]
    total_cost_batch = np.zeros(batch_size, dtype=np.float32)

    for index in prange(horizon):  
        total_cost_batch += cost_function_batch(state_batch_rollout[index], Q_batch_sequence[index], waypoints)

    return total_cost_batch


@njit(fastmath=True, parallel=True)
def exponential_weighting(Q_batch_sequence, total_cost_batch, temperature=0.5):
    # Softmax manually implemented for Numba compatibility
    max_cost = np.max(-total_cost_batch)  # Subtract max for numerical stability
    exp_weights = np.exp((-total_cost_batch - max_cost) / temperature)
    weights = exp_weights / np.sum(exp_weights)

    # Reshape weights for broadcasting
    Q_batch_sequence_weighted = Q_batch_sequence * weights.reshape(1, -1, 1)

    # Compute weighted sum over the batch dimension
    Q_sequence = np.sum(Q_batch_sequence_weighted, axis=1)

    return Q_sequence
    
@njit(fastmath=True, parallel=True)
def batch_sequence(Q_batch_sequence, s_batch, car_params):
    horizon, batch_size, _ = Q_batch_sequence.shape
    state_batch_sequence = np.zeros((horizon, batch_size, 7), dtype=np.float32)

    dt = 0.01  

    for i in range(horizon):  
        # Process all batches in parallel
        state_batch_sequence[i] = car_step_parallel(s_batch, Q_batch_sequence[i], car_params, dt)

        # Instead of modifying `s_batch` in-place, store the last state explicitly
        if i < horizon - 1:
            s_batch[:] = state_batch_sequence[i]

    return state_batch_sequence



@njit(fastmath=True)
def compute_waypoint_distance(state, waypoints):
    """
    Computes the squared Euclidean distance between the car state and all waypoints,
    and returns the minimum distance found.
    """
    min_dist_sq = 1e6  # Large initial value
    car_x, car_y = state[0], state[1]  # Extract car position

    for i in range(waypoints.shape[0]):  # Loop through waypoints
        wp_x, wp_y = waypoints[i, 1], waypoints[i, 2]
        dist_sq = (car_x - wp_x) ** 2 + (car_y - wp_y) ** 2  # Squared Euclidean distance
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq  # Update minimum distance

    return min_dist_sq  # Return squared distance to avoid sqrt computation
