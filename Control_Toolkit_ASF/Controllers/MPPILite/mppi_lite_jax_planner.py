import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax
import os

from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils
from utilities.Settings import Settings
from Control_Toolkit_ASF.Controllers import template_planner
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import NUMBER_OF_STATES, POSE_X_IDX, POSE_Y_IDX, LINEAR_VEL_X_IDX

from sim.f110_sim.envs.car_model_jax import (car_steps_sequential_jax)


# Configure JAX for optimal GPU usage
jax.config.update('jax_enable_x64', False)  # Use 32-bit for better GPU performance
# jax.config.update('jax_default_device', jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0])

class MPPILitePlanner(template_planner):

    def __init__(self):
        print('Loading JAX MPPI Lite Planner')
        
        # Get available devices safely
        try:
            available_devices = jax.devices()
            print(f'JAX devices available: {available_devices}')
            print(f'Default JAX device: {jax.default_backend()}')
            
            # Try to get GPU device, fall back to CPU
            gpu_devices = [d for d in available_devices if d.device_kind == 'gpu']
            if gpu_devices:
                self.default_device = gpu_devices[0]
                print(f'Using GPU device: {self.default_device}')
            else:
                self.default_device = [d for d in available_devices if d.device_kind == 'cpu'][0]
                print(f'Using CPU device: {self.default_device}')
                
        except Exception as e:
            print(f'Warning: Error getting JAX devices: {e}')
            print('Falling back to CPU-only mode')
            # Force CPU mode
            import os
            os.environ['JAX_PLATFORMS'] = 'cpu'
            available_devices = jax.devices()
            self.default_device = available_devices[0]
            print(f'Using fallback device: {self.default_device}')
        
        super().__init__()

        self.simulation_index = 0
        self.car_state = None
        self.waypoints = None
        self.imu_data = None
        self.car_state_history = []

        self.render_utils = RenderUtils()
        self.waypoint_utils = WaypointUtils()

        self.angular_control = 0
        self.translational_control = 0
        self.control_index = 0

        self.dt = 0.02
        self.batch_size = 256
        self.horizon = 75
        
        # Control smoothness parameters
        self.intra_horizon_smoothness_weight = 1.0  # Weight for smoothness within horizon
        self.angular_smoothness_weight = 1.0  # Relative weight for angular control smoothness
        self.translational_smoothness_weight = 0.1  # Relative weight for translational control smoothness
        
        # Control output smoothing
        self.control_smoothing_alpha = 0.5  # EMA smoothing factor (0 = no smoothing, 1 = no memory)
        self.last_executed_angular = 0.0
        self.last_executed_translational = 0.0

        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.car_params_array = VehicleParameters('mpc_car_parameters.yml').to_np_array().astype(np.float32)
        
        # Ensure car params are on the selected device
        with jax.default_device(self.default_device):
            self.car_params_jax = jnp.array(self.car_params_array)

        self.key = jax.random.PRNGKey(0)
        
        # Pre-compile the main computation function
        print("Pre-compiling JAX functions...")
        self._precompile_functions()

    def _precompile_functions(self):
        """Pre-compile JAX functions to ensure they run on the selected device"""
        # Create dummy data for compilation
        dummy_state = jnp.zeros(10, dtype=jnp.float32)
        dummy_last_Q = jnp.zeros((self.horizon, 2), dtype=jnp.float32)
        dummy_waypoints = jnp.zeros((100, 6), dtype=jnp.float32)
        dummy_key = jax.random.PRNGKey(42)
        
        print("Compiling process_observation_jax...")
        
        # Trigger compilation on the selected device
        _ = process_observation_jax(
            dummy_state, dummy_last_Q, self.batch_size, self.horizon,
            self.car_params_jax, dummy_waypoints, dummy_key,
            execute_control_index=4,
            intra_horizon_smoothness_weight=self.intra_horizon_smoothness_weight,
            angular_smoothness_weight=self.angular_smoothness_weight,
            translational_smoothness_weight=self.translational_smoothness_weight
        )
        print(f"JAX functions compiled and ready for execution on {self.default_device}!")

    def process_observation(self, ranges=None, ego_odom=None):
        # Ensure all data is on the default device
        with jax.default_device(self.default_device):
            s = jnp.array(self.car_state, dtype=jnp.float32)
            waypoints = jnp.array(self.waypoint_utils.next_waypoints, dtype=jnp.float32)

            self.key, subkey = jax.random.split(self.key)
            
            Q_sequence, optimal_traj, state_batch_sequence, total_cost_batch = process_observation_jax(
                s, jnp.array(self.last_Q_sq), self.batch_size, self.horizon,
                self.car_params_jax, waypoints, subkey,
                execute_control_index=4,
                intra_horizon_smoothness_weight=self.intra_horizon_smoothness_weight,
                angular_smoothness_weight=self.angular_smoothness_weight,
                translational_smoothness_weight=self.translational_smoothness_weight
            )

            # Get CPU devices safely for Adam optimization
            try:
                cpu_devices = [d for d in jax.devices() if d.device_kind == 'cpu']
                cpu_device = cpu_devices[0] if cpu_devices else self.default_device
                
                # Move data to CPU for Adam optimization
                with jax.default_device(cpu_device):
                    Q_sequence_cpu = jax.device_put(Q_sequence, cpu_device)
                    s_cpu = jax.device_put(s, cpu_device)
                    car_params_cpu = jax.device_put(self.car_params_jax, cpu_device)
                    waypoints_cpu = jax.device_put(waypoints, cpu_device)
                
                    Q_sequence_refined = refine_optimal_control_adam(
                        Q_sequence_cpu, s_cpu, car_params_cpu, waypoints_cpu, self.horizon
                    )
                    
                    # Move refined result back to default device
                    Q_sequence = jax.device_put(Q_sequence_refined, self.default_device)
                    
            except Exception as e:
                print(f"Warning: Adam optimization failed: {e}, using unrefined sequence")
                # Use the original Q_sequence if refinement fails

            # Move results back to CPU for rendering (if needed)
            self.rollout_trajectories = np.array(state_batch_sequence)
            self.trajectory_costs = np.array(total_cost_batch)
            
            self.optimal_trajectory = np.array(optimal_traj)
            self.render_utils.update_mpc(
                rollout_trajectory=self.rollout_trajectories,
                optimal_trajectory=np.expand_dims(self.optimal_trajectory, axis=0),
            )

            execute_control_index = int(Settings.CONTROL_DELAY / self.dt)
            raw_angular, raw_translational = Q_sequence[execute_control_index]
            
            # Apply exponential moving average smoothing to control outputs
            self.angular_control = (self.control_smoothing_alpha * float(raw_angular) + 
                                  (1 - self.control_smoothing_alpha) * self.last_executed_angular)
            self.translational_control = (self.control_smoothing_alpha * float(raw_translational) + 
                                        (1 - self.control_smoothing_alpha) * self.last_executed_translational)
            
            # Update last executed values for next iteration
            self.last_executed_angular = self.angular_control
            self.last_executed_translational = self.translational_control
            
            self.last_Q_sq = np.array(Q_sequence)
            self.control_index += 1

            return float(self.angular_control), float(self.translational_control)


@jax.jit
def compute_waypoint_distance_jax(state, waypoints):
    dx = state[POSE_X_IDX] - waypoints[:, 1]
    dy = state[POSE_Y_IDX] - waypoints[:, 2]
    dist_sq = dx ** 2 + dy ** 2
    min_dist_sq = jnp.min(dist_sq)
    min_idx = jnp.argmin(dist_sq)
    return min_dist_sq, min_idx

@jax.jit
def cost_function_jax(state, control, waypoints):
    waypoint_dist_sq, min_idx = compute_waypoint_distance_jax(state, waypoints)
    angular_control_cost = jnp.abs(control[0]) * 1.0
    translational_control_cost = jnp.abs(control[1]) * 0.1
    waypoint_cost = waypoint_dist_sq * 10.0
    target_speed = waypoints[min_idx, 5]
    speed_cost = 0.25 * (state[LINEAR_VEL_X_IDX] - target_speed) ** 2
    
    # Add quadratic penalty for large angular controls to discourage extreme values
    angular_quadratic_penalty = control[0] ** 2 * 15.0  # Heavy penalty for large steering angles
    
    return speed_cost + angular_control_cost + translational_control_cost + waypoint_cost + angular_quadratic_penalty

@jax.jit
def cost_function_sequence_jax(state_sequence, control_sequence, waypoints, 
                              intra_horizon_smoothness_weight=2.0, 
                              angular_smoothness_weight=1.0, 
                              translational_smoothness_weight=0.1):
    # Standard cost for each state-control pair
    cost_fn = lambda s, u: cost_function_jax(s, u, waypoints)
    standard_costs = jax.vmap(cost_fn)(state_sequence, control_sequence)
    
    # Intra-horizon smoothness penalty - penalize sudden changes within the control sequence
    control_diff = control_sequence[1:] - control_sequence[:-1]
    
    # Individual step smoothness costs
    step_angular_smoothness = control_diff[:, 0] ** 2 * intra_horizon_smoothness_weight * angular_smoothness_weight
    step_translational_smoothness = control_diff[:, 1] ** 2 * intra_horizon_smoothness_weight * translational_smoothness_weight
    step_smoothness_costs = step_angular_smoothness + step_translational_smoothness
    
    # Add individual smoothness penalties to corresponding cost elements
    total_costs = standard_costs.at[1:].add(step_smoothness_costs)
    
    # Also add overall smoothness penalty to first element for extra emphasis
    total_smoothness_penalty = jnp.sum(step_smoothness_costs) * 0.1  # Small additional overall penalty
    total_costs = total_costs.at[0].add(total_smoothness_penalty)
    
    return total_costs

@jax.jit
def cost_function_batch_sequence_jax(state_batch_sequence, Q_batch_sequence, waypoints, 
                                   discount_factor=0.99, 
                                   intra_horizon_smoothness_weight=2.0,
                                   angular_smoothness_weight=1.0, 
                                   translational_smoothness_weight=0.1):
    horizon = Q_batch_sequence.shape[1]
    discount = discount_factor ** jnp.arange(horizon - 1, -1, -1)
    
    def compute_total_cost(states, controls):
        # Standard sequence cost with intra-horizon smoothness only
        sequence_costs = cost_function_sequence_jax(states, controls, waypoints, 
                                                  intra_horizon_smoothness_weight,
                                                  angular_smoothness_weight, 
                                                  translational_smoothness_weight)
        return sequence_costs * discount
    
    cost_batch = jax.vmap(compute_total_cost)(state_batch_sequence, Q_batch_sequence)
    return cost_batch

@jax.jit
def exponential_weighting_jax(Q_batch_sequence, total_cost_batch, temperature=5.):
    weights = jax.nn.softmax(-total_cost_batch / temperature)
    Q_weighted = Q_batch_sequence * weights[:, None, None]
    Q_mean = jnp.sum(Q_weighted, axis=0)
    return Q_mean


# Use the car dynamics functions from dynamic_model_pacejka_jax.py
# No need to redefine them here - they're imported above

@partial(jax.jit, static_argnames=["batch_size", "horizon", "execute_control_index"])
def process_observation_jax(state, last_Q_sq, batch_size, horizon, car_params, waypoints, key,
                          execute_control_index=4,
                          intra_horizon_smoothness_weight=2.0,
                          angular_smoothness_weight=1.0,
                          translational_smoothness_weight=0.1):
    s_batch = jnp.repeat(state[None, :], batch_size, axis=0)
    base_Q = jnp.roll(last_Q_sq, shift=-1, axis=0).at[-1].set(jax.random.normal(key, (2,), dtype=jnp.float32))

    # jax print for debugging
    # jax.debug.print("last_Q first 3 values: {}", last_Q_sq[:3])
    # jax.debug.print("base_Q first 3 values: {}, base_Q last element: {}", base_Q[:3], base_Q[-1])
    
    Q_batch_sequence = jnp.repeat(base_Q[None, :, :], batch_size, axis=0)
    key1, key2 = jax.random.split(key)
    noise = jnp.stack([
        jax.random.normal(key1, (batch_size, horizon)) * 0.1,
        jax.random.normal(key2, (batch_size, horizon)) * 1.2
    ], axis=-1)
    Q_batch_sequence += noise
    Q_batch_sequence = jnp.clip(Q_batch_sequence, jnp.array([-0.4, -5.0]), jnp.array([0.4, 20.0]))
    traj_batch = jax.vmap(lambda s_single, Q_single: car_steps_sequential_jax(
        s_single, Q_single, car_params, 0.02, horizon, model_type='pacejka'
    ))(s_batch, Q_batch_sequence)
    
    # Compute costs with only intra-horizon smoothness
    cost_batch = cost_function_batch_sequence_jax(
        traj_batch, Q_batch_sequence, waypoints, 0.99,
        intra_horizon_smoothness_weight, angular_smoothness_weight, translational_smoothness_weight
    )
    
    total_cost = jnp.sum(cost_batch, axis=1)
    Q_sequence = exponential_weighting_jax(Q_batch_sequence, total_cost)
    optimal_trajectory = car_steps_sequential_jax(state, Q_sequence, car_params, 0.02, horizon, model_type='pacejka')
    return Q_sequence, optimal_trajectory, traj_batch, total_cost



@partial(jax.jit, static_argnames=["horizon"])
def refine_optimal_control_adam(Q_init, s0, car_params, waypoints, horizon, 
                              intra_horizon_smoothness_weight=2.0,
                              angular_smoothness_weight=1.0,
                              translational_smoothness_weight=0.1):

    # Alternative: Cosine decay schedule (often works better for optimization)

    gradient_steps = 20
    lr_max = 0.04  # Initial learning rate
    lr_min = 0.005

    gradmax_clip = 1.0

    lr_schedule = optax.cosine_decay_schedule(lr_max, decay_steps=gradient_steps, alpha=lr_min)
    
    opt = optax.adam(lr_schedule)
    opt_state = opt.init(Q_init)

    def cost_fn(Q_seq):
        traj = car_steps_sequential_jax(s0, Q_seq, car_params, 0.02, horizon=horizon, model_type='pacejka')
        costs = cost_function_sequence_jax(traj, Q_seq, waypoints, 
                                         intra_horizon_smoothness_weight,
                                         angular_smoothness_weight, 
                                         translational_smoothness_weight)
        return jnp.sum(costs)

    def step(Q_seq, opt_state):
        loss, grad = jax.value_and_grad(cost_fn)(Q_seq)
        grad = jnp.clip(grad, -gradmax_clip, gradmax_clip)
        updates, opt_state = opt.update(grad, opt_state)
        Q_seq = optax.apply_updates(Q_seq, updates)
        return Q_seq, opt_state, loss, jnp.linalg.norm(grad)

    def scan_step(carry, step_idx):
        Q_seq, opt_state = carry
        Q_seq, opt_state, cost, grad_norm = step(Q_seq, opt_state)
        # jax.debug.print("Adam Step {0}: Cost = {1:.4f}, Grad Norm = {2:.4f}", step_idx, cost, grad_norm)
        return (Q_seq, opt_state), None

    (Q_final, _), _ = jax.lax.scan(scan_step, (Q_init, opt_state), jnp.arange(gradient_steps))
    return Q_final
