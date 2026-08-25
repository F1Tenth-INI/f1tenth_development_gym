import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax

from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils
from utilities.Settings import Settings
from Control_Toolkit_ASF.Controllers import template_planner
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import (
    NUMBER_OF_STATES, POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX, ANGULAR_VEL_Z_IDX,
)
from sim.f110_sim.envs.car_model_jax import (car_steps_sequential_jax)

# Configure JAX for optimal GPU usage
jax.config.update('jax_enable_x64', False)  # Use 32-bit for better GPU performance

T_CONTROL = 0.04  # Control timestep
"""
RPGD Planner Optimizations for Performance:

1. Maintained original horizon (50 steps: 30@T_CONTROLs + 20@T_CONTROLs like MPPI)
2. Hybrid timestep approach: consistent dt for gradient optimization, variable dt for evaluation
3. Eliminated redundant cost calculations (was computing costs twice per gradient step)
4. Removed expensive CPU-GPU transfers for Adam refinement
5. Optimized gradient steps (8 vs original 15) with better learning rate
6. Optimized elite plan management
7. Reduced debug output frequency

Key insight: Use consistent timestep during gradient optimization to avoid discontinuities,
but evaluate final costs with variable timestep to match MPPI's planning horizon.
This preserves RPGD's planning capability while maintaining smooth gradients.
"""
# MODEL_TYPE = 'residual'  # Use residual model for dynamics
MODEL_TYPE = 'pacejka'  # Use pacejka model for dynamics


class RPGDJaxSafetyFilter(template_planner):
    """Full-horizon RPGD safety-filter prototype.

    Unlike the racing planner, this optimizer has no waypoint-progress or
    target-speed tracking objective.  It only enforces track clearance along
    the horizon, treats waypoint speed as an upper bound, and prefers a
    stopped (or slow) terminal state — not raceline alignment.
    """

    def __init__(
        self,
        horizon: int | None = None,
        batch_size: int | None = None,
        gradient_steps: int | None = None,
        elite_size: int | None = None,
        control_smoothing_alpha: float | None = None,
        safety_margin: float = 0.30,
        w_u: float = 100.0,
        quiet: bool = False,
    ):
        """JAX RPGD planner.

        Optional kwargs let callers (e.g. the MPC safety filter) use a lighter
        population / shorter horizon without forking this class.
        """
        if not quiet:
            print('Loading JAX RPGD Planner')
        
        # Get available devices safely
        try:
            available_devices = jax.devices()
            if not quiet:
                print(f'JAX devices available: {available_devices}')
                print(f'Default JAX device: {jax.default_backend()}')
            
            # Try to get GPU device, fall back to CPU
            gpu_devices = [d for d in available_devices if d.device_kind == 'gpu']
            if gpu_devices:
                self.default_device = gpu_devices[0]
                if not quiet:
                    print(f'Using GPU device: {self.default_device}')
            else:
                self.default_device = [d for d in available_devices if d.device_kind == 'cpu'][0]
                if not quiet:
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
        self.quiet = bool(quiet)
        self.safety_margin = float(safety_margin)
        self.w_u = float(w_u)
        
        # History for residual model (HISTORY_LENGTH = 10)
        HISTORY_LENGTH = 10
        self.state_history = jnp.zeros((HISTORY_LENGTH, 10), dtype=jnp.float32)
        self.control_history = jnp.zeros((HISTORY_LENGTH, 2), dtype=jnp.float32)

        self.render_utils = RenderUtils()
        self.waypoint_utils = WaypointUtils()

        self.angular_control = 0
        self.translational_control = 0
        self.control_index = 0

        self.dt = T_CONTROL
        self.batch_size = int(batch_size) if batch_size is not None else 8
        self.horizon = int(horizon) if horizon is not None else 30
        
        # RPGD specific parameters 
        self.elite_size = int(elite_size) if elite_size is not None else min(4, self.batch_size)
        # Keep at least one fresh (non-elite) sample so resampling never has size 0.
        self.elite_size = min(self.elite_size, max(1, self.batch_size - 1))
        self.gradient_steps = int(gradient_steps) if gradient_steps is not None else 10
        self.resampling_freq = 5 
        
        self.rollout_trajectories = None  # Store trajectories for rendering
        
        # Interpolation
        self.use_interpolation = False  
        self.num_interpolation_points = 1
        
        # Control smoothness parameters
        self.intra_horizon_smoothness_weight = 5.0  # Weight for smoothness within horizon
        self.angular_smoothness_weight = 1.0  # Relative weight for angular control smoothness
        self.translational_smoothness_weight = 0.1  # Relative weight for translational control smoothness
        
        # Control output smoothing (low pass filter). Safety filter uses 1.0 (no lag).
        self.control_smoothing_alpha = (
            float(control_smoothing_alpha) if control_smoothing_alpha is not None else 0.5
        )
        self.last_executed_angular = 0.0
        self.last_executed_translational = 0.0


        self.optimal_trajectory = np.zeros((self.horizon, NUMBER_OF_STATES), dtype=np.float32)
        self.optimal_control_sequence = np.zeros((self.horizon, 2), dtype=np.float32)
        self.config_optimizer = dict()
        self.config_optimizer["mpc_timestep"] = self.dt  # Fixed timestep for RPGD optimization
            
        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.last_Q_sq[:, 0] = 0.0  # Straight steering (0 steering angle)
        self.last_Q_sq[:, 1] = 1.0  # Acceleration (1 m/s²)
        vehicle = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
        self.car_params_array = vehicle.to_np_array().astype(np.float32)
        self.vehicle_width = float(vehicle.width)

        # RPGD state: maintain elite plans and their costs
        self.elite_plans = None
        self.elite_costs = None
        self.iteration_count = 0
        
        # Warmstart flag
        self.warmstart_done = False
        
        # Adam optimizer states for each trajectory 
        self.adam_m = None  # First moment estimates
        self.adam_v = None  # Second moment estimates  
        self.adam_step = 0  # Global step counter
        self.trajectory_ages = None  # Track how long each trajectory has been optimized
        
        # Interpolation setup 
        self.interpolation_inducing_points = jnp.linspace(0, self.horizon-1, self.num_interpolation_points, dtype=jnp.int32)
        
        # Ensure car params are on the selected device
        with jax.default_device(self.default_device):
            self.car_params_jax = jnp.array(self.car_params_array)

        self.key = jax.random.PRNGKey(0)
        
        # Pre-compile the main computation function
        if not self.quiet:
            print("Pre-compiling JAX functions...")
        self._precompile_functions()

    def _precompile_functions(self):
        """Pre-compile JAX functions to ensure they run on the selected device"""
        # Create dummy data for compilation
        dummy_state = jnp.zeros(10, dtype=jnp.float32)
        # Safety cost needs x/y/psi/vx and left/right border distances.
        dummy_waypoints = jnp.zeros((100, 10), dtype=jnp.float32)
        dummy_waypoints = dummy_waypoints.at[:, WP_D_RIGHT_IDX].set(1.0)
        dummy_waypoints = dummy_waypoints.at[:, WP_D_LEFT_IDX].set(1.0)
        dummy_nominal = jnp.zeros(2, dtype=jnp.float32)
        dummy_key = jax.random.PRNGKey(42)
        
        if not self.quiet:
            print("Compiling rpgd_process_observation_jax...")
        # Create dummy full control sequences (no interpolation)
        dummy_Q_batch = jnp.zeros((self.batch_size, self.horizon, 2), dtype=jnp.float32)
        dummy_adam_m = jnp.zeros_like(dummy_Q_batch)
        dummy_adam_v = jnp.zeros_like(dummy_Q_batch)
        
        # Create dummy history for compilation
        HISTORY_LENGTH = 10
        dummy_state_history = jnp.zeros((HISTORY_LENGTH, 10), dtype=jnp.float32)
        dummy_control_history = jnp.zeros((HISTORY_LENGTH, 2), dtype=jnp.float32)
        
        # Trigger compilation on the selected device
        dummy_dt = jnp.full(self.horizon, self.dt)  # Dummy timestep array for compilation
        _ = rpgd_process_observation_jax(
            dummy_state, dummy_Q_batch, self.batch_size, self.horizon,
            self.car_params_jax, dummy_waypoints, dummy_key, dummy_dt,
            nominal_control=dummy_nominal,
            safety_margin=self.safety_margin,
            vehicle_width=self.vehicle_width,
            w_u=self.w_u,
            execute_control_index=4,
            intra_horizon_smoothness_weight=self.intra_horizon_smoothness_weight,
            angular_smoothness_weight=self.angular_smoothness_weight,
            translational_smoothness_weight=self.translational_smoothness_weight,
            gradient_steps=self.gradient_steps,
            adam_m=dummy_adam_m, adam_v=dummy_adam_v, adam_step=0,
            state_history=dummy_state_history, control_history=dummy_control_history
        )
        if not self.quiet:
            print(f"JAX functions compiled and ready for execution on {self.default_device}!")

    def process_observation(self, controller_observation):
        # Ensure all data is on the default device
        with jax.default_device(self.default_device):
            s = jnp.array(self.get_car_state(controller_observation), dtype=jnp.float32)
            waypoints = jnp.array(controller_observation["next_waypoints"], dtype=jnp.float32)
            nominal_control = jnp.asarray(
                controller_observation.get("nominal_control", (0.0, 0.0)),
                dtype=jnp.float32,
            )

            self.key, subkey = jax.random.split(self.key)
            
            # RPGD Step 1: Sample/maintain population of full control sequences (no interpolation)
            if self.elite_plans is None or self.iteration_count % self.resampling_freq == 0:
                # Safety-filter population is centred on the desired command.
                Q_batch_sequence = self._initialize_or_resample_safety_sequences(
                    subkey, nominal_control
                )
                # Reset Adam states when resampling
                self._reset_adam_states_for_resampling()
            else:
                # Use existing elite plans + time shift (work with full sequences)
                Q_batch_sequence = self._time_shift_and_expand_elite_sequences(
                    subkey, nominal_control
                )
                # Update Adam states for time shifting
                self._update_adam_states_for_time_shift()
            
            # RPGD Step 2: Gradient optimization loop (operates on full sequences)
            Q_batch_sequence, total_cost_batch, Q_final_unused, adam_m_new, adam_v_new, adam_step_new = rpgd_process_observation_jax(
                s, Q_batch_sequence, self.batch_size, self.horizon,
                self.car_params_jax, waypoints, subkey, self.dt,
                nominal_control=nominal_control,
                safety_margin=self.safety_margin,
                vehicle_width=self.vehicle_width,
                w_u=self.w_u,
                execute_control_index=int(Settings.CONTROL_DELAY / self.dt),
                intra_horizon_smoothness_weight=self.intra_horizon_smoothness_weight,
                angular_smoothness_weight=self.angular_smoothness_weight,
                translational_smoothness_weight=self.translational_smoothness_weight,
                gradient_steps=self.gradient_steps,
                adam_m=self.adam_m, adam_v=self.adam_v, adam_step=self.adam_step,
                state_history=self.state_history, control_history=self.control_history
            )
            
            # Update Adam states
            self.adam_m = adam_m_new
            self.adam_v = adam_v_new
            self.adam_step = adam_step_new
            
            # RPGD Step 3: Select best plan
            best_plan_idx = jnp.argmin(total_cost_batch)
            Q_sequence = Q_batch_sequence[best_plan_idx]
            
            # Debug info (reduced frequency)
            # if self.iteration_count % 100 == 0:  # Print every 100 iterations
            #     print(f"RPGD Iteration {self.iteration_count}: Best cost = {float(total_cost_batch[best_plan_idx]):.3f}, Avg age = {float(jnp.mean(self.trajectory_ages)) if self.trajectory_ages is not None else 0:.1f}")
            
            # Update elite plans for next iteration (store full sequences)
            self._update_elite_plans_with_full_sequences(Q_batch_sequence, total_cost_batch)
            
            # Update trajectory ages
            self._update_trajectory_ages()
            
            # Compute optimal trajectory for visualization (using constant dt)
            intermediate_steps = self.dt / 0.01
            optimal_traj = car_steps_sequential_jax(s, Q_sequence, self.car_params_jax, self.dt, self.horizon, model_type=MODEL_TYPE, state_history=self.state_history, control_history=self.control_history, intermediate_steps=intermediate_steps)
            
            # Batch rollout using vmap over car_steps_sequential_jax
            # Note: Each trajectory in the batch uses the same initial history
            batch_rollout_fn = jax.vmap(lambda s_single, Q_single: car_steps_sequential_jax(
                s_single, Q_single, self.car_params_jax, self.dt, self.horizon, model_type=MODEL_TYPE, state_history=self.state_history, control_history=self.control_history
            ))
            state_batch_sequence = batch_rollout_fn(jnp.repeat(s[None, :], self.batch_size, axis=0), Q_batch_sequence)
            min_clearance_batch = jax.vmap(
                lambda traj: jnp.min(
                    jax.vmap(
                        lambda state: track_clearance_jax(
                            state,
                            waypoints,
                            self.car_params_jax[1],
                            self.vehicle_width,
                            self.safety_margin,
                        )
                    )(traj)
                )
            )(state_batch_sequence)

            # Move results back to CPU for rendering (if needed)
            self.rollout_trajectories = np.array(state_batch_sequence)
            self.candidate_control_sequences = np.array(Q_batch_sequence)
            self.candidate_min_clearances = np.array(min_clearance_batch)
            self.trajectory_costs = np.array(total_cost_batch)
            
            self.optimal_trajectory = np.array(optimal_traj)
            self.optimal_control_sequence = np.array(Q_sequence)
            
            self.render_utils.update_mpc(
                rollout_trajectory=self.rollout_trajectories,
                optimal_trajectory=np.expand_dims(np.array(optimal_traj), axis=0),
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
            
            # Update state and control history for residual model
            # Roll history forward and add current state/control
            executed_control = jnp.array([self.angular_control, self.translational_control], dtype=jnp.float32)
            self.state_history = jnp.roll(self.state_history, -1, axis=0)
            self.state_history = self.state_history.at[-1, :].set(s)
            self.control_history = jnp.roll(self.control_history, -1, axis=0)
            self.control_history = self.control_history.at[-1, :].set(executed_control)
            
            self.last_Q_sq = np.array(Q_sequence)
            self.control_index += 1
            self.iteration_count += 1

            return float(self.angular_control), float(self.translational_control)


    def _update_elite_plans(self, Q_batch_sequence, total_cost_batch):
        """Update elite plans based on costs"""
        # Get indices of elite plans (lowest cost)
        elite_indices = jnp.argsort(total_cost_batch)[:self.elite_size]
        self.elite_plans = Q_batch_sequence[elite_indices]
        self.elite_costs = total_cost_batch[elite_indices]

    def _interpolate_control_sequence(self, Q_inducing):
        """Interpolate control sequence from inducing points (like original RPGD)"""
        # Q_inducing shape: (batch, num_interpolation_points, 2)
        # Output shape: (batch, horizon, 2)
        
        # Create interpolation indices
        inducing_indices = jnp.linspace(0, self.horizon-1, self.num_interpolation_points)
        target_indices = jnp.arange(self.horizon, dtype=jnp.float32)
        
        def interpolate_single_batch(q_inducing):
            # Interpolate each control dimension separately
            angular_interp = jnp.interp(target_indices, inducing_indices, q_inducing[:, 0])
            trans_interp = jnp.interp(target_indices, inducing_indices, q_inducing[:, 1])
            return jnp.stack([angular_interp, trans_interp], axis=1)
        
        return jax.vmap(interpolate_single_batch)(Q_inducing)

    def _sample_inducing_points(self, key, batch_size):
        """Sample inducing points for interpolation - match config uniform_dist_min/max"""
        return jax.random.uniform(key, (batch_size, self.num_interpolation_points, 2), 
                                 minval=jnp.array([-0.4, -10]), # Match config: uniform_dist_min: [-0.4, -10]
                                 maxval=jnp.array([0.4, 10]))   # Match config: uniform_dist_max: [0.4, 10]

    def _sample_full_sequences(self, key, batch_size, warmstart_sequence=None):
        """Sample full control sequences - with optional warmstart"""
        if warmstart_sequence is not None:
            # Use warmstart sequence as base for all trajectories
            base_sequences = jnp.tile(warmstart_sequence[None, :, :], (batch_size, 1, 1))
            # Add exploration noise
            noise = jax.random.normal(key, (batch_size, self.horizon, 2)) * jnp.array([0.05, 0.3])
            sequences = base_sequences + noise
        else:
            # Original initialization: Start with straight steering (0) and moderate acceleration (1 m/s²)
            base_control = jnp.array([0.0, 1.0])
            # Add some noise around this base for exploration
            noise = jax.random.normal(key, (batch_size, self.horizon, 2)) * jnp.array([0.1, 0.5])
            sequences = jnp.tile(base_control, (batch_size, self.horizon, 1)) + noise
        
        # Clip to valid bounds
        return jnp.clip(sequences, 
                       jnp.array([-0.4, -10]), # Match config: uniform_dist_min: [-0.4, -10]
                       jnp.array([0.4, 10]))   # Match config: uniform_dist_max: [0.4, 10]

    def _initialize_or_resample_full_sequences(self, key, current_state=None, waypoints=None):
        """Initialize full sequences or resample non-elite ones (no interpolation)"""
        if self.elite_plans is None:
            # First iteration: use warmstart if available
            warmstart_seq = None
            if not self.warmstart_done and current_state is not None and waypoints is not None:
                warmstart_seq = self._generate_warmstart_control_sequence(current_state, waypoints)
                warmstart_seq = jnp.array(warmstart_seq)
                self.warmstart_done = True
                if not self.quiet:
                    print("RPGD: Using warmstart control sequence for initial trajectory generation")
            
            Q_batch_sequence = self._sample_full_sequences(key, self.batch_size, warmstart_seq)
        else:
            # Keep elite sequences, resample the rest
            num_new = self.batch_size - self.elite_size
            key1, key2 = jax.random.split(key)
            new_sequences = self._sample_full_sequences(key1, num_new, None)
            Q_batch_sequence = jnp.concatenate([self.elite_plans, new_sequences], axis=0)
        return Q_batch_sequence

    def _sample_safety_sequences(self, key, batch_size, nominal_control):
        """Sample smooth plans around u_d; candidate zero is exactly u_d."""
        nominal = jnp.asarray(nominal_control, dtype=jnp.float32)
        if int(batch_size) <= 0:
            return jnp.zeros((0, self.horizon, 2), dtype=jnp.float32)
        base = jnp.tile(nominal[None, None, :], (batch_size, self.horizon, 1))
        noise = jax.random.normal(key, base.shape) * jnp.array([0.06, 0.8])
        sequences = jnp.clip(
            base + noise,
            jnp.array([-0.4, -10.0]),
            jnp.array([0.4, 10.0]),
        )
        return sequences.at[0].set(base[0])

    def _initialize_or_resample_safety_sequences(self, key, nominal_control):
        if self.elite_plans is None:
            return self._sample_safety_sequences(
                key, self.batch_size, nominal_control
            )
        num_new = self.batch_size - self.elite_size
        new_sequences = self._sample_safety_sequences(
            key, num_new, nominal_control
        )
        return jnp.concatenate([self.elite_plans, new_sequences], axis=0)
    
    def _time_shift_and_expand_elite_sequences(self, key, nominal_control=None):
        """Time shift elite sequences and expand with new random ones """
        # Time shift elite plans (shift by 3 steps as in config)
        shift_steps = 2  # Why 2 ? 2 works best. Discuss
        shifted_elite = jnp.roll(self.elite_plans, shift=-shift_steps, axis=1)
        # Fill the last shift_steps with the last control value + small noise
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, (self.elite_size, shift_steps, 2)) * 0.1
        last_controls = jnp.tile(self.elite_plans[:, -1:, :], (1, shift_steps, 1))
        shifted_elite = shifted_elite.at[:, -shift_steps:].set(
            jnp.clip(last_controls + noise, jnp.array([-0.4, -10]), jnp.array([0.4, 10]))
        )
        
        # Add new random sequences
        num_new = self.batch_size - self.elite_size
        if nominal_control is None:
            new_sequences = self._sample_full_sequences(key2, num_new, None)
        else:
            new_sequences = self._sample_safety_sequences(
                key2, num_new, nominal_control
            )
        
        return jnp.concatenate([shifted_elite, new_sequences], axis=0)
    
    def _update_elite_plans_with_full_sequences(self, Q_batch_sequence, total_cost_batch):
        """Update elite plans based on costs (store full sequences)"""
        # Get indices of elite plans (lowest cost)
        elite_indices = jnp.argsort(total_cost_batch)[:self.elite_size]
        self.elite_plans = Q_batch_sequence[elite_indices]
        self.elite_costs = total_cost_batch[elite_indices]

    def _initialize_or_resample_inducing_points(self, key):
        """Initialize inducing points or resample non-elite ones"""
        if self.elite_plans is None:
            # First iteration: sample all inducing points randomly
            Q_inducing_batch = self._sample_inducing_points(key, self.batch_size)
        else:
            # Keep elite inducing points, resample the rest
            num_new = self.batch_size - self.elite_size
            key1, key2 = jax.random.split(key)
            new_inducing_points = self._sample_inducing_points(key1, num_new)
            Q_inducing_batch = jnp.concatenate([self.elite_plans, new_inducing_points], axis=0)
        return Q_inducing_batch
    
    def _time_shift_and_expand_elite_inducing_points(self, key):
        """Time shift elite inducing points and expand with new random ones (like original RPGD)"""
        # Time shift is handled differently for inducing points
        # We keep the inducing points but adjust their influence
        # Add small noise to elite inducing points for exploration
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, self.elite_plans.shape) * 0.1  # Small exploration noise
        shifted_elite = jnp.clip(self.elite_plans + noise, 
                               jnp.array([-0.4, -10]), jnp.array([0.4, 10]))  # Match config bounds
        
        # Add new random inducing points
        num_new = self.batch_size - self.elite_size
        new_inducing_points = self._sample_inducing_points(key2, num_new)
        
        return jnp.concatenate([shifted_elite, new_inducing_points], axis=0)
    
    def _update_elite_plans_with_inducing_points(self, Q_inducing_final, total_cost_batch):
        """Update elite plans based on costs (store inducing points)"""
        # Get indices of elite plans (lowest cost)
        elite_indices = jnp.argsort(total_cost_batch)[:self.elite_size]
        self.elite_plans = Q_inducing_final[elite_indices]
        self.elite_costs = total_cost_batch[elite_indices]
        
    def _reset_adam_states_for_resampling(self):
        """Reset Adam states when resampling (like original RPGD) - for full sequences"""
        if self.elite_plans is not None:
            # Keep Adam states for elite plans, reset for new ones
            elite_indices = jnp.arange(self.elite_size)
            if self.adam_m is not None:
                # Keep elite states, add zeros for new plans
                elite_m = self.adam_m[elite_indices] if self.adam_m is not None else None
                new_m = jnp.zeros((self.batch_size - self.elite_size, self.horizon, 2))  # Full sequences
                self.adam_m = jnp.concatenate([elite_m, new_m], axis=0) if elite_m is not None else new_m
                
                elite_v = self.adam_v[elite_indices] if self.adam_v is not None else None
                new_v = jnp.zeros((self.batch_size - self.elite_size, self.horizon, 2))  # Full sequences
                self.adam_v = jnp.concatenate([elite_v, new_v], axis=0) if elite_v is not None else new_v
        else:
            # Complete reset for first iteration
            self.adam_m = jnp.zeros((self.batch_size, self.horizon, 2))  # Full sequences
            self.adam_v = jnp.zeros((self.batch_size, self.horizon, 2))  # Full sequences
            self.adam_step = 0
            
    def _update_adam_states_for_time_shift(self):
        """Update Adam states when time shifting (match original RPGD warmstarting)"""
        # For full sequences, we need to time shift Adam states like the original RPGD
        if self.adam_m is not None:
            shift_steps = 3  # Match config: shift_previous: 3
            # Time shift Adam states
            self.adam_m = jnp.roll(self.adam_m, shift=-shift_steps, axis=1)
            self.adam_v = jnp.roll(self.adam_v, shift=-shift_steps, axis=1)
            # Zero out the last shift_steps to reset them
            self.adam_m = self.adam_m.at[:, -shift_steps:].set(0.0)
            self.adam_v = self.adam_v.at[:, -shift_steps:].set(0.0)
            
    def _update_trajectory_ages(self):
        """Update trajectory ages (track optimization history like original RPGD)"""
        if self.trajectory_ages is None:
            self.trajectory_ages = jnp.zeros(self.batch_size)
        else:
            # Age all trajectories
            self.trajectory_ages = self.trajectory_ages + 1
            # Reset ages for resampled trajectories
            if self.iteration_count % self.resampling_freq == 0:
                # Elite trajectories keep their ages, new ones start at 0
                elite_ages = self.trajectory_ages[:self.elite_size]
                new_ages = jnp.zeros(self.batch_size - self.elite_size)
                self.trajectory_ages = jnp.concatenate([elite_ages, new_ages], axis=0)

    def _generate_warmstart_control_sequence(self, current_state, waypoints):
        """Generate a warmstart control sequence based on waypoints and current state"""
        controls = np.zeros((self.horizon, 2), dtype=np.float32)
        
        # Extract current position and heading
        current_x = float(current_state[POSE_X_IDX])
        current_y = float(current_state[POSE_Y_IDX])
        current_heading = float(current_state[POSE_THETA_IDX])
        current_speed = float(current_state[LINEAR_VEL_X_IDX])
        
        # Convert waypoints to numpy for easier handling
        wp_array = np.array(waypoints)
        
        # Find nearest waypoint
        distances = np.sqrt((wp_array[:, 1] - current_x)**2 + (wp_array[:, 2] - current_y)**2)
        nearest_idx = np.argmin(distances)
        
        # Generate control sequence
        for i in range(self.horizon):
            # Look ahead waypoint (with some lookahead distance)
            lookahead_idx = min(nearest_idx + i + 5, len(wp_array) - 1)  # Look ahead 5 waypoints
            target_x = wp_array[lookahead_idx, 1]
            target_y = wp_array[lookahead_idx, 2]
            target_speed = wp_array[lookahead_idx, 5]
            
            # Compute desired heading to target
            dx = target_x - current_x
            dy = target_y - current_y
            desired_heading = np.arctan2(dy, dx)
            
            # Compute steering angle (simple proportional controller)
            heading_error = desired_heading - current_heading
            # Normalize heading error to [-pi, pi]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # Steering control (proportional)
            steering = np.clip(heading_error * 0.5, -0.3, 0.3)  # Conservative steering
            
            # Speed control (simple proportional to target speed)
            speed_error = target_speed - current_speed
            acceleration = np.clip(speed_error * 2.0, -3.0, 3.0)  # Conservative acceleration
            
            controls[i, 0] = steering
            controls[i, 1] = acceleration
            
            # Update position estimate for next step (simple forward simulation)
            dt = self.dt
            current_x += current_speed * np.cos(current_heading) * dt
            current_y += current_speed * np.sin(current_heading) * dt
            current_heading += current_speed * np.tan(steering) / 2.5 * dt  # Approximate bicycle model
            current_speed += acceleration * dt
            current_speed = np.clip(current_speed, 0.1, 15.0)  # Reasonable speed bounds
        
        return controls


@jax.jit
def compute_waypoint_distance_jax(state, waypoints):
    dx = state[POSE_X_IDX] - waypoints[:, 1]
    dy = state[POSE_Y_IDX] - waypoints[:, 2]
    dist_sq = dx ** 2 + dy ** 2
    min_dist_sq = jnp.min(dist_sq)
    min_idx = jnp.argmin(dist_sq)
    return min_dist_sq, min_idx


@jax.jit
def track_clearance_jax(
    state,
    waypoints,
    lf,
    vehicle_width,
    safety_margin,
):
    """Minimum front-corner clearance using nearest-segment track projection."""
    x = state[POSE_X_IDX]
    y = state[POSE_Y_IDX]
    p0 = waypoints[:-1, (WP_X_IDX, WP_Y_IDX)]
    p1 = waypoints[1:, (WP_X_IDX, WP_Y_IDX)]
    v = p1 - p0
    seg_len2 = jnp.sum(v * v, axis=1) + 1e-12
    pt = jnp.array([x, y], dtype=jnp.float32)
    w = pt[None, :] - p0
    t = jnp.sum(w * v, axis=1) / seg_len2
    t = jnp.clip(t, 0.0, 1.0)
    proj = p0 + t[:, None] * v
    diff = pt[None, :] - proj
    dist2 = jnp.sum(diff * diff, axis=1)
    seg_idx = jnp.argmin(dist2)

    t_best = t[seg_idx]
    closest = proj[seg_idx]
    v_seg = v[seg_idx]
    psi = jnp.arctan2(v_seg[1], v_seg[0])
    offset = pt - closest
    n_x = -jnp.sin(psi)
    n_y = jnp.cos(psi)
    e_lat = offset[0] * n_x + offset[1] * n_y
    mu = jnp.arctan2(
        jnp.sin(state[POSE_THETA_IDX] - psi),
        jnp.cos(state[POSE_THETA_IDX] - psi),
    )

    half_width = 0.5 * vehicle_width
    e_lf = e_lat + lf * jnp.sin(mu) + half_width * jnp.cos(mu)
    e_rf = e_lat + lf * jnp.sin(mu) - half_width * jnp.cos(mu)
    seg_ip1 = jnp.minimum(seg_idx + 1, waypoints.shape[0] - 1)
    left = (
        (1.0 - t_best) * waypoints[seg_idx, WP_D_LEFT_IDX]
        + t_best * waypoints[seg_ip1, WP_D_LEFT_IDX]
        - safety_margin
    )
    right = (
        (1.0 - t_best) * waypoints[seg_idx, WP_D_RIGHT_IDX]
        + t_best * waypoints[seg_ip1, WP_D_RIGHT_IDX]
        - safety_margin
    )
    return jnp.minimum(
        jnp.minimum(left - e_lf, right + e_lf),
        jnp.minimum(left - e_rf, right + e_rf),
    )


@jax.jit
def safety_state_cost_jax(
    state,
    waypoints,
    lf,
    lr,
    vehicle_width,
    safety_margin,
):
    """Differentiable soft version of the intermediate state constraints."""
    _, min_idx = compute_waypoint_distance_jax(state, waypoints)
    wp = waypoints[min_idx]
    min_clearance = track_clearance_jax(
        state, waypoints, lf, vehicle_width, safety_margin
    )
    border_violation = jax.nn.relu(-min_clearance)

    # The waypoint velocity is a ceiling, not a tracking target.
    speed_excess = jax.nn.relu(state[LINEAR_VEL_X_IDX] - wp[WP_VX_IDX])
    v_x_safe = jnp.maximum(state[LINEAR_VEL_X_IDX], 0.1)
    v_y_rear = state[LINEAR_VEL_Y_IDX] - state[ANGULAR_VEL_Z_IDX] * lr
    slip = v_y_rear / v_x_safe

    return (
        2.0e4 * border_violation**2
        + 20.0 * speed_excess**2
        + 0.1 * slip**2
    )


@jax.jit
def terminal_safety_cost_jax(state, waypoints):
    """Terminal set: slow / stopped anywhere on track (no raceline pull)."""
    del waypoints
    speed_sq = state[LINEAR_VEL_X_IDX] ** 2 + state[LINEAR_VEL_Y_IDX] ** 2
    spin_sq = state[ANGULAR_VEL_Z_IDX] ** 2
    return 80.0 * speed_sq + 5.0 * spin_sq


@jax.jit
def cost_function_sequence_jax(
                              state_sequence, control_sequence, waypoints, lr,
                              lf, nominal_control, safety_margin, vehicle_width,
                              w_u,
                              intra_horizon_smoothness_weight=2.0, 
                              angular_smoothness_weight=1.0, 
                              translational_smoothness_weight=0.1):
    state_cost_fn = lambda s: safety_state_cost_jax(
        s, waypoints, lf, lr, vehicle_width, safety_margin
    )
    total_costs = jax.vmap(state_cost_fn)(state_sequence)

    # Tearle objective: minimally alter the command applied now.
    total_costs = total_costs.at[0].add(
        w_u * jnp.sum((control_sequence[0] - nominal_control) ** 2)
    )
    
    # Intra-horizon smoothness penalty - penalize sudden changes within the control sequence
    control_diff = control_sequence[1:] - control_sequence[:-1]
    
    # Individual step smoothness costs
    step_angular_smoothness = control_diff[:, 0] ** 2 * intra_horizon_smoothness_weight * angular_smoothness_weight
    step_translational_smoothness = control_diff[:, 1] ** 2 * intra_horizon_smoothness_weight * translational_smoothness_weight
    step_smoothness_costs = 10 * step_angular_smoothness + step_translational_smoothness
    
    # Add individual smoothness penalties to corresponding cost elements
    total_costs = total_costs.at[1:].add(step_smoothness_costs)
    
    # Also add overall smoothness penalty to first element for extra emphasis
    total_smoothness_penalty = jnp.sum(step_smoothness_costs) * 0.1  # Small additional overall penalty
    total_costs = total_costs.at[0].add(total_smoothness_penalty)

    # Layer 2: prefer a slow/stopped terminal state (anywhere on track).
    total_costs = total_costs.at[-1].add(
        terminal_safety_cost_jax(state_sequence[-1], waypoints)
    )
    
    return total_costs

@jax.jit
def rpgd_select_best_plan_jax(Q_batch_sequence, total_cost_batch):
    """RPGD Step 3: Select the plan with lowest cost"""
    best_idx = jnp.argmin(total_cost_batch)
    return Q_batch_sequence[best_idx], best_idx


@partial(jax.jit, static_argnames=["batch_size", "horizon", "execute_control_index", "gradient_steps"])
def rpgd_process_observation_jax(state, Q_batch_sequence, batch_size, horizon, car_params, waypoints, key, dt,
                               nominal_control, safety_margin, vehicle_width, w_u,
                               execute_control_index=4,
                               intra_horizon_smoothness_weight=2.0,
                               angular_smoothness_weight=1.0,
                               translational_smoothness_weight=0.1,
                               gradient_steps=5,
                               adam_m=None, adam_v=None, adam_step=0,
                               state_history=None, control_history=None):
    """
    RPGD Step 2: Gradient optimization - SIMPLIFIED to match original exactly
    NO interpolation (period_interpolation_inducing_points: 1 means no interpolation)
    """
    
    # Initialize Adam states if not provided
    if adam_m is None:
        adam_m = jnp.zeros_like(Q_batch_sequence)
    if adam_v is None:
        adam_v = jnp.zeros_like(Q_batch_sequence)
    
    # RPGD hyperparameters (match config_optimizers.yml exactly)
    learning_rate = 0.01  # Match config: learning_rate: 0.01
    beta1, beta2, eps = 0.9, 0.999, 1e-8  # Match config: adam_beta_1: 0.9, adam_beta_2: 0.999, adam_epsilon: 1.0e-08
    gradmax_clip = 5.0  # Match config: gradmax_clip: 5
    
    # Initialize histories if None
    if state_history is None:
        HISTORY_LENGTH = 10
        state_history = jnp.zeros((HISTORY_LENGTH, 10), dtype=jnp.float32)
    if control_history is None:
        HISTORY_LENGTH = 10
        control_history = jnp.zeros((HISTORY_LENGTH, 2), dtype=jnp.float32)

    lf = car_params[1]
    lr = car_params[2]
    
    # Cost function for gradient computation (operates directly on full sequences)
    def gradient_cost_fn(Q_full):
        # Rollout with constant timestep (use literal value for static compilation)
        trajectory = car_steps_sequential_jax(state, Q_full, car_params, T_CONTROL, horizon=horizon, model_type=MODEL_TYPE, state_history=state_history, control_history=control_history)
        # Compute cost
        costs = cost_function_sequence_jax(trajectory, Q_full, waypoints, lr,
                                         lf, nominal_control, safety_margin, vehicle_width,
                                         w_u,
                                         intra_horizon_smoothness_weight,
                                         angular_smoothness_weight, 
                                         translational_smoothness_weight)
        return jnp.sum(costs)
    
    def gradient_step(carry, step_idx):
        Q_batch, m_state, v_state, step_count = carry
        
        # Compute gradients w.r.t. full sequences (like original RPGD)
        gradients = jax.vmap(jax.grad(gradient_cost_fn))(Q_batch)
        
        # Clip gradients by norm (match original RPGD)
        gradients = jnp.clip(gradients, -gradmax_clip, gradmax_clip)
        
        # Update step counter
        t = step_count + 1
        
        # Adam update
        m_new = beta1 * m_state + (1 - beta1) * gradients
        v_new = beta2 * v_state + (1 - beta2) * (gradients ** 2)
        
        # Bias correction
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        
        # Parameter update
        Q_batch_new = Q_batch - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        
        # Clip to bounds (match config uniform_dist_min/max)
        Q_batch_new = jnp.clip(Q_batch_new, jnp.array([-0.4, -10]), jnp.array([0.4, 10]))
        
        return (Q_batch_new, m_new, v_new, t), None
    
    # Run gradient descent loop
    (Q_final, adam_m_final, adam_v_final, final_step), _ = jax.lax.scan(
        gradient_step, (Q_batch_sequence, adam_m, adam_v, adam_step), jnp.arange(gradient_steps))
    
    # Evaluate final costs
    def evaluation_cost_fn(Q_plan):
        trajectory = car_steps_sequential_jax(state, Q_plan, car_params, T_CONTROL, horizon=horizon, model_type=MODEL_TYPE, state_history=state_history, control_history=control_history)
        costs = cost_function_sequence_jax(trajectory, Q_plan, waypoints, lr,
                                         lf, nominal_control, safety_margin, vehicle_width,
                                         w_u,
                                         intra_horizon_smoothness_weight,
                                         angular_smoothness_weight, 
                                         translational_smoothness_weight)
        return jnp.sum(costs)
    
    final_costs = jax.vmap(evaluation_cost_fn)(Q_final)
    
    return Q_final, final_costs, Q_final, adam_m_final, adam_v_final, final_step


