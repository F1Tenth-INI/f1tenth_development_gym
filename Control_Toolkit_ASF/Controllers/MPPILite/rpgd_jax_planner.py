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
from utilities.controller_utilities import ControllerUtilities
from utilities.state_utilities import (
    NUMBER_OF_STATES,
    POSE_X_IDX,
    POSE_Y_IDX,
    POSE_THETA_IDX,
    LINEAR_VEL_X_IDX,
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
class RPGDPlanner(template_planner):

    def __init__(self):
        print('Loading JAX RPGD Planner')
        
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
        
        # History for residual model (HISTORY_LENGTH = 10)
        HISTORY_LENGTH = 10
        self.state_history = jnp.zeros((HISTORY_LENGTH, NUMBER_OF_STATES), dtype=jnp.float32)
        self.control_history = jnp.zeros((HISTORY_LENGTH, 2), dtype=jnp.float32)

        self.render_utils = RenderUtils()
        self.waypoint_utils = WaypointUtils()

        self.angular_control = 0
        self.translational_control = 0
        self.control_index = 0

        self.dt = T_CONTROL
        # batch_size = 8 was way too small: with 6 elites and only 2 fresh
        # samples per resample, argmin over 8 candidates is dominated by
        # randomness and frequently picks brake-heavy plans even when the
        # speed cost says "go faster". 16 gives much steadier selection.
        self.batch_size = 16
        self.horizon = 30

        # RPGD specific parameters
        self.elite_size = 8  # roughly half the batch
        self.gradient_steps = 16  # 16 * lr=0.01 -> up to ~0.16 throttle change per cycle
        self.resampling_freq = 5
        
        self.rollout_trajectories = None  # Store trajectories for rendering
        
        # Interpolation
        self.use_interpolation = False  
        self.num_interpolation_points = 1
        
        # Control smoothness parameters. NOTE: weights are calibrated for
        # the normalized stick (steer/throttle in [-1, 1]). The angular
        # weight used to be 1.0 when steer was in [-0.4189, 0.4189] rad —
        # we scale it by s_max^2 so the per-step penalty magnitude stays
        # similar (otherwise the planner becomes ~6x more averse to
        # steering and can't corner).
        self.intra_horizon_smoothness_weight = 0.5
        self.angular_smoothness_weight = 0.1  # ~ 1.0 * 0.4189^2
        self.translational_smoothness_weight = 0.1
        
        # Output smoothing in normalized stick space. Was 0.5 (= halve every
        # plan) which compounded with the planner's own oscillations and
        # left the executed throttle at ~0.1 even with v_target=2 m/s.
        self.control_smoothing_alpha = 0.8
        self.last_executed_angular = 0.0
        self.last_executed_translational = 0.0


        self.optimal_trajectory = np.zeros((self.horizon, NUMBER_OF_STATES), dtype=np.float32)
        self.optimal_control_sequence = np.zeros((self.horizon, 2), dtype=np.float32)
        self.config_optimizer = dict()
        self.config_optimizer["mpc_timestep"] = self.dt  # Fixed timestep for RPGD optimization

        # Q is in normalized stick space [-1, 1] x [-1, 1] — same as what
        # the JAX dynamics consume. We convert back to physical units only
        # at the very end (see ControllerUtilities.normalized_to_physical),
        # so the env's _normalize_controls round-trips exactly.
        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.last_Q_sq[:, 0] = 0.0
        self.last_Q_sq[:, 1] = 0.5   # default: 50% forward stick

        self.controller_utils = ControllerUtilities()
        self.vehicle_parameters = self.controller_utils.vehicle_parameters
        self.car_params_array = self.vehicle_parameters.to_np_array().astype(np.float32)

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
        print("Pre-compiling JAX functions...")
        self._precompile_functions()

    def _precompile_functions(self):
        """Pre-compile JAX functions to ensure they run on the selected device"""
        # Create dummy data for compilation
        dummy_state = jnp.zeros(NUMBER_OF_STATES, dtype=jnp.float32)
        dummy_waypoints = jnp.zeros((100, 6), dtype=jnp.float32)
        dummy_key = jax.random.PRNGKey(42)
        
        print("Compiling rpgd_process_observation_jax...")
        # Create dummy full control sequences (no interpolation)
        dummy_Q_batch = jnp.zeros((self.batch_size, self.horizon, 2), dtype=jnp.float32)
        dummy_adam_m = jnp.zeros_like(dummy_Q_batch)
        dummy_adam_v = jnp.zeros_like(dummy_Q_batch)
        
        # Create dummy history for compilation
        HISTORY_LENGTH = 10
        dummy_state_history = jnp.zeros((HISTORY_LENGTH, NUMBER_OF_STATES), dtype=jnp.float32)
        dummy_control_history = jnp.zeros((HISTORY_LENGTH, 2), dtype=jnp.float32)
        
        # Trigger compilation on the selected device
        dummy_dt = jnp.full(self.horizon, self.dt)  # Dummy timestep array for compilation
        _ = rpgd_process_observation_jax(
            dummy_state, dummy_Q_batch, self.batch_size, self.horizon,
            self.car_params_jax, dummy_waypoints, dummy_key, dummy_dt,
            execute_control_index=4,
            intra_horizon_smoothness_weight=self.intra_horizon_smoothness_weight,
            angular_smoothness_weight=self.angular_smoothness_weight,
            translational_smoothness_weight=self.translational_smoothness_weight,
            gradient_steps=self.gradient_steps,
            adam_m=dummy_adam_m, adam_v=dummy_adam_v, adam_step=0,
            state_history=dummy_state_history, control_history=dummy_control_history
        )
        print(f"JAX functions compiled and ready for execution on {self.default_device}!")

    def process_observation(self, ranges=None, ego_odom=None):
        # Ensure all data is on the default device
        with jax.default_device(self.default_device):
            s = jnp.array(self.car_state, dtype=jnp.float32)
            waypoints = jnp.array(self.waypoint_utils.next_waypoints, dtype=jnp.float32)

            self.key, subkey = jax.random.split(self.key)
            
            # RPGD Step 1: Sample/maintain population of full control sequences (no interpolation)
            if self.elite_plans is None or self.iteration_count % self.resampling_freq == 0:
                # Initialize or resample full control sequences
                Q_batch_sequence = self._initialize_or_resample_full_sequences(subkey, s, waypoints)
                # Reset Adam states when resampling
                self._reset_adam_states_for_resampling()
            else:
                # Use existing elite plans + time shift (work with full sequences)
                Q_batch_sequence = self._time_shift_and_expand_elite_sequences(subkey)
                # Update Adam states for time shifting
                self._update_adam_states_for_time_shift()
            
            # RPGD Step 2: Gradient optimization loop (operates on full sequences)
            Q_batch_sequence, total_cost_batch, Q_final_unused, adam_m_new, adam_v_new, adam_step_new = rpgd_process_observation_jax(
                s, Q_batch_sequence, self.batch_size, self.horizon,
                self.car_params_jax, waypoints, subkey, self.dt,
                execute_control_index=0,
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
            optimal_traj = car_steps_sequential_jax(s, Q_sequence, self.car_params_jax, self.dt, self.horizon, model_type=MODEL_TYPE, state_history=self.state_history, control_history=self.control_history)
            
            # Batch rollout using vmap over car_steps_sequential_jax
            # Note: Each trajectory in the batch uses the same initial history
            batch_rollout_fn = jax.vmap(lambda s_single, Q_single: car_steps_sequential_jax(
                s_single, Q_single, self.car_params_jax, self.dt, self.horizon, model_type=MODEL_TYPE, state_history=self.state_history, control_history=self.control_history
            ))
            state_batch_sequence = batch_rollout_fn(jnp.repeat(s[None, :], self.batch_size, axis=0), Q_batch_sequence)

            # Move results back to CPU for rendering (if needed)
            self.rollout_trajectories = np.array(state_batch_sequence)
            self.trajectory_costs = np.array(total_cost_batch)
            
            self.optimal_trajectory = np.array(optimal_traj)
            # Storage convention for optimal_control_sequence: physical
            # units. car_system.extract_control_from_control_sequence reads
            # this and feeds it to update_pose unmodified, so it must match
            # what the env expects. The internal Q (last_Q_sq, elite_plans,
            # control_history, ...) stays in normalized stick space.
            self.optimal_control_sequence = self.controller_utils.normalized_sequence_to_physical(
                np.array(Q_sequence)
            )

            self.render_utils.update_mpc(
                rollout_trajectory=self.rollout_trajectories,
                optimal_trajectory=np.expand_dims(np.array(optimal_traj), axis=0),
            )

            execute_control_index = int(Settings.CONTROL_DELAY / self.dt)
            raw_angular_norm, raw_translational_norm = Q_sequence[execute_control_index]

            # Smooth in normalized space (so the smoothing constant is unit-
            # less and unaffected by drive_mode / per-car limits).
            steer_norm_smoothed = (
                self.control_smoothing_alpha * float(raw_angular_norm)
                + (1 - self.control_smoothing_alpha) * self.last_executed_angular
            )
            throttle_norm_smoothed = (
                self.control_smoothing_alpha * float(raw_translational_norm)
                + (1 - self.control_smoothing_alpha) * self.last_executed_translational
            )

            self.last_executed_angular = steer_norm_smoothed
            self.last_executed_translational = throttle_norm_smoothed

            # Update state and control history (in normalized space, same as
            # the elite plans we keep around).
            executed_control = jnp.array(
                [steer_norm_smoothed, throttle_norm_smoothed], dtype=jnp.float32
            )
            self.state_history = jnp.roll(self.state_history, -1, axis=0)
            self.state_history = self.state_history.at[-1, :].set(s)
            self.control_history = jnp.roll(self.control_history, -1, axis=0)
            self.control_history = self.control_history.at[-1, :].set(executed_control)

            self.last_Q_sq = np.array(Q_sequence)
            self.control_index += 1
            self.iteration_count += 1

            # Convert the normalized stick the planner optimized into the
            # physical units the env's update_pose expects. The env then
            # divides by the same per-mode scaling and feeds the dynamics
            # the same normalized stick we just optimized -> exact round
            # trip -> what we predicted is what the simulator does.
            steer_phys, throttle_phys = self.controller_utils.normalized_to_physical(
                steer_norm_smoothed, throttle_norm_smoothed
            )
            self.angular_control = steer_phys
            self.translational_control = throttle_phys

            # car_system.process_observation ignores the return value and
            # re-reads optimal_control_sequence[CONTROL_DELAY / dt]. That row
            # must be the same physical command we just computed (including
            # EMA smoothing), not the raw plan row.
            self.optimal_control_sequence[execute_control_index] = np.array(
                [steer_phys, throttle_phys], dtype=np.float32
            )

            return float(steer_phys), float(throttle_phys)


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
        """Sample inducing points uniformly in the normalized stick box."""
        return jax.random.uniform(
            key, (batch_size, self.num_interpolation_points, 2),
            minval=jnp.array([-1.0, -1.0]),
            maxval=jnp.array([1.0, 1.0]),
        )

    def _sample_full_sequences(self, key, batch_size, warmstart_sequence=None):
        """Sample full control sequences in normalized stick space.

        Throttle is biased forward (default 0.5, lower clip -0.3) so resamples
        don't keep injecting brake-heavy plans that the planner sometimes
        latches onto when the cost gradient is weak.
        """
        if warmstart_sequence is not None:
            base_sequences = jnp.tile(warmstart_sequence[None, :, :], (batch_size, 1, 1))
            noise = jax.random.normal(key, (batch_size, self.horizon, 2)) * jnp.array([0.15, 0.2])
            sequences = base_sequences + noise
        else:
            base_control = jnp.array([0.0, 0.5])
            noise = jax.random.normal(key, (batch_size, self.horizon, 2)) * jnp.array([0.1, 0.3])
            sequences = jnp.tile(base_control, (batch_size, self.horizon, 1)) + noise

        return jnp.clip(sequences, jnp.array([-1.0, -0.3]), jnp.array([1.0, 1.0]))

    def _initialize_or_resample_full_sequences(self, key, current_state=None, waypoints=None):
        """Initialize full sequences or resample non-elite ones (no interpolation)"""
        if self.elite_plans is None:
            # First iteration: use warmstart if available
            warmstart_seq = None
            if not self.warmstart_done and current_state is not None and waypoints is not None:
                warmstart_seq = self._generate_warmstart_control_sequence(current_state, waypoints)
                warmstart_seq = jnp.array(warmstart_seq)
                self.warmstart_done = True
                print("RPGD: Using warmstart control sequence for initial trajectory generation")
            
            Q_batch_sequence = self._sample_full_sequences(key, self.batch_size, warmstart_seq)
        else:
            # Keep elite sequences, resample the rest
            num_new = self.batch_size - self.elite_size
            key1, key2 = jax.random.split(key)
            new_sequences = self._sample_full_sequences(key1, num_new, None)
            Q_batch_sequence = jnp.concatenate([self.elite_plans, new_sequences], axis=0)
        return Q_batch_sequence
    
    def _time_shift_and_expand_elite_sequences(self, key):
        """Time shift elite sequences and expand with new random ones."""
        shift_steps = 2
        shifted_elite = jnp.roll(self.elite_plans, shift=-shift_steps, axis=1)
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, (self.elite_size, shift_steps, 2)) * 0.1
        last_controls = jnp.tile(self.elite_plans[:, -1:, :], (1, shift_steps, 1))
        shifted_elite = shifted_elite.at[:, -shift_steps:].set(
            jnp.clip(last_controls + noise, jnp.array([-1.0, -0.3]), jnp.array([1.0, 1.0]))
        )

        num_new = self.batch_size - self.elite_size
        new_sequences = self._sample_full_sequences(key2, num_new, None)

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
        """Time shift elite inducing points and expand with new random ones."""
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, self.elite_plans.shape) * 0.1
        shifted_elite = jnp.clip(
            self.elite_plans + noise,
            jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]),
        )

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
        """Warmstart in normalized stick space.

        Outputs a (horizon, 2) array in [-1, 1]^2 (steer_norm, throttle_norm).
        Internally we still simulate a rough bicycle model in physical units
        for state propagation, but the controls we *emit* are normalized.
        """
        s_max_phys = float(self.vehicle_parameters.s_max)
        a_max_phys = float(self.vehicle_parameters.a_max)
        v_max_phys = float(self.vehicle_parameters.v_max)
        wb = float(self.vehicle_parameters.lf + self.vehicle_parameters.lr)

        controls = np.zeros((self.horizon, 2), dtype=np.float32)

        current_x = float(current_state[POSE_X_IDX])
        current_y = float(current_state[POSE_Y_IDX])
        # State is sorted alphabetically (state_utilities.STATE_VARIABLES);
        # index 2 is linear_vel_y, NOT yaw. Use POSE_THETA_IDX explicitly.
        current_heading = float(current_state[POSE_THETA_IDX])
        current_speed = float(current_state[LINEAR_VEL_X_IDX])

        wp_array = np.array(waypoints)
        distances = np.sqrt((wp_array[:, 1] - current_x) ** 2 + (wp_array[:, 2] - current_y) ** 2)
        nearest_idx = np.argmin(distances)

        for i in range(self.horizon):
            lookahead_idx = min(nearest_idx + i + 5, len(wp_array) - 1)
            target_x = wp_array[lookahead_idx, 1]
            target_y = wp_array[lookahead_idx, 2]
            target_speed = wp_array[lookahead_idx, 5]

            dx = target_x - current_x
            dy = target_y - current_y
            desired_heading = np.arctan2(dy, dx)
            heading_error = desired_heading - current_heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

            # Normalized steering: a 0.6 rad heading error -> ~70% stick.
            steering_norm = float(np.clip(heading_error / max(s_max_phys, 1e-3) * 0.6, -0.7, 0.7))
            # Normalized throttle: 1 m/s under target -> ~40% stick.
            speed_error = target_speed - current_speed
            throttle_norm = float(np.clip(speed_error * 2.0 / max(a_max_phys, 1e-3), -0.6, 0.6))

            controls[i, 0] = steering_norm
            controls[i, 1] = throttle_norm

            # Forward-simulate (physical units, separate from emitted norm).
            dt = self.dt
            steer_phys = steering_norm * s_max_phys
            accel_phys = throttle_norm * a_max_phys
            current_x += current_speed * np.cos(current_heading) * dt
            current_y += current_speed * np.sin(current_heading) * dt
            current_heading += current_speed * np.tan(steer_phys) / max(wb, 0.1) * dt
            current_speed = float(np.clip(current_speed + accel_phys * dt, 0.1, v_max_phys))

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
def cost_function_jax(state, control, waypoints):
    """Per-step cost. control is in normalized stick space [-1, 1]^2.

    Includes an explicit progress signal (negative cost for advancing along
    the waypoint index) so a stationary car is NOT optimal even if it sits
    on the line. Without this, ``waypoint_dist_sq`` is minimised by not
    moving and the planner converges to a crawl.
    """
    waypoint_dist_sq, min_idx = compute_waypoint_distance_jax(state, waypoints)
    angular_control_cost = jnp.abs(control[0]) * 0.04
    translational_control_cost = jnp.abs(control[1]) * 0.0
    waypoint_cost = waypoint_dist_sq * 20.0
    target_speed = waypoints[min_idx, 5]
    speed_cost = 20.0 * (state[LINEAR_VEL_X_IDX] - target_speed) ** 2
    # Reward forward motion directly. Capped because target_speed already
    # provides a saturation point via speed_cost.
    progress_reward = -3.0 * jnp.clip(state[LINEAR_VEL_X_IDX], -2.0, 5.0)
    angular_quadratic_penalty = control[0] ** 2 * 1.76  # ~ 10 * 0.4189^2

    return (
        speed_cost
        + angular_control_cost
        + translational_control_cost
        + waypoint_cost
        + angular_quadratic_penalty
        + progress_reward
    )

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
    
    # Individual step smoothness costs. The legacy code multiplied the
    # angular term by an extra 10x, which on top of the radian-units
    # tuning made the planner extremely averse to changing throttle and
    # caused it to latch onto whatever throttle it started with.
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
def rpgd_select_best_plan_jax(Q_batch_sequence, total_cost_batch):
    """RPGD Step 3: Select the plan with lowest cost"""
    best_idx = jnp.argmin(total_cost_batch)
    return Q_batch_sequence[best_idx], best_idx


@partial(jax.jit, static_argnames=["batch_size", "horizon", "execute_control_index", "gradient_steps"])
def rpgd_process_observation_jax(state, Q_batch_sequence, batch_size, horizon, car_params, waypoints, key, dt,
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
        state_history = jnp.zeros((HISTORY_LENGTH, NUMBER_OF_STATES), dtype=jnp.float32)
    if control_history is None:
        HISTORY_LENGTH = 10
        control_history = jnp.zeros((HISTORY_LENGTH, 2), dtype=jnp.float32)
    
    # Cost function for gradient computation (operates directly on full sequences)
    def gradient_cost_fn(Q_full):
        # Rollout with constant timestep (use literal value for static compilation)
        trajectory = car_steps_sequential_jax(state, Q_full, car_params, T_CONTROL, horizon=horizon, model_type=MODEL_TYPE, state_history=state_history, control_history=control_history)
        # Compute cost
        costs = cost_function_sequence_jax(trajectory, Q_full, waypoints,
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

        # Clip to the normalized stick box.
        Q_batch_new = jnp.clip(Q_batch_new, jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0]))
        
        return (Q_batch_new, m_new, v_new, t), None
    
    # Run gradient descent loop
    (Q_final, adam_m_final, adam_v_final, final_step), _ = jax.lax.scan(
        gradient_step, (Q_batch_sequence, adam_m, adam_v, adam_step), jnp.arange(gradient_steps))
    
    # Evaluate final costs
    def evaluation_cost_fn(Q_plan):
        trajectory = car_steps_sequential_jax(state, Q_plan, car_params, T_CONTROL, horizon=horizon, model_type=MODEL_TYPE, state_history=state_history, control_history=control_history)
        costs = cost_function_sequence_jax(trajectory, Q_plan, waypoints,
                                         intra_horizon_smoothness_weight,
                                         angular_smoothness_weight, 
                                         translational_smoothness_weight)
        return jnp.sum(costs)
    
    final_costs = jax.vmap(evaluation_cost_fn)(Q_final)
    
    return Q_final, final_costs, Q_final, adam_m_final, adam_v_final, final_step


