import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax

from utilities.waypoint_utils import WaypointUtils
from utilities.render_utilities import RenderUtils
from utilities.Settings import Settings
from Control_Toolkit_ASF.Controllers import template_planner
from utilities.car_files.vehicle_parameters import VehicleParameters
from Control_Toolkit_ASF.Controllers.MPPILite.shared_cost_jax import (
    cost_function_batch_sequence_jax,
    cost_function_sequence_jax,
    load_wall_map,
    wall_distance_jax,
)

from sim.f110_sim.envs.car_model_jax import (car_steps_sequential_jax)


# Configure JAX for optimal GPU usage
jax.config.update('jax_enable_x64', False)  # Use 32-bit for better GPU performance
# jax.config.update('jax_default_device', jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0])

# Cost weights. Passed to the jitted functions as a pytree, so changing a value needs no recompile.
# Tuned on RCA1 (yokomo1 parameters, mu 0.5, CONTROL_DELAY 0.08, default state noise).
DEFAULT_COST_WEIGHTS = {
    'lateral': 5.0,           # (lateral offset from raceline [m])^2. Higher pulls the car onto the raceline, which passes close to wall tips
    'heading': 2.0,           # (heading error to raceline [rad])^2
    'speed': 0.0,             # (v - speed_factor * raceline speed)^2. Off: the raceline speeds exceed the grip in the RCA1 hairpins
    'speed_factor': 0.0,      # scales the raceline speed profile
    'progress': 3.0,          # reward per m/s driven along the raceline. Higher drives more raggedly and is slower
    'border': 200.0,          # (penetration into the border margin [m])^2
    'border_margin': 0.1,     # [m] distance of the car centre to the raceline's track edge (d_left / d_right) where the border cost starts
    'wall': 200.0,            # (penetration of the closest body corner into the wall margin [m])^2
    'wall_margin': 0.3,       # [m] wall distance of the closest body corner where the wall cost starts
    'crash': 1000.0,          # constant per step once a body corner is closer than crash_margin to a wall
    'crash_margin': 0.15,     # [m] the simulator crashes at 0.005. The rest is buffer for state noise and model error
    'angular_control': 0.1,   # |steering command|
    'translational_control': 0.01,  # |acceleration command|
    'angular_smoothness': 1.0,      # (steering change per step)^2
    'translational_smoothness': 0.01,  # (acceleration change per step)^2
    # Terminal cost on the last state of the horizon (see terminal_cost_jax). Off for MPPI
    'terminal_lateral': 25.0,        # (lateral offset from raceline [m])^2
    'terminal_heading': 5.0,         # (heading error to raceline [rad])^2
    'terminal_speed': 1000.0,        # (v - terminal_speed_factor * waypoint speed)^2, only above that speed. Slower is free: speed is rewarded by the stage progress cost
    'terminal_speed_factor': 1.0,    # the waypoint speed is the maximum terminal speed
    'terminal_heading_bound': 500.0, # (|heading error| - terminal_max_heading)^2, only beyond that bound
    'terminal_max_heading': 0.087,   # [rad] (5 deg)
    'temperature': 3.0,       # softmax temperature on (cost - min cost). Lower: one sample dominates and the steering jitters (0.1: ~40% more than 1.0)
    'filter_tolerance': 20.0, # the smoothed plan is used unless its cost exceeds the raw plan's cost by more than this
    'filter_clearance': 0.4,  # [m] ... and only while both plans keep their body corners this far from every wall
}

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

        self.dt = Settings.TIMESTEP_CONTROL
        self.batch_size = 512
        self.horizon = 35
        self.intermediate_steps = 2  # Euler substeps per control step. 1 is too coarse at the hairpins, 4 (sim resolution) is no better
        self.noise_std = (0.1, 0.6)  # [steering, acceleration] sampling noise
        # Odd window of a Savitzky-Golay filter (quadratic) on the weighted plan. 0: off. 9 makes the steering ~4x
        # calmer at equal lap time, but on RCA1 it raised the crash rate from ~2.5% to ~6-8% per lap (hairpin wall tips)
        self.plan_filter_window = 0

        # Controls already sent but not yet applied (CONTROL_DELAY). They are fixed at the start of every rollout.
        self.delay_steps = int(round(Settings.CONTROL_DELAY / self.dt))
        self.cost_weights = {k: jnp.float32(v) for k, v in DEFAULT_COST_WEIGHTS.items()}

        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.sent_controls = np.zeros((self.delay_steps, 2), dtype=np.float32)
        self.car_params_array = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE).to_np_array().astype(np.float32)
        if Settings.FRICTION_FOR_CONTROLLER is not None:
            self.car_params_array[0] = Settings.FRICTION_FOR_CONTROLLER

        # Commands beyond these limits saturate in the car model (steering angle limit, acceleration and
        # friction limits after the multipliers). Sampling outside them only makes the plan drift.
        vp = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
        mu = float(self.car_params_array[0])
        steering_limit = vp.s_max / vp.steering_multiplier
        accel_limit = min(vp.a_max, mu * vp.g) / vp.accel_multiplier
        brake_limit = max(vp.a_min, -mu * vp.g) / vp.brake_multiplier
        
        # Ensure car params are on the selected device
        with jax.default_device(self.default_device):
            self.car_params_jax = jnp.array(self.car_params_array)
            self.control_limits = jnp.array([[-steering_limit, brake_limit], [steering_limit, accel_limit]], dtype=jnp.float32)

        self.wall_map = load_wall_map(vp.length, vp.width, self.default_device)

        self.key = jax.random.PRNGKey(0)
        
        # Pre-compile the main computation function
        print("Pre-compiling JAX functions...")
        self._precompile_functions()

    def _precompile_functions(self):
        """Pre-compile JAX functions to ensure they run on the selected device"""
        # Create dummy data for compilation
        dummy_state = jnp.zeros(10, dtype=jnp.float32)
        dummy_last_Q = jnp.zeros((self.horizon, 2), dtype=jnp.float32)
        dummy_waypoints = jnp.zeros((Settings.LOOK_AHEAD_STEPS, 10), dtype=jnp.float32)
        dummy_key = jax.random.PRNGKey(42)
        
        print("Compiling process_observation_jax...")
        
        # Trigger compilation on the selected device
        _ = process_observation_jax(
            dummy_state, dummy_last_Q, jnp.array(self.sent_controls), self.batch_size, self.horizon,
            self.car_params_jax, dummy_waypoints, dummy_key, self.cost_weights, self.wall_map, self.control_limits,
            self.delay_steps, self.dt, self.intermediate_steps, self.noise_std, self.plan_filter_window
        )
        print(f"JAX functions compiled and ready for execution on {self.default_device}!")

    def reset(self):
        super().reset()
        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.sent_controls = np.zeros((self.delay_steps, 2), dtype=np.float32)

    def process_observation(self, controller_observation):
        # Ensure all data is on the default device
        with jax.default_device(self.default_device):
            s = jnp.array(self.get_car_state(controller_observation), dtype=jnp.float32)
            waypoints = jnp.array(controller_observation["next_waypoints"], dtype=jnp.float32)

            self.key, subkey = jax.random.split(self.key)

            Q_sequence, optimal_traj, state_batch_sequence, total_cost_batch = process_observation_jax(
                s, jnp.array(self.last_Q_sq), jnp.array(self.sent_controls), self.batch_size, self.horizon,
                self.car_params_jax, waypoints, subkey, self.cost_weights, self.wall_map, self.control_limits,
                self.delay_steps, self.dt, self.intermediate_steps, self.noise_std, self.plan_filter_window
            )

            # Move results back to CPU for rendering (if needed)
            self.rollout_trajectories = np.array(state_batch_sequence)
            self.trajectory_costs = np.array(total_cost_batch)

            self.optimal_trajectory = np.array(optimal_traj)
            self.render_utils.update_mpc(
                rollout_trajectory=self.rollout_trajectories[:20,:,:],
                optimal_trajectory=np.expand_dims(self.optimal_trajectory, axis=0),
            )

            # Q_sequence[:delay_steps] are the controls already in the delay buffer; this one is applied next
            Q_sequence = np.array(Q_sequence)
            self.angular_control, self.translational_control = Q_sequence[self.delay_steps]

            if self.delay_steps > 0:
                self.sent_controls = np.concatenate([self.sent_controls[1:], Q_sequence[self.delay_steps][None]], axis=0)
            self.last_Q_sq = Q_sequence
            self.control_index += 1

            return float(self.angular_control), float(self.translational_control)


@jax.jit
def exponential_weighting_jax(Q_batch_sequence, total_cost_batch, temperature):
    weights = jax.nn.softmax(-(total_cost_batch - jnp.min(total_cost_batch)) / temperature)
    return jnp.sum(Q_batch_sequence * weights[:, None, None], axis=0)


def savgol_filter_jax(Q_sequence, window):
    """Quadratic Savitzky-Golay smoothing along the horizon, edges padded by repetition."""
    offsets = np.arange(window) - window // 2
    kernel = np.linalg.pinv(np.vander(offsets, 3, increasing=True))[0]
    padded = jnp.pad(Q_sequence, ((window // 2, window // 2), (0, 0)), mode='edge')
    return jax.vmap(lambda c: jnp.convolve(c, jnp.array(kernel[::-1], dtype=jnp.float32), mode='valid'),
                    in_axes=1, out_axes=1)(padded)


@partial(jax.jit, static_argnames=["batch_size", "horizon", "delay_steps", "dt", "intermediate_steps", "noise_std",
                                   "plan_filter_window"])
def process_observation_jax(state, last_Q_sq, sent_controls, batch_size, horizon, car_params, waypoints, key, w, wall_map, control_limits,
                            delay_steps, dt, intermediate_steps, noise_std, plan_filter_window):
    s_batch = jnp.repeat(state[None, :], batch_size, axis=0)
    # Warm start: shift the last plan by one step and repeat its last control
    base_Q = jnp.concatenate([last_Q_sq[1:], last_Q_sq[-1:]], axis=0)

    Q_batch_sequence = jnp.repeat(base_Q[None, :, :], batch_size, axis=0)
    key1, key2 = jax.random.split(key)
    noise = jnp.stack([
        jax.random.normal(key1, (batch_size, horizon)) * noise_std[0],
        jax.random.normal(key2, (batch_size, horizon)) * noise_std[1]
    ], axis=-1)
    noise = noise.at[0].set(0.0)  # sample 0 re-evaluates the shifted last plan
    Q_batch_sequence = jnp.clip(Q_batch_sequence + noise, control_limits[0], control_limits[1])
    # The first delay_steps controls are already in the delay buffer
    Q_batch_sequence = Q_batch_sequence.at[:, :delay_steps].set(sent_controls)

    traj_batch = jax.vmap(lambda s_single, Q_single: car_steps_sequential_jax(
        s_single, Q_single, car_params, dt, horizon, model_type='pacejka',
        intermediate_steps=intermediate_steps
    ))(s_batch, Q_batch_sequence)

    cost_batch = cost_function_batch_sequence_jax(traj_batch, Q_batch_sequence, waypoints, w, wall_map)
    total_cost = jnp.sum(cost_batch, axis=1)
    Q_sequence = exponential_weighting_jax(Q_batch_sequence, total_cost, w['temperature'])
    if plan_filter_window > 1:
        # The smoothed plan was never evaluated: check it against the raw one, it may cut a hairpin too close.
        # It must not cost much more, and both plans must stay clear of the walls: near wall tips the raw plan steers sharper
        Q_smooth = savgol_filter_jax(Q_sequence, plan_filter_window)
        Q_smooth = jnp.clip(Q_smooth.at[:delay_steps].set(sent_controls), control_limits[0], control_limits[1])
        def plan_cost_and_clearance(Q):
            traj = car_steps_sequential_jax(state, Q, car_params, dt, horizon, model_type='pacejka',
                                            intermediate_steps=intermediate_steps)
            clearance = jnp.min(jax.vmap(lambda s: wall_distance_jax(s, wall_map))(traj))
            return jnp.sum(cost_function_sequence_jax(traj, Q, waypoints, w, wall_map)), clearance
        cost_smooth, clearance_smooth = plan_cost_and_clearance(Q_smooth)
        cost_raw, clearance_raw = plan_cost_and_clearance(Q_sequence)
        use_smooth = ((cost_smooth <= cost_raw + w['filter_tolerance'])
                      & (jnp.minimum(clearance_smooth, clearance_raw) >= w['filter_clearance']))
        Q_sequence = jnp.where(use_smooth, Q_smooth, Q_sequence)
    optimal_trajectory = car_steps_sequential_jax(state, Q_sequence, car_params, dt, horizon,
                                                  model_type='pacejka', intermediate_steps=intermediate_steps)
    return Q_sequence, optimal_trajectory, traj_batch, total_cost



@partial(jax.jit, static_argnames=["horizon"])
def refine_optimal_control_adam(Q_init, s0, car_params, waypoints, horizon, w, wall_map):

    # Alternative: Cosine decay schedule (often works better for optimization)

    gradient_steps = 20
    lr_max = 0.04  # Initial learning rate
    lr_min = 0.005

    gradmax_clip = 1.0

    lr_schedule = optax.cosine_decay_schedule(lr_max, decay_steps=gradient_steps, alpha=lr_min)
    
    opt = optax.adam(lr_schedule)
    opt_state = opt.init(Q_init)

    def cost_fn(Q_seq):
        traj = car_steps_sequential_jax(s0, Q_seq, car_params, 0.04, horizon=horizon, model_type='pacejka')
        costs = cost_function_sequence_jax(traj, Q_seq, waypoints, w, wall_map)
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
