import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import optax

from utilities.waypoint_utils import *
from utilities.render_utilities import RenderUtils
from Control_Toolkit_ASF.Controllers import template_planner
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import NUMBER_OF_STATES, POSE_X_IDX, POSE_Y_IDX, LINEAR_VEL_X_IDX

# Configure JAX for optimal GPU usage
jax.config.update('jax_enable_x64', False)  # Use 32-bit for better GPU performance
jax.config.update('jax_default_device', jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0])

class MPPILitePlanner(template_planner):

    def __init__(self):
        print('Loading JAX MPPI Lite Planner')
        print(f'JAX devices available: {jax.devices()}')
        print(f'Default JAX device: {jax.default_backend()}')
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
        self.batch_size = 2048
        self.horizon = 75

        self.last_Q_sq = np.zeros((self.horizon, 2), dtype=np.float32)
        self.car_params_array = VehicleParameters().to_np_array().astype(np.float32)
        
        # Ensure car params are on GPU
        with jax.default_device(jax.devices()[0]):
            self.car_params_jax = jnp.array(self.car_params_array)

        self.key = jax.random.PRNGKey(0)
        
        # Pre-compile the main computation function to GPU
        print("Pre-compiling JAX functions on GPU...")
        self._precompile_functions()

        # Pre-compile the main computation function to GPU
        print("Pre-compiling JAX functions on GPU...")
        self._precompile_functions()

    def _precompile_functions(self):
        """Pre-compile JAX functions to ensure they run on GPU"""
        # Create dummy data for compilation
        dummy_state = jnp.zeros(10, dtype=jnp.float32)
        dummy_last_Q = jnp.zeros((self.horizon, 2), dtype=jnp.float32)
        dummy_waypoints = jnp.zeros((100, 6), dtype=jnp.float32)
        dummy_key = jax.random.PRNGKey(42)
        
        print("Compiling process_observation_jax...")
        # Trigger compilation on GPU
        _ = process_observation_jax(
            dummy_state, dummy_last_Q, self.batch_size, self.horizon,
            self.car_params_jax, dummy_waypoints, dummy_key
        )
        print("JAX functions compiled and ready for GPU execution!")

    def process_observation(self, ranges=None, ego_odom=None):
        # Ensure all data is on the default device (GPU)
        with jax.default_device(jax.devices()[0]):
            s = jnp.array(self.car_state, dtype=jnp.float32)
            waypoints = jnp.array(self.waypoint_utils.next_waypoints, dtype=jnp.float32)

            self.key, subkey = jax.random.split(self.key)
            Q_sequence, optimal_traj, state_batch_sequence, total_cost_batch = process_observation_jax(
                s, jnp.array(self.last_Q_sq), self.batch_size, self.horizon,
                self.car_params_jax, waypoints, subkey
            )

            # Todo for RPGD 
            # Q_sequence = refine_optimal_control_adam(Q_sequence, s, self.car_params_jax, waypoints, self.dt, self.horizon)

            # Move results back to CPU for rendering (if needed)
            self.render_utils.update_mpc(
                rollout_trajectory=np.array(state_batch_sequence),
                optimal_trajectory=np.expand_dims(np.array(optimal_traj), axis=0),
            )

            execute_control_index = 4
            self.angular_control, self.translational_control = Q_sequence[execute_control_index]
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
    angular_control_cost = jnp.abs(control[0]) * 5.0
    translational_control_cost = jnp.abs(control[1]) * 1.0
    waypoint_cost = waypoint_dist_sq * 5.0
    target_speed = waypoints[min_idx, 5]
    speed_cost = 0.1 * (state[LINEAR_VEL_X_IDX] - target_speed) ** 2
    return speed_cost + angular_control_cost + translational_control_cost + waypoint_cost

@jax.jit
def cost_function_sequence_jax(state_sequence, control_sequence, waypoints):
    cost_fn = lambda s, u: cost_function_jax(s, u, waypoints)
    return jax.vmap(cost_fn)(state_sequence, control_sequence)

@jax.jit
def cost_function_batch_sequence_jax(state_batch_sequence, Q_batch_sequence, waypoints, discount_factor=0.99):
    horizon = Q_batch_sequence.shape[1]
    discount = discount_factor ** jnp.arange(horizon - 1, -1, -1)
    cost_fn = lambda states, controls: cost_function_sequence_jax(states, controls, waypoints) * discount
    cost_batch = jax.vmap(cost_fn)(state_batch_sequence, Q_batch_sequence)
    return cost_batch

@jax.jit
def exponential_weighting_jax(Q_batch_sequence, total_cost_batch, temperature=0.25):
    weights = jax.nn.softmax(-total_cost_batch / temperature)
    Q_weighted = Q_batch_sequence * weights[:, None, None]
    Q_mean = jnp.sum(Q_weighted, axis=0)
    return Q_mean


def car_step_jax(state, control, car_params, dt):
    mu, lf, lr, h_cg, m, I_z, g_, B_f, C_f, D_f, E_f, B_r, C_r, D_r, E_r, \
    servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch = car_params

    psi_dot, v_x, v_y, psi, _, _, s_x, s_y, _, delta = state
    desired_steering_angle, translational_control = control

    steering_angle_difference = desired_steering_angle - delta
    delta_dot = jnp.where(jnp.abs(steering_angle_difference) > 0.0001,
                          steering_angle_difference * servo_p,
                          0.0)
    delta_dot = jnp.clip(delta_dot, sv_min, sv_max)

    v_x_dot = translational_control
    pos_limit = jnp.where(v_x > v_switch, a_max * v_switch / v_x, a_max)
    v_x_dot = jnp.minimum(v_x_dot, pos_limit)
    max_a_friction = mu * g_
    v_x_dot = jnp.clip(v_x_dot, -max_a_friction, max_a_friction)

    v_x_safe = jnp.where(v_x == 0.0, 1e-5, v_x)
    alpha_f = -jnp.arctan((v_y + psi_dot * lf) / v_x_safe) + delta
    alpha_r = -jnp.arctan((v_y - psi_dot * lr) / v_x_safe)

    F_zf = m * (-v_x_dot * h_cg + g_ * lr) / (lr + lf)
    F_zr = m * (v_x_dot * h_cg + g_ * lf) / (lr + lf)

    Fy_f = mu * F_zf * D_f * jnp.sin(C_f * jnp.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - jnp.arctan(B_f * alpha_f))))
    Fy_r = mu * F_zr * D_r * jnp.sin(C_r * jnp.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - jnp.arctan(B_r * alpha_r))))

    d_s_x = v_x * jnp.cos(psi) - v_y * jnp.sin(psi)
    d_s_y = v_x * jnp.sin(psi) + v_y * jnp.cos(psi)
    d_psi = psi_dot
    d_v_x = v_x_dot
    d_v_y = (Fy_r + Fy_f) / m - v_x * psi_dot
    d_psi_dot = (-lr * Fy_r + lf * Fy_f) / I_z

    s_x += dt * d_s_x
    s_y += dt * d_s_y
    delta = jnp.clip(delta + dt * delta_dot, s_min, s_max)
    v_x += dt * d_v_x
    v_y += dt * d_v_y
    psi += dt * d_psi
    psi_dot += dt * d_psi_dot

    low_speed_threshold, high_speed_threshold = 0.5, 3.0
    weight = (v_x - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
    weight = jnp.clip(weight, 0.0, 1.0)

    s_x_ks = s_x + dt * (v_x * jnp.cos(psi))
    s_y_ks = s_y + dt * (v_x * jnp.sin(psi))
    psi_ks = psi + dt * (v_x / lf * jnp.tan(delta))
    v_y_ks = 0.0

    s_x = (1.0 - weight) * s_x_ks + weight * s_x
    s_y = (1.0 - weight) * s_y_ks + weight * s_y
    psi = (1.0 - weight) * psi_ks + weight * psi
    v_y = (1.0 - weight) * v_y_ks + weight * v_y

    psi_sin = jnp.sin(psi)
    psi_cos = jnp.cos(psi)

    return jnp.array([psi_dot, v_x, v_y, psi, psi_cos, psi_sin, s_x, s_y, 0.0, delta], dtype=jnp.float32)


@partial(jax.jit, static_argnames=["dt", "horizon"])
def car_steps_sequential_jax(s0, Q_sequence, car_params, dt, horizon):
    def rollout_fn(state, control):
        next_state = car_step_jax(state, control, car_params, dt)
        return next_state, next_state
    _, trajectory = jax.lax.scan(rollout_fn, s0, Q_sequence)
    return trajectory

@jax.jit
def car_batch_sequence_jax(s_batch, Q_batch_sequence, car_params):
    dt = 0.02
    horizon = Q_batch_sequence.shape[1]
    rollout_fn = lambda s, Q: car_steps_sequential_jax(s, Q, car_params, dt=dt, horizon=horizon)
    return jax.vmap(rollout_fn)(s_batch, Q_batch_sequence)

@partial(jax.jit, static_argnames=["batch_size", "horizon"])
def process_observation_jax(state, last_Q_sq, batch_size, horizon, car_params, waypoints, key):
    # Ensure computation happens on GPU
    with jax.default_device(jax.devices()[0]):
        s_batch = jnp.repeat(state[None, :], batch_size, axis=0)
        base_Q = jnp.roll(last_Q_sq, shift=-1, axis=0).at[-1].set(jax.random.normal(key, (2,), dtype=jnp.float32))
        Q_batch_sequence = jnp.repeat(base_Q[None, :, :], batch_size, axis=0)
        key1, key2 = jax.random.split(key)
        noise = jnp.stack([
            jax.random.normal(key1, (batch_size, horizon)) * 0.2,
            jax.random.normal(key2, (batch_size, horizon)) * 1.2
        ], axis=-1)
        Q_batch_sequence += noise
        Q_batch_sequence = jnp.clip(Q_batch_sequence, jnp.array([-0.4, -5.0]), jnp.array([0.4, 10.0]))
        traj_batch = car_batch_sequence_jax(s_batch, Q_batch_sequence, car_params)
        cost_batch = cost_function_batch_sequence_jax(traj_batch, Q_batch_sequence, waypoints)
        total_cost = jnp.sum(cost_batch, axis=1)
        Q_sequence = exponential_weighting_jax(Q_batch_sequence, total_cost)
        optimal_trajectory = car_steps_sequential_jax(state, Q_sequence, car_params, 0.02, horizon)
        return Q_sequence, optimal_trajectory, traj_batch, total_cost



@partial(jax.jit, static_argnames=["dt", "horizon", "num_steps"])
def refine_optimal_control_adam(Q_init, s0, car_params, waypoints, dt, horizon, num_steps=2, lr=0.01):
    opt = optax.adam(lr)
    opt_state = opt.init(Q_init)

    def cost_fn(Q_seq):
        traj = car_steps_sequential_jax(s0, Q_seq, car_params, dt=dt, horizon=horizon)
        costs = cost_function_sequence_jax(traj, Q_seq, waypoints)
        return jnp.sum(costs)

    def step(Q_seq, opt_state):
        loss, grad = jax.value_and_grad(cost_fn)(Q_seq)
        updates, opt_state = opt.update(grad, opt_state)
        Q_seq = optax.apply_updates(Q_seq, updates)
        return Q_seq, opt_state, loss, jnp.linalg.norm(grad)

    def scan_step(carry, step_idx):
        Q_seq, opt_state = carry
        Q_seq, opt_state, cost, grad_norm = step(Q_seq, opt_state)
        # jax.debug.print("Adam Step {0}: Cost = {1:.4f}, Grad Norm = {2:.4f}", step_idx, cost, grad_norm)
        return (Q_seq, opt_state), None

    (Q_final, _), _ = jax.lax.scan(scan_step, (Q_init, opt_state), jnp.arange(num_steps))
    return Q_final
