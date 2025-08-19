# utilities/InverseDynamics.py
# Plausible past-trajectory reconstruction from current state and past controls.
# Warm-start with per-step LM; refine in short overlapping windows with robust losses.
# Scaling is hard-coded from your dataset's std row so you don't need to import any table.

from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple

###############################################################################
# Assumes these exist in your environment (same as before):
#  - TensorFlowLibrary
#  - car_model (your forward simulator; see ks/pacejka implementations)
#  - Settings
#  - POSE_THETA_IDX, SLIP_ANGLE_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX
###############################################################################
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit_ASF.car_model import car_model
from utilities.Settings import Settings
from utilities.state_utilities import (
    POSE_THETA_IDX,
    SLIP_ANGLE_IDX,
    POSE_THETA_SIN_IDX,
    POSE_THETA_COS_IDX,
)

# ======================================================================================
# Utilities
# ======================================================================================

def _angle_wrap_diff(thA: tf.Tensor, thB: tf.Tensor) -> tf.Tensor:
    """Wrap (θA-θB) into [-π, π] to avoid 2π-discontinuities in residuals."""
    thA = tf.cast(thA, tf.float32)
    thB = tf.cast(thB, tf.float32)
    return tf.atan2(tf.sin(thA - thB), tf.cos(thA - thB))


def _fix_sin_cos(x: tf.Tensor) -> tf.Tensor:
    """
    Project sin/cos channels to match the heading angle in x.
    This eliminates redundant degrees of freedom during optimization.
    x: shape [state_dim]
    """
    x = tf.cast(x, tf.float32)
    angle = x[POSE_THETA_IDX]
    sin_val = tf.sin(angle)
    cos_val = tf.cos(angle)
    return tf.tensor_scatter_nd_update(
        x,
        indices=tf.constant([[POSE_THETA_SIN_IDX], [POSE_THETA_COS_IDX]], dtype=tf.int32),
        updates=tf.stack([sin_val, cos_val], axis=0),
    )


def _huber(z: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """
    Elementwise Huber penalty for robustness to occasional large mismatches.
    Smooth near 0 (quadratic), linear for |z|>delta. Returns vector-valued penalties.
    """
    z = tf.cast(z, tf.float32)
    delta = tf.cast(delta, tf.float32)
    a = tf.abs(z)
    return tf.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


# ======================================================================================
# Dataset-aware scaling (hard-coded default)
# ======================================================================================

# State layout (next_step_output):
# 0: ωz [rad/s], 1: vx [m/s], 2: vy [m/s], 3: ψ [rad],
# 4: sinψ, 5: cosψ, 6: x [m], 7: y [m], 8: β [rad], 9: δ [rad]
STATE_DIM = 10
KEEP_IDX = tf.constant(
    [d for d in range(STATE_DIM) if d not in {SLIP_ANGLE_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX}],
    dtype=tf.int32
)

# Hard-coded dataset stds extracted from your stats table (std row).
# Order: [ωz, vx, vy, ψ, sinψ, cosψ, x, y, β, δ]
HARD_CODED_STATE_STD = np.array([
    1.29531,  # angular_vel_z [rad/s]
    1.73801,  # linear_vel_x [m/s]
    0.18586,  # linear_vel_y [m/s]
    1.91481,  # pose_theta [rad]
    1.0,      # sinψ (unused in residuals)
    1.0,      # cosψ (unused in residuals)
    8.07322,  # pose_x [m]
    3.53573,  # pose_y [m]
    0.04242,  # slip_angle [rad]
    0.23988,  # steering_angle [rad]
], dtype=np.float32)

# Conservative floors to avoid division by near-zero when dataset std is tiny
_SCALE_FLOORS = np.array([
    0.10,   # ωz
    0.50,   # vx
    0.20,   # vy
    0.10,   # ψ
    1.00,   # sinψ (unused in residuals)
    1.00,   # cosψ (unused in residuals)
    0.50,   # x
    0.50,   # y
    0.05,   # β
    0.05,   # δ
], dtype=np.float32)


def _build_state_scale_from_std(std_map: Dict[str, float]) -> np.ndarray:
    """
    Optional override: build per-channel scaling from a dict of dataset std values.
    Expected keys in std_map: 'angular_vel_z','linear_vel_x','linear_vel_y',
                              'pose_theta','pose_x','pose_y','slip_angle','steering_angle'.
    Missing keys fall back to floors.
    """
    scale = np.array([
        float(std_map.get('angular_vel_z', _SCALE_FLOORS[0])),
        float(std_map.get('linear_vel_x', _SCALE_FLOORS[1])),
        float(std_map.get('linear_vel_y', _SCALE_FLOORS[2])),
        float(std_map.get('pose_theta',    _SCALE_FLOORS[3])),
        1.0,  # sinψ (dropped)
        1.0,  # cosψ (dropped)
        float(std_map.get('pose_x',        _SCALE_FLOORS[6])),
        float(std_map.get('pose_y',        _SCALE_FLOORS[7])),
        float(std_map.get('slip_angle',    _SCALE_FLOORS[8])),
        float(std_map.get('steering_angle',_SCALE_FLOORS[9])),
    ], dtype=np.float32)
    return np.maximum(scale, _SCALE_FLOORS)


# ======================================================================================
# Residual builders that are scale-aware
# ======================================================================================

def _residual_vector_step(x_pred: tf.Tensor,
                          x_target: tf.Tensor,
                          keep_idx: tf.Tensor,
                          state_scale: tf.Tensor) -> tf.Tensor:
    """
    Residual φ(x_pred, x_target) used by the *per-step* LM solver:
      - wrap heading,
      - DROP slip & sin/cos residuals (redundant once heading is used),
      - pre-whiten by 1/state_scale for the kept channels.
    Returns: vector r of shape [len(keep_idx)]
    """
    # raw difference
    dif = x_pred - x_target
    # replace heading component by wrapped difference
    th_diff = _angle_wrap_diff(x_pred[POSE_THETA_IDX], x_target[POSE_THETA_IDX])
    dif = tf.tensor_scatter_nd_update(
        dif, indices=tf.constant([[POSE_THETA_IDX]], dtype=tf.int32), updates=tf.reshape(th_diff, [1])
    )
    # keep only selected indices
    r = tf.gather(dif, keep_idx)
    weights = 1.0 / tf.gather(state_scale, keep_idx)
    return r * weights


def _state_diff_wrapped_drop_sincos(xA: tf.Tensor, xB: tf.Tensor) -> tf.Tensor:
    """
    Full-length diff vector (len=10) with heading wrapped and sin/cos zeroed (dropped).
    """
    xA = tf.cast(xA, tf.float32); xB = tf.cast(xB, tf.float32)
    dif = xA - xB
    th_diff = _angle_wrap_diff(xA[POSE_THETA_IDX], xB[POSE_THETA_IDX])
    dif = tf.tensor_scatter_nd_update(
        dif, indices=tf.constant([[POSE_THETA_IDX]], dtype=tf.int32), updates=tf.reshape(th_diff, [1])
    )
    zeros2 = tf.zeros([2], dtype=tf.float32)
    dif = tf.tensor_scatter_nd_update(
        dif, indices=tf.constant([[POSE_THETA_SIN_IDX],[POSE_THETA_COS_IDX]], dtype=tf.int32), updates=zeros2
    )
    return dif


def _partial_state_huber(xA: tf.Tensor,
                         xB: tf.Tensor,
                         delta: float,
                         state_scale: tf.Tensor,
                         slip_lambda: float = 0.0) -> tf.Tensor:
    """
    Robust whole-horizon residual:
      - heading wrapped
      - sin/cos channels dropped
      - Huber on *scaled* residuals (divide by state_scale)
      - optional tiny prior on slip to suppress drift
    Returns scalar cost.
    """
    delta = tf.convert_to_tensor(delta, tf.float32)
    dif_full = _state_diff_wrapped_drop_sincos(_fix_sin_cos(xA), _fix_sin_cos(xB))
    z = dif_full / state_scale
    cost = tf.reduce_sum(_huber(z, delta))
    if slip_lambda > 0.0:
        slip_diff = (xA[SLIP_ANGLE_IDX] - xB[SLIP_ANGLE_IDX]) / state_scale[SLIP_ANGLE_IDX]
        cost += tf.convert_to_tensor(slip_lambda, tf.float32) * _huber(slip_diff, delta)
    return cost


# ======================================================================================
# Fast sequential LM: robust, O(T), no horizon-wide backprop.
# ======================================================================================

class CarInverseDynamicsFast:
    """
    Backward one-step inversion with Levenberg–Marquardt + backtracking.
    Solves for k = T-1..0 (newest→older):
        x_{k} (older) = argmin_x ||φ(f(x, u_k), x_{k+1})||^2
    Where φ wraps heading and drops [slip, sin, cos] in the residual.
    Uses residual pre-whitening and a trust-region measured in **scaled** units.

    controls_are_pid:
        True  → your controls are *pre-servo* (“desired angle, translational control”).
                 The forward map f includes servo+constraints (car_model.step_dynamics).
        False → your controls are already (delta_dot, v_x_dot).
                 The forward map is the *core* dynamics (car_model.step_dynamics_core).

    controls_time_order:
        'old_to_new' (default): Q[0] is oldest, Q[T-1] is newest. We'll invert using Q in reverse.
        'new_to_old': Q[0] is newest. We'll use it as-is.
    """

    def __init__(self, mu: Optional[float]=None, controls_are_pid: bool=True, dt: Optional[float]=None,
                 controls_time_order: str='old_to_new', state_scale_std_map: Optional[Dict[str,float]]=None):
        self.computation_lib = TensorFlowLibrary()
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size=1,
            car_parameter_file=Settings.CONTROLLER_CAR_PARAMETER_FILE,
            dt=(float(dt) if dt is not None else 0.01),
            intermediate_steps=1,
            computation_lib=self.computation_lib,
        )
        if mu is not None:
            self.car_model.change_friction_coefficient(mu)

        self._f = self.car_model.step_dynamics if controls_are_pid else self.car_model.step_dynamics_core
        self.state_dim   = STATE_DIM
        self.control_dim = 2

        # Scaling (dataset-aware, default = hard-coded constants)
        if state_scale_std_map is None:
            scale_np = HARD_CODED_STATE_STD.copy()
        else:
            scale_np = _build_state_scale_from_std(state_scale_std_map)
        self.state_scale = tf.Variable(scale_np, dtype=tf.float32, trainable=False, name='state_scale')

        # Config
        self.controls_time_order = controls_time_order

        # LM hyperparameters (tuned to scaled units; robust defaults)
        self.max_iters_per_step = 16
        self.init_damping       = tf.constant(1e-2,  dtype=tf.float32)
        self.min_damping        = tf.constant(1e-6,  dtype=tf.float32)
        self.max_damping        = tf.constant(1e+6,  dtype=tf.float32)
        self.backtrack_steps    = 6
        self.trust_clip         = tf.constant(0.5,   dtype=tf.float32)  # in scaled-norm
        self.stop_tol_cost      = tf.constant(1e-6,  dtype=tf.float32)
        self.stop_tol_step      = tf.constant(1e-5,  dtype=tf.float32)
        self._I = tf.eye(self.state_dim, dtype=tf.float32)

    # ----- API -----

    def change_friction_coefficient(self, mu: float):
        self.car_model.change_friction_coefficient(mu)

    def set_state_scale_from_std_map(self, std_map: Dict[str,float]):
        new_scale = _build_state_scale_from_std(std_map)
        self.state_scale.assign(new_scale)

    def inverse_entire_trajectory(self,
                                  x_T: np.ndarray,
                                  Q: np.ndarray,
                                  x_init_sequence: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Invert a whole control sequence.
        Args:
            x_T:  [1,10] most recent state (pinned)
            Q:    [T,2] controls; default 'old_to_new' time order (we reverse internally)
            x_init_sequence: optional [T,10] warm start for the *older* states

        Returns:
            states: [T+1,10] with states[0]==x_T (newest), states[T]==x_0 (oldest)
            converged: [T] bool flags for each inverted step
        """
        x_T = np.asarray(x_T, dtype=np.float32)
        Q = np.asarray(Q, dtype=np.float32)
        if x_T.shape != (1, self.state_dim):
            raise ValueError(f"x_T must be [1,{self.state_dim}]. Got {x_T.shape}")
        if Q.ndim != 2 or Q.shape[1] != self.control_dim:
            raise ValueError(f"Q must be [T,{self.control_dim}]. Got {Q.shape}")

        if self.controls_time_order == 'old_to_new':
            Q_use = Q[::-1].copy()    # newest first
        elif self.controls_time_order == 'new_to_old':
            Q_use = Q
        else:
            raise ValueError("controls_time_order must be 'old_to_new' or 'new_to_old'")

        T = Q_use.shape[0]
        states = tf.TensorArray(tf.float32, size=T+1, clear_after_read=False)
        conv   = tf.TensorArray(tf.bool,    size=T,   clear_after_read=False)

        xk = _fix_sin_cos(tf.convert_to_tensor(x_T[0], dtype=tf.float32))  # pinned newest
        states = states.write(0, xk)

        x_inits = None
        if x_init_sequence is not None:
            x_init_sequence = np.asarray(x_init_sequence, dtype=np.float32)
            if x_init_sequence.shape != (T, self.state_dim):
                raise ValueError(f"x_init_sequence must be [T,{self.state_dim}]")
            x_inits = tf.convert_to_tensor(x_init_sequence, dtype=tf.float32)

        for k in range(T):
            uk = tf.convert_to_tensor(Q_use[k], dtype=tf.float32)
            x0 = x_inits[k] if x_inits is not None else xk
            x_prev, ok = self._invert_one_step_LM(xk, uk, x0)
            states = states.write(k+1, x_prev)
            conv   = conv.write(k, ok)
            xk = x_prev

        return states.stack().numpy(), conv.stack().numpy()

    # ----- core LM (no early return; TF-while with done flag) -----

    @tf.function(experimental_relax_shapes=True)
    def _invert_one_step_LM(self, x_target: tf.Tensor, u: tf.Tensor, x_start: tf.Tensor):
        """
        Core LM step with scale-aware residuals and trust-region measured in scaled units.
        Implemented with tf.while_loop to avoid 'return inside TF loop' issues.
        """
        x0 = _fix_sin_cos(tf.identity(x_start))
        lam0 = tf.identity(self.init_damping)
        best_x0 = x0
        best_cost0 = tf.constant(np.inf, dtype=tf.float32)
        i0 = tf.constant(0, tf.int32)
        done0 = tf.constant(False, tf.bool)

        def cond(i, x, lam, best_x, best_cost, done):
            return tf.logical_and(tf.less(i, tf.convert_to_tensor(self.max_iters_per_step, tf.int32)),
                                  tf.logical_not(done))

        def body(i, x, lam, best_x, best_cost, done):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                x_pred = self._f(x[tf.newaxis, :], u[tf.newaxis, :])[0]
                x_pred = _fix_sin_cos(x_pred)
                r = _residual_vector_step(x_pred, x_target, KEEP_IDX, self.state_scale)
                cost = tf.reduce_sum(r * r)

            # Defaults for this iteration
            x_candidate = x
            new_cost = cost
            applied_alpha = tf.constant(0.0, tf.float32)

            small_enough = cost < self.stop_tol_cost

            def after_small():
                # Mark done; keep x, record as best
                return (i + 1, x, lam, x, cost, tf.constant(True, tf.bool))

            def do_rest():
                # Build normal equations
                J = tape.jacobian(r, x)  # [m,10]
                JT = tf.transpose(J)
                H = tf.matmul(JT, J)  # [10,10]
                g = tf.matmul(JT, r[:, tf.newaxis])[:, 0]  # [10]

                # Symmetrize & dampen
                I = tf.eye(self.state_dim, dtype=tf.float32)
                A = 0.5 * (H + tf.transpose(H)) + lam * I + 1e-9 * I
                delta = tf.linalg.solve(A, -g[:, tf.newaxis])[:, 0]  # [10]

                # Trust-region clip in scaled space
                scaled_step_norm = tf.norm(delta / self.state_scale)
                delta = tf.cond(
                    scaled_step_norm > self.trust_clip,
                    lambda: delta * (self.trust_clip / (scaled_step_norm + 1e-9)),
                    lambda: delta
                )

                # Backtracking — evaluate all alphas; keep best improvement
                def bt_cond(t, x_c, nc, aalpha):
                    return tf.less(t, tf.convert_to_tensor(self.backtrack_steps, tf.int32))

                def bt_body(t, x_c, nc, aalpha):
                    alpha = tf.pow(0.5, tf.cast(t, tf.float32))
                    xc = _fix_sin_cos(x + alpha * delta)
                    xp = self._f(xc[tf.newaxis, :], u[tf.newaxis, :])[0]
                    xp = _fix_sin_cos(xp)
                    rc = _residual_vector_step(xp, x_target, KEEP_IDX, self.state_scale)
                    c = tf.reduce_sum(rc * rc)
                    improved = c < nc
                    x_c = tf.where(improved, xc, x_c)
                    nc = tf.where(improved, c, nc)
                    aalpha = tf.where(improved, alpha, aalpha)
                    return (t + 1, x_c, nc, aalpha)

                t0 = tf.constant(0, tf.int32)
                _, x_cand, ncost, aalpha = tf.while_loop(
                    bt_cond, bt_body, loop_vars=[t0, x_candidate, new_cost, applied_alpha],
                    maximum_iterations=self.backtrack_steps
                )

                accepted = ncost < cost

                x_new = tf.where(accepted, x_cand, x)
                best_x_new = tf.where(tf.logical_and(accepted, ncost < best_cost), x_cand, best_x)
                best_cost_new = tf.where(tf.logical_and(accepted, ncost < best_cost), ncost, best_cost)
                lam_new = tf.where(accepted,
                                   tf.maximum(self.min_damping, lam * 0.3),
                                   tf.minimum(self.max_damping, lam * 10.0))

                step_small = tf.logical_and(
                    accepted, tf.norm((aalpha * delta) / self.state_scale) < self.stop_tol_step
                )
                done_new = tf.logical_or(done, step_small)

                return (i + 1, x_new, lam_new, best_x_new, best_cost_new, done_new)

            return tf.cond(small_enough, after_small, do_rest)

        iN, xN, lamN, best_xN, best_costN, doneN = tf.while_loop(
            cond, body, loop_vars=[i0, x0, lam0, best_x0, best_cost0, done0],
            maximum_iterations=self.max_iters_per_step
        )

        ok = best_costN < self.stop_tol_cost
        return _fix_sin_cos(best_xN), ok

# ======================================================================================
# Whole-horizon refinement (short, robust, uses Huber); compiled inner loop.
# ======================================================================================

class CarInverseDynamics:
    """
    Short “bundle-adjustment” style refinement over the whole (sub)horizon.
    Uses Huber loss in both dynamics residual and temporal regularization.
    Tolerances are normalized to Huber delta and residual length → scale-invariant flags.
    """

    def __init__(self, mu: Optional[float]=None, controls_are_pid: bool=True, dt: Optional[float]=None,
                 state_scale_std_map: Optional[Dict[str,float]]=None):
        self.computation_lib = TensorFlowLibrary()
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size=1,
            car_parameter_file=Settings.CONTROLLER_CAR_PARAMETER_FILE,
            dt=(float(dt) if dt is not None else 0.01),
            intermediate_steps=1,
            computation_lib=self.computation_lib,
        )
        if mu is not None:
            self.car_model.change_friction_coefficient(mu)

        self._f = self.car_model.step_dynamics if controls_are_pid else self.car_model.step_dynamics_core

        self.state_dim   = STATE_DIM
        self.control_dim = 2
        self.MAX_T       = 5000

        # Scaling (dataset-aware, default = hard-coded constants)
        if state_scale_std_map is None:
            scale_np = HARD_CODED_STATE_STD.copy()
        else:
            scale_np = _build_state_scale_from_std(state_scale_std_map)
        self.state_scale = tf.Variable(scale_np, dtype=tf.float32, trainable=False, name='state_scale')

        # Persistent buffers
        self.X_var      = tf.Variable(tf.zeros([self.MAX_T, self.state_dim], dtype=tf.float32), trainable=True,  name='X_var')
        self.Q_var      = tf.Variable(tf.zeros([self.MAX_T, self.control_dim], dtype=tf.float32), trainable=False, name='Q_var')
        self.x_next_var = tf.Variable(tf.zeros([self.state_dim], dtype=tf.float32),            trainable=False, name='x_next_var')
        self.T_var      = tf.Variable(0, dtype=tf.int32, trainable=False, name='T_var')

        # Optimizer (legacy path to be TF 2.x compatible)
        try:
            opt = tf.keras.optimizers.legacy.Adam
        except AttributeError:
            opt = tf.keras.optimizers.Adam
        self.base_optimizer = opt(learning_rate=1e-3)

        # Robustness/regularization (scale-aware via self.state_scale in cost).
        self.REG_TEMPORAL   = 1e-3        # temporal smoothness weight
        self.RES_DELTA      = 1.0         # Huber delta (dynamics) in scaled units
        self.TEMP_DELTA     = 0.5         # Huber delta (temporal)
        self.SLIP_PRIOR     = 1e-4        # tiny anchor on slip

        kept = [d for d in range(self.state_dim)
                if d not in {SLIP_ANGLE_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX}]
        self._M_PARTIAL = float(len(kept))
        self._TOL_MULT  = 3.0
        self.TOL_HUBER  = tf.constant(self._TOL_MULT * (self.RES_DELTA ** 2) * self._M_PARTIAL, tf.float32)

        self.PHASE1_STEPS = 8
        self.PHASE2_STEPS = 8
        self.LR1          = 5e-3
        self.LR2          = 1e-3

    # ----- API -----

    def change_friction_coefficient(self, mu: float):
        self.car_model.change_friction_coefficient(mu)

    def set_state_scale_from_std_map(self, std_map: Dict[str,float]):
        new_scale = _build_state_scale_from_std(std_map)
        self.state_scale.assign(new_scale)

    def inverse_entire_trajectory(self, x_T: np.ndarray, Q: np.ndarray,
                                  X_init: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        x_T = np.asarray(x_T, dtype=np.float32)
        Q = np.asarray(Q, dtype=np.float32)

        if x_T.shape != (1, self.state_dim):
            raise ValueError(f"x_T must be [1,{self.state_dim}]")
        if Q.ndim != 2 or Q.shape[1] != self.control_dim:
            raise ValueError(f"Q must be [T,{self.control_dim}]")

        T = Q.shape[0]
        if T > self.MAX_T:
            raise ValueError(f"T={T} exceeds MAX_T={self.MAX_T}")

        self.x_next_var.assign(x_T[0])
        self.Q_var[:T].assign(Q[::-1])     # newest first for our compiled kernel
        self.T_var.assign(T)

        if X_init is None:
            self.X_var[:T].assign(tf.repeat(self.x_next_var[tf.newaxis, :], repeats=T, axis=0))
        else:
            X_init = np.asarray(X_init, dtype=np.float32)
            if X_init.shape != (T, self.state_dim):
                raise ValueError(f"X_init must be [T,{self.state_dim}]")
            self.X_var[:T].assign(X_init)

        states_tf, conv_tf = self._run_inverse_trajectory_compiled()
        return states_tf.numpy(), conv_tf.numpy()

    # ----- compiled inner loop -----

    @tf.function(experimental_relax_shapes=True)
    def _run_inverse_trajectory_compiled(self) -> Tuple[tf.Tensor, tf.Tensor]:
        T = self.T_var

        def cost_fn():
            cost_val = tf.constant(0.0, dtype=tf.float32)

            # Dynamics consistency with robust Huber; sin/cos projected each use.
            for i in tf.range(T):
                x_i  = _fix_sin_cos(self.x_next_var) if tf.equal(i, 0) else _fix_sin_cos(self.X_var[i-1, :])
                xim1 = _fix_sin_cos(self.X_var[i, :])
                qi   = self.Q_var[i, :]

                x_pred = self._f(xim1[tf.newaxis, :], qi[tf.newaxis, :])[0]
                x_pred = _fix_sin_cos(x_pred)
                cost_val += _partial_state_huber(x_pred, x_i, delta=self.RES_DELTA,
                                                 state_scale=self.state_scale, slip_lambda=self.SLIP_PRIOR)

            # Temporal regularization (robust), mild weight.
            for i in tf.range(1, T):
                x_cur  = _fix_sin_cos(self.X_var[i, :])
                x_prev = _fix_sin_cos(self.X_var[i-1, :])
                cost_val += self.REG_TEMPORAL * _partial_state_huber(
                    x_cur, x_prev, delta=self.TEMP_DELTA, state_scale=self.state_scale, slip_lambda=self.SLIP_PRIOR
                )

            return cost_val

        # Two short phases with decaying LR.
        try:
            self.base_optimizer.learning_rate.assign(self.LR1)
        except Exception:
            pass
        for _ in tf.range(self.PHASE1_STEPS):
            with tf.GradientTape() as tape:
                c = cost_fn()
            grads = tape.gradient(c, [self.X_var])
            self.base_optimizer.apply_gradients(zip(grads, [self.X_var]))

        try:
            self.base_optimizer.learning_rate.assign(self.LR2)
        except Exception:
            pass
        for _ in tf.range(self.PHASE2_STEPS):
            with tf.GradientTape() as tape:
                c = cost_fn()
            grads = tape.gradient(c, [self.X_var])
            self.base_optimizer.apply_gradients(zip(grads, [self.X_var]))

        # Build output states and per-step consistency flags.
        states_ta = tf.TensorArray(dtype=tf.float32, size=T+1)
        states_ta = states_ta.write(0, _fix_sin_cos(self.x_next_var))
        for i in tf.range(T):
            x_i = _fix_sin_cos(self.X_var[i, :])
            self.X_var[i, :].assign(x_i)
            states_ta = states_ta.write(i+1, x_i)

        conv_ta = tf.TensorArray(dtype=tf.bool, size=T)
        for i in tf.range(T):
            x_i   = states_ta.read(0) if tf.equal(i, 0) else states_ta.read(i)
            x_im1 = states_ta.read(i+1)
            qi    = self.Q_var[i, :]
            x_pred = self._f(x_im1[tf.newaxis, :], qi[tf.newaxis, :])[0]
            x_pred = _fix_sin_cos(x_pred)
            err_scalar = _partial_state_huber(x_pred, x_i, delta=self.RES_DELTA,
                                              state_scale=self.state_scale, slip_lambda=self.SLIP_PRIOR)
            conv_ta = conv_ta.write(i, err_scalar < self.TOL_HUBER)

        return states_ta.stack(), conv_ta.stack()


# ======================================================================================
# Hybrid orchestrator: warm start (LM) + windowed whole-horizon refinement.
# ======================================================================================

class CarInverseDynamicsHybrid:
    """
    1) Fast per-step LM warm start (multiple shooting) → robust local consistency.
    2) Windowed short bundle-adjustment refinements with Huber loss over each window.
       Overlaps are **linearly blended** to avoid seams and reduce compounding error.
    """

    def __init__(self, mu: Optional[float]=None, window_size: int=200, overlap: int=50,
                 controls_are_pid: bool=True, dt: Optional[float]=None,
                 controls_time_order: str='old_to_new',
                 state_scale_std_map: Optional[Dict[str,float]]=None):
        if overlap < 0 or window_size <= 0 or overlap >= window_size:
            raise ValueError("Require window_size>0 and 0 <= overlap < window_size.")

        self.fast   = CarInverseDynamicsFast(mu=mu, controls_are_pid=controls_are_pid, dt=dt,
                                             controls_time_order=controls_time_order,
                                             state_scale_std_map=state_scale_std_map)
        self.refine = CarInverseDynamics(mu=mu, controls_are_pid=controls_are_pid, dt=dt,
                                         state_scale_std_map=state_scale_std_map)
        self.window_size = int(window_size)
        self.overlap     = int(overlap)

    # ----- API -----

    def change_friction_coefficient(self, mu: float):
        self.fast.change_friction_coefficient(mu)
        self.refine.change_friction_coefficient(mu)

    def set_state_scale_from_std_map(self, std_map: Dict[str,float]):
        self.fast.set_state_scale_from_std_map(std_map)
        self.refine.set_state_scale_from_std_map(std_map)

    def inverse_entire_trajectory(self, x_T: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1) Warm start with per-step LM (O(T), robust).
        states0, _ = self.fast.inverse_entire_trajectory(x_T, Q)   # [T+1,10], newest first
        refined = tf.Variable(tf.convert_to_tensor(states0, dtype=tf.float32))  # mutable [T+1,10]

        # 2) Windowed refinement (short, robust) with overlap blending.
        T = int(Q.shape[0])
        W = self.window_size
        O = self.overlap

        start = 0
        while start < T:
            end = min(start + W, T)
            Tw  = end - start  # steps in this window

            # Prepare inputs for this window:
            x_pin = refined[start]                                         # [10]
            Q_w   = tf.convert_to_tensor(Q[::-1][start:end], tf.float32)   # newest first
            X_w0  = refined[start+1:start+1+Tw]                            # [Tw,10]

            # Seed refinement buffers and run short global pass.
            self.refine.x_next_var.assign(x_pin)
            self.refine.Q_var[:Tw].assign(Q_w)
            self.refine.T_var.assign(Tw)
            self.refine.X_var[:Tw].assign(X_w0)

            states_w_tf, _ = self.refine._run_inverse_trajectory_compiled()  # [Tw+1,10]; states_w[0]==x_pin

            # Write back with overlap blending.
            w_start_idx_full = start + 1
            w_end_idx_full   = start + Tw   # inclusive

            if start > 0 and O > 0:
                ov_start_full = start + 1
                ov_end_full   = min(start + O, w_end_idx_full)  # inclusive
                ov_len        = ov_end_full - ov_start_full + 1

                if ov_len > 0:
                    alpha = tf.linspace(0.0, 1.0, ov_len)[:, None]               # [L,1]
                    old   = refined[ov_start_full:ov_end_full+1]                 # [L,10]
                    neww  = states_w_tf[(ov_start_full - start):(ov_end_full - start + 1)]  # [L,10]
                    blend = (1.0 - alpha) * old + alpha * neww
                    refined[ov_start_full:ov_end_full+1].assign(blend)

                    tail_begin = ov_end_full + 1
                    if tail_begin <= w_end_idx_full:
                        refined[tail_begin:w_end_idx_full+1].assign(
                            states_w_tf[(tail_begin - start):(w_end_idx_full - start + 1)]
                        )
                else:
                    refined[w_start_idx_full:w_end_idx_full+1].assign(states_w_tf[1:Tw+1])
            else:
                refined[w_start_idx_full:w_end_idx_full+1].assign(states_w_tf[1:Tw+1])

            if end == T:
                break
            start = end - O

        # 3) Final consistency flags computed against the refined trajectory.
        conv = self._compute_consistency(refined, tf.convert_to_tensor(Q[::-1], tf.float32)).numpy()
        return refined.numpy(), conv

    @tf.function(experimental_relax_shapes=True)
    def _compute_consistency(self, refined_states: tf.Tensor, Q_newest_first: tf.Tensor) -> tf.Tensor:
        """
        Compute per-step dynamic-consistency flags across the whole horizon.
        Uses the same robust metric threshold as the refinement pass.
        Q_newest_first: controls ordered newest→older with shape [T,2].
        """
        T = tf.shape(Q_newest_first)[0]
        conv_ta = tf.TensorArray(dtype=tf.bool, size=T)

        for i in tf.range(T):
            x_i   = refined_states[i]     # this is x_{k+1}
            x_im1 = refined_states[i+1]   # this is x_{k}
            qi    = Q_newest_first[i]
            x_pred = self.refine._f(x_im1[tf.newaxis, :], qi[tf.newaxis, :])[0]
            x_pred = _fix_sin_cos(x_pred)
            err = _partial_state_huber(x_pred, _fix_sin_cos(x_i),
                                       delta=self.refine.RES_DELTA,
                                       state_scale=self.refine.state_scale,
                                       slip_lambda=self.refine.SLIP_PRIOR)
            conv_ta = conv_ta.write(i, err < self.refine.TOL_HUBER)

        return conv_ta.stack()


# ======================================================================================
# Run tests when executed as a script (no CLI parser; defaults live in tests/inverse_dynamics/config.py)
# ======================================================================================

if __name__ == "__main__":
    # Running this file triggers the test suite with hard-coded settings.
    try:
        import pytest
        print(">> Running inverse dynamics tests (hard-coded settings in tests/inverse_dynamics/config.py)...")
        raise SystemExit(pytest.main(["-q", "tests/inverse_dynamics/test_inverse_dynamics_eval.py"]))
    except Exception as e:
        print("Failed to run tests:", e)
