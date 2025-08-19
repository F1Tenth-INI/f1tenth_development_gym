# utilities/InverseDynamics.py
# Refine-only inverse dynamics with kinematic warm starts.
# Progressive overlapping windows (no LM), robust losses, and pinned newest state kept unchanged.

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# External deps (unchanged)
# ---------------------------------------------------------------------------
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit_ASF.car_model import car_model
from utilities.Settings import Settings
from utilities.state_utilities import (
    POSE_THETA_IDX,
    SLIP_ANGLE_IDX,
    POSE_THETA_SIN_IDX,
    POSE_THETA_COS_IDX,
)

# ---------------------------------------------------------------------------
# XLA toggle for the refine optimizer (safe default OFF on CPU)
# Set ID_USE_XLA=1 to enable on supported platforms (e.g., Linux+CUDA).
# ---------------------------------------------------------------------------
XLA_REFINE = bool(int(os.getenv("ID_USE_XLA", "0")))

# ======================================================================================
# Utilities
# ======================================================================================

def _angle_wrap_diff(thA: tf.Tensor, thB: tf.Tensor) -> tf.Tensor:
    thA = tf.cast(thA, tf.float32); thB = tf.cast(thB, tf.float32)
    return tf.atan2(tf.sin(thA - thB), tf.cos(thA - thB))

def _fix_sin_cos(x: tf.Tensor) -> tf.Tensor:
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
    z = tf.cast(z, tf.float32); delta = tf.cast(delta, tf.float32)
    a = tf.abs(z)
    return tf.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))

# ======================================================================================
# Scaling (hard-coded)
# ======================================================================================

STATE_DIM = 10
KEEP_IDX = tf.constant(
    [d for d in range(STATE_DIM) if d not in {SLIP_ANGLE_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX}],
    dtype=tf.int32
)

HARD_CODED_STATE_STD = np.array([
    1.29531,  # ωz
    1.73801,  # vx
    0.18586,  # vy
    1.91481,  # ψ
    1.0,      # sinψ (unused)
    1.0,      # cosψ (unused)
    8.07322,  # x
    3.53573,  # y
    0.04242,  # β
    0.23988,  # δ
], dtype=np.float32)

_SCALE_FLOORS = np.array([0.10, 0.50, 0.20, 0.10, 1.0, 1.0, 0.50, 0.50, 0.05, 0.05], dtype=np.float32)

def _build_state_scale_from_std(std_map: Dict[str, float]) -> np.ndarray:
    scale = np.array([
        float(std_map.get('angular_vel_z', _SCALE_FLOORS[0])),
        float(std_map.get('linear_vel_x', _SCALE_FLOORS[1])),
        float(std_map.get('linear_vel_y', _SCALE_FLOORS[2])),
        float(std_map.get('pose_theta',    _SCALE_FLOORS[3])),
        1.0,
        1.0,
        float(std_map.get('pose_x',        _SCALE_FLOORS[6])),
        float(std_map.get('pose_y',        _SCALE_FLOORS[7])),
        float(std_map.get('slip_angle',    _SCALE_FLOORS[8])),
        float(std_map.get('steering_angle',_SCALE_FLOORS[9])),
    ], dtype=np.float32)
    return np.maximum(scale, _SCALE_FLOORS)

# ======================================================================================
# Residuals
# ======================================================================================

def _residual_vector_step(x_pred: tf.Tensor, x_target: tf.Tensor,
                          keep_idx: tf.Tensor, state_scale: tf.Tensor) -> tf.Tensor:
    dif = x_pred - x_target
    th_diff = _angle_wrap_diff(x_pred[POSE_THETA_IDX], x_target[POSE_THETA_IDX])
    dif = tf.tensor_scatter_nd_update(
        dif, indices=tf.constant([[POSE_THETA_IDX]], dtype=tf.int32), updates=tf.reshape(th_diff, [1])
    )
    r = tf.gather(dif, keep_idx)
    weights = 1.0 / tf.gather(state_scale, keep_idx)
    return r * weights

def _state_diff_wrapped_drop_sincos(xA: tf.Tensor, xB: tf.Tensor) -> tf.Tensor:
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

def _partial_state_huber(xA: tf.Tensor, xB: tf.Tensor, delta: float,
                         state_scale: tf.Tensor, slip_lambda: float = 0.0) -> tf.Tensor:
    delta = tf.convert_to_tensor(delta, tf.float32)
    dif_full = _state_diff_wrapped_drop_sincos(_fix_sin_cos(xA), _fix_sin_cos(xB))
    z = dif_full / state_scale
    cost = tf.reduce_sum(_huber(z, delta))
    if slip_lambda > 0.0:
        slip_diff = (xA[SLIP_ANGLE_IDX] - xB[SLIP_ANGLE_IDX]) / state_scale[SLIP_ANGLE_IDX]
        cost += tf.convert_to_tensor(slip_lambda, tf.float32) * _huber(slip_diff, delta)
    return cost

# ======================================================================================
# Kinematic back-prop seed (cheap O(T) initializer)
# ======================================================================================

def build_kinematic_seed(x_T_vec: np.ndarray, Q_old_to_new: np.ndarray, dt: float) -> np.ndarray:
    """
    newest→older seed via simple backward kinematics
      ψ_{k-1} = ψ_k - dt*ωz_k
      v_x_{k-1} = v_x_k - dt*a_k
      x,y back-integrated in world frame, v_y, ωz, β, δ carried over.
    """
    T = int(Q_old_to_new.shape[0])
    out = np.zeros((T, STATE_DIM), dtype=np.float32)
    x_cur = x_T_vec.astype(np.float32).copy()
    for j in range(T):
        a   = float(Q_old_to_new[T - 1 - j, 1])
        wz  = float(x_cur[0]);  vx = float(x_cur[1]); vy = float(x_cur[2])
        psi = float(x_cur[3]);  xw = float(x_cur[6]); yw = float(x_cur[7])
        beta = float(x_cur[8]); delta = float(x_cur[9])

        psi_prev = psi - dt*wz
        vx_prev  = vx - dt*a
        vy_prev  = vy
        x_prev   = xw - dt*(vx*np.cos(psi) - vy*np.sin(psi))
        y_prev   = yw - dt*(vx*np.sin(psi) + vy*np.cos(psi))
        wz_prev  = wz
        delta_prev = delta
        beta_prev  = beta

        out[j] = np.array([
            wz_prev, vx_prev, vy_prev, psi_prev,
            np.sin(psi_prev), np.cos(psi_prev),
            x_prev, y_prev, beta_prev, delta_prev
        ], dtype=np.float32)
        x_cur  = out[j]
    return out  # newest→older

# ======================================================================================
# Refine-only solver
# ======================================================================================

class TrajectoryRefiner:
    """
    Whole-horizon refine-only inverse dynamics (Huber + temporal smoothness).
    Returns newest→older states, with the pinned newest state returned exactly as provided.
    """

    def __init__(self, mu: Optional[float]=None, controls_are_pid: bool=True, dt: Optional[float]=None,
                 state_scale_std_map: Optional[Dict[str,float]]=None):
        self.dt = float(dt) if dt is not None else 0.01
        self.computation_lib = TensorFlowLibrary()
        self.car_model = car_model(
            model_of_car_dynamics=Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size=1,
            car_parameter_file=Settings.CONTROLLER_CAR_PARAMETER_FILE,
            dt=self.dt,
            intermediate_steps=1,
            computation_lib=self.computation_lib,
        )
        if mu is not None:
            self.car_model.change_friction_coefficient(mu)

        self._f = self.car_model.step_dynamics if controls_are_pid else self.car_model.step_dynamics_core

        self.state_dim   = STATE_DIM
        self.control_dim = 2
        self.MAX_T       = 5000

        # Scaling
        if state_scale_std_map is None:
            scale_np = HARD_CODED_STATE_STD.copy()
        else:
            scale_np = _build_state_scale_from_std(state_scale_std_map)
        self.state_scale = tf.Variable(scale_np, dtype=tf.float32, trainable=False)

        # Persistent buffers
        self.X_var      = tf.Variable(tf.zeros([self.MAX_T, self.state_dim], tf.float32), trainable=True)
        self.Q_var      = tf.Variable(tf.zeros([self.MAX_T, self.control_dim], tf.float32), trainable=False)
        self.x_next_var = tf.Variable(tf.zeros([self.state_dim], tf.float32), trainable=False)
        self.T_var      = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Optimizer
        try:
            opt = tf.keras.optimizers.legacy.Adam
        except AttributeError:
            opt = tf.keras.optimizers.Adam
        self.base_optimizer = opt(learning_rate=1e-3)

        # Loss settings
        self.REG_TEMPORAL   = 1e-3
        self.RES_DELTA      = 1.0
        self.TEMP_DELTA     = 0.5
        self.SLIP_PRIOR     = 1e-4

        kept = [d for d in range(self.state_dim) if d not in {SLIP_ANGLE_IDX, POSE_THETA_SIN_IDX, POSE_THETA_COS_IDX}]
        self._M_PARTIAL = float(len(kept))
        self._TOL_MULT  = 3.0
        self.TOL_HUBER  = tf.constant(self._TOL_MULT * (self.RES_DELTA ** 2) * self._M_PARTIAL, tf.float32)

        # Schedule
        self.PHASE1_STEPS = 4
        self.PHASE2_STEPS = 4
        self.LR1          = 5e-3
        self.LR2          = 1e-3

    def set_schedule_for_T(self, T: int):
        if T <= 30:
            self.PHASE1_STEPS, self.PHASE2_STEPS = 4, 4
        elif T <= 120:
            self.PHASE1_STEPS, self.PHASE2_STEPS = 3, 3
        else:
            self.PHASE1_STEPS, self.PHASE2_STEPS = 2, 2

    def inverse_entire_trajectory(self, x_T: np.ndarray, Q: np.ndarray,
                                  X_init: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        x_T = np.asarray(x_T, dtype=np.float32); Q = np.asarray(Q, dtype=np.float32)
        if x_T.shape != (1, self.state_dim):
            raise ValueError(f"x_T must be [1,{self.state_dim}]")
        if Q.ndim != 2 or Q.shape[1] != self.control_dim:
            raise ValueError(f"Q must be [T,{self.control_dim}]")

        T = Q.shape[0]
        if T > self.MAX_T:
            raise ValueError(f"T={T} exceeds MAX_T={self.MAX_T}")

        self.set_schedule_for_T(T)

        if X_init is None:
            X_init = build_kinematic_seed(x_T[0], Q, self.dt)

        # Load buffers
        self.x_next_var.assign(x_T[0])      # pinned as given
        self.Q_var[:T].assign(Q[::-1])      # newest→older for internal loop
        self.T_var.assign(T)
        self.X_var[:T].assign(X_init)       # newest→older seed

        # Optimize (compiled; no outputs)
        self._optimize_compiled()

        # Assemble outputs (eager)
        X_np = self.X_var[:T].numpy()
        for i in range(T):
            X_np[i] = _fix_sin_cos(tf.convert_to_tensor(X_np[i], tf.float32)).numpy()
        states = np.vstack([x_T[0], X_np])  # newest→older

        # Consistency flags (eager)
        Q_newest_first = self.Q_var[:T].numpy()
        flags = np.zeros((T,), dtype=bool)
        for i in range(T):
            x_i   = states[0] if i == 0 else states[i]
            x_im1 = states[i+1]
            qi    = Q_newest_first[i]
            x_pred = self._f(
                tf.convert_to_tensor(x_im1, tf.float32)[tf.newaxis, :],
                tf.convert_to_tensor(qi, tf.float32)[tf.newaxis, :]
            )[0]
            x_pred = _fix_sin_cos(x_pred)
            err = _partial_state_huber(x_pred, tf.convert_to_tensor(x_i, tf.float32),
                                       delta=self.RES_DELTA,
                                       state_scale=self.state_scale,
                                       slip_lambda=self.SLIP_PRIOR)
            flags[i] = bool(err.numpy() < float(self.TOL_HUBER.numpy()))

        return states, flags

    @tf.function(experimental_relax_shapes=True, jit_compile=XLA_REFINE)
    def _optimize_compiled(self) -> None:
        T = self.T_var

        def cost():
            c = tf.constant(0.0, tf.float32)
            for i in tf.range(T):
                x_i  = self.x_next_var if tf.equal(i, 0) else self.X_var[i-1, :]
                x_i  = _fix_sin_cos(x_i)
                xim1 = _fix_sin_cos(self.X_var[i, :])
                qi   = self.Q_var[i, :]
                x_pred = self._f(xim1[tf.newaxis, :], qi[tf.newaxis, :])[0]
                x_pred = _fix_sin_cos(x_pred)
                c += _partial_state_huber(x_pred, x_i, delta=self.RES_DELTA,
                                          state_scale=self.state_scale, slip_lambda=self.SLIP_PRIOR)
            for i in tf.range(1, T):
                x_cur  = _fix_sin_cos(self.X_var[i, :])
                x_prev = _fix_sin_cos(self.X_var[i-1, :])
                c += self.REG_TEMPORAL * _partial_state_huber(
                    x_cur, x_prev, delta=self.TEMP_DELTA, state_scale=self.state_scale, slip_lambda=self.SLIP_PRIOR
                )
            return c

        try:
            self.base_optimizer.learning_rate.assign(self.LR1)
        except Exception:
            pass
        for _ in tf.range(self.PHASE1_STEPS):
            with tf.GradientTape() as tape:
                c = cost()
            g = tape.gradient(c, [self.X_var])
            self.base_optimizer.apply_gradients(zip(g, [self.X_var]))

        try:
            self.base_optimizer.learning_rate.assign(self.LR2)
        except Exception:
            pass
        for _ in tf.range(self.PHASE2_STEPS):
            with tf.GradientTape() as tape:
                c = cost()
            g = tape.gradient(c, [self.X_var])
            self.base_optimizer.apply_gradients(zip(g, [self.X_var]))

# ======================================================================================
# Progressive overlapping windows (refine-only)
# ======================================================================================

class ProgressiveWindowRefiner:
    """
    Refine-only, progressive-grow with overlap (no LM).
    - Build full newest→older kinematic seed once.
    - For k_end = W, W+(W-O), ...: refine the prefix pinned at the measured newest state.
    """

    def __init__(self, mu: Optional[float]=None, controls_are_pid: bool=True, dt: Optional[float]=None,
                 window_size: int = 30, overlap: int = 10,
                 state_scale_std_map: Optional[Dict[str,float]]=None):
        if overlap < 0 or window_size <= 0 or overlap >= window_size:
            raise ValueError("window_size>0 and 0 <= overlap < window_size required.")
        self.refiner = TrajectoryRefiner(mu=mu, controls_are_pid=controls_are_pid, dt=dt,
                                         state_scale_std_map=state_scale_std_map)
        # Blend windows a bit more strongly
        self.refiner.REG_TEMPORAL = 1e-2
        self.window_size = int(window_size)
        self.overlap     = int(overlap)
        self.dt = self.refiner.dt

    def refine(self, x_T: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_T = np.asarray(x_T, np.float32); Q = np.asarray(Q, np.float32)
        T = int(Q.shape[0]); W = self.window_size; O = self.overlap

        # Full kinematic seed
        seed_full = build_kinematic_seed(x_T[0], Q, self.dt)  # [T,10] newest→older
        refined = np.vstack([x_T[0], seed_full]).astype(np.float32)  # [T+1,10]

        k_end = min(W, T)
        while True:
            Q_seg  = Q[T - k_end : T]            # old→new
            X_init = refined[1 : k_end + 1]      # newest→older current prefix
            states_w, _ = self.refiner.inverse_entire_trajectory(x_T=x_T, Q=Q_seg, X_init=X_init)
            refined[:k_end+1] = states_w
            if k_end == T:
                break
            k_end = min(k_end + (W - O), T)

        # Consistency flags (eager)
        flags = np.zeros((T,), dtype=bool)
        Q_newest_first = Q[::-1]
        for i in range(T):
            x_i   = refined[0] if i == 0 else refined[i]
            x_im1 = refined[i+1]
            qi    = Q_newest_first[i]
            x_pred = self.refiner._f(
                tf.convert_to_tensor(x_im1, tf.float32)[tf.newaxis, :],
                tf.convert_to_tensor(qi, tf.float32)[tf.newaxis, :]
            )[0]
            x_pred = _fix_sin_cos(x_pred)
            err = _partial_state_huber(x_pred, tf.convert_to_tensor(x_i, tf.float32),
                                       delta=self.refiner.RES_DELTA,
                                       state_scale=self.refiner.state_scale,
                                       slip_lambda=self.refiner.SLIP_PRIOR)
            flags[i] = bool(err.numpy() < float(self.refiner.TOL_HUBER.numpy()))

        return refined, flags

# ======================================================================================
# (Optional) warm start cache (deployment reuse)
# ======================================================================================

class TrajectoryWarmStartCache:
    def __init__(self):
        self.last = None  # (x_T, Q, states)

    def update(self, x_T: np.ndarray, Q: np.ndarray, states: np.ndarray):
        self.last = (x_T.copy(), Q.copy(), states.copy())

    def get(self, x_T: np.ndarray, Q: np.ndarray) -> Optional[np.ndarray]:
        if self.last is None: return None
        xT_prev, Q_prev, X_prev = self.last
        if Q.shape[0] <= X_prev.shape[0]-1 and np.allclose(x_T, xT_prev, atol=1e-6):
            return X_prev[1:Q.shape[0]+1].copy()
        return None

# ======================================================================================
# Run tests when executed as a script
# ======================================================================================

if __name__ == "__main__":
    try:
        import pytest
        print(">> Running inverse dynamics tests (hard-coded settings in tests/inverse_dynamics/config.py)...")
        raise SystemExit(pytest.main(["-s", "-q", "tests/inverse_dynamics/test_inverse_dynamics_eval.py"]))
    except Exception as e:
        print("Failed to run tests:", e)
