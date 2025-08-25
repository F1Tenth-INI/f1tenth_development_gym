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

import contextlib  # ← add if missing

from utilities.prior_trajectories import make_prior

# --- Portable device heuristics & defaults ---
_LOGICAL_GPUS = tf.config.list_logical_devices('GPU')
_HAS_GPU = len(_LOGICAL_GPUS) > 0
_IS_METAL = any('METAL' in d.name.upper() for d in _LOGICAL_GPUS)

PIN_REFINER_TO_CPU = bool(int(os.getenv('ID_PIN_REFINER_CPU', '1')))
CPU_DEVICE = "/device:CPU:0"

XLA_REFINE = bool(int(os.getenv('ID_USE_XLA', '0')))

ID_PRIOR_KIND = os.getenv("ID_PRIOR_KIND", "kinematic")  # NEW



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

        dev_ctx = tf.device(CPU_DEVICE) if PIN_REFINER_TO_CPU else contextlib.nullcontext()
        with dev_ctx:
            self.state_scale = tf.Variable(scale_np, dtype=tf.float32, trainable=False)

            # Persistent buffers
            self.X_var      = tf.Variable(tf.zeros([self.MAX_T, self.state_dim], tf.float32), trainable=True)
            self.Q_var      = tf.Variable(tf.zeros([self.MAX_T, self.control_dim], tf.float32), trainable=False)
            self.x_next_var = tf.Variable(tf.zeros([self.state_dim], tf.float32), trainable=False)
            self.T_var      = tf.Variable(0, dtype=tf.int32, trainable=False)

            # ---- PATH PRIOR (optional; filled by ProgressiveWindowRefiner) ----
            # Prior targets (kinematic reference) and per-step weights for current horizon
            self.KP_var = tf.Variable(tf.zeros([self.MAX_T, self.state_dim], tf.float32), trainable=False, name="prior_targets")
            self.KW_var = tf.Variable(tf.zeros([self.MAX_T], tf.float32),                    trainable=False, name="prior_weights")

        if int(os.getenv("ID_LOG_REFINER_DEV", "1")):
            print(f"[ID] Refiner pinned_to_cpu={PIN_REFINER_TO_CPU} | XLA_REFINE={XLA_REFINE} | device={CPU_DEVICE}")

        # Optimizer
        try:
            opt = tf.keras.optimizers.legacy.Adam
        except AttributeError:
            opt = tf.keras.optimizers.Adam
        self.base_optimizer = opt(learning_rate=1e-3)

        # ---- Diagnostics (filled per call) ----
        self.debug_last_timings: Dict[str, float] = {}
        self.debug_last_dyn_costs: Tuple[float, float] | None = None  # (mean, max)


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
        self.ADAM_ITERATIONS_LIMIT = 40
        self.LR1          = 5e-2
        self.LR2          = 1e-4

    def inverse_entire_trajectory(self, x_T: np.ndarray, Q: np.ndarray,
                                  X_init: Optional[np.ndarray]=None,
                                  compute_flags: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine newest→older states over the full horizon.
        If compute_flags=False (default), we skip the expensive per-step rollout that
        produces 'flags' and return an all-True vector (shape-correct, cheap).
        Set compute_flags=True only for one final consistency check or standalone use.
        """
        import time
        t_start = time.time()

        x_T = np.asarray(x_T, dtype=np.float32); Q = np.asarray(Q, dtype=np.float32)
        if x_T.shape != (1, self.state_dim):
            raise ValueError(f"x_T must be [1,{self.state_dim}]")
        if Q.ndim != 2 or Q.shape[1] != self.control_dim:
            raise ValueError(f"Q must be [T,{self.control_dim}]")

        T = Q.shape[0]
        if T > self.MAX_T:
            raise ValueError(f"T={T} exceeds MAX_T={self.MAX_T}")

        if X_init is None:
            raise ValueError("X_init must be provided by caller (seed from prior).")

        # Load buffers (SLICE assignments only; avoid O(MAX_T) host->device copies)
        self.x_next_var.assign(x_T[0])  # pinned as given
        self.T_var.assign(T)

        dev_ctx = tf.device(CPU_DEVICE) if PIN_REFINER_TO_CPU else contextlib.nullcontext()
        # Build full buffers to avoid ResourceStridedSliceAssign (GPU-unsafe)
        q_full = np.zeros((self.MAX_T, self.control_dim), dtype=np.float32)
        x_full = np.zeros((self.MAX_T, self.state_dim), dtype=np.float32)
        q_full[:T, :] = Q[::-1].astype(np.float32)  # newest→older internal order
        x_full[:T, :] = X_init.astype(np.float32)  # newest→older seed

        with dev_ctx:
            self.Q_var.assign(q_full)
            self.X_var.assign(x_full)

        t_prep = time.time()

        # Optimize (compiled; no outputs)
        self._optimize_compiled()
        t_opt = time.time()

        # Assemble outputs (eager)
        X_np = self.X_var[:T].numpy()
        for i in range(T):
            X_np[i] = _fix_sin_cos(tf.convert_to_tensor(X_np[i], tf.float32)).numpy()
        states = np.vstack([x_T[0], X_np])  # newest→older

        # Optional per-step rollout for consistency flags (expensive)
        dyn_costs = []
        if compute_flags:
            Q_newest_first = self.Q_var[:T].numpy()
            flags = np.zeros((T,), dtype=bool)
            for i in range(T):
                x_i = states[0] if i == 0 else states[i]
                x_im1 = states[i + 1]
                qi = Q_newest_first[i]
                x_pred = self._f(
                    tf.convert_to_tensor(x_im1, tf.float32)[tf.newaxis, :],
                    tf.convert_to_tensor(qi, tf.float32)[tf.newaxis, :]
                )[0]
                x_pred = _fix_sin_cos(x_pred)
                err = _partial_state_huber(x_pred, tf.convert_to_tensor(x_i, tf.float32),
                                           delta=self.RES_DELTA,
                                           state_scale=self.state_scale,
                                           slip_lambda=self.SLIP_PRIOR)
                v = float(err.numpy())
                dyn_costs.append(v)
                flags[i] = bool(v < float(self.TOL_HUBER.numpy()))
        else:
            # Cheap, shape-correct placeholder; callers that care compute real flags once later.
            flags = np.ones((T,), dtype=bool)

        t_assemble = time.time()

        # Save diagnostics
        self.debug_last_timings = {
            "T": float(T),
            "prep_ms": 1000.0*(t_prep - t_start),
            "opt_ms":  1000.0*(t_opt - t_prep),
            "assemble_ms": 1000.0*(t_assemble - t_opt),
            "total_ms": 1000.0*(t_assemble - t_start),
            "traces": float(self.tracing_count() or -1),
            "pinned_cpu": float(1 if PIN_REFINER_TO_CPU else 0),
            "xla": float(1 if XLA_REFINE else 0),
            "adam_iterations_limit": float(self.ADAM_ITERATIONS_LIMIT),
        }
        if len(dyn_costs):
            self.debug_last_dyn_costs = (float(np.mean(dyn_costs)), float(np.max(dyn_costs)))
            self.debug_last_timings["dyn_cost_mean"] = self.debug_last_dyn_costs[0]
            self.debug_last_timings["dyn_cost_max"] = self.debug_last_dyn_costs[1]

        return states, flags

    @tf.function(experimental_relax_shapes=True, reduce_retracing=True, jit_compile=XLA_REFINE)
    def _optimize_compiled(self) -> None:
        """
        Compiled optimization only.
        - No TensorArray creation; no graph outputs (updates self.X_var in place).
        - Robust dynamics + robust temporal smoothness.
        - Optional PATH PRIOR (small Huber penalty towards kinematic seed) applied
          ONLY where self.KW_var[i] > 0.0 (set/cleared by ProgressiveWindowRefiner).
        - bounded adaptive loop (cosine LR, early-stop, gradient clipping).
        """
        dev_ctx = tf.device(CPU_DEVICE) if PIN_REFINER_TO_CPU else contextlib.nullcontext()
        with dev_ctx:
            T = self.T_var

            def cost():
                cost_val = tf.constant(0.0, tf.float32)
                for i in tf.range(T):
                    x_i  = self.x_next_var if tf.equal(i, 0) else self.X_var[i-1, :]
                    x_i  = _fix_sin_cos(x_i)
                    xim1 = _fix_sin_cos(self.X_var[i, :])
                    qi   = self.Q_var[i, :]
                    x_pred = self._f(xim1[tf.newaxis, :], qi[tf.newaxis, :])[0]
                    x_pred = _fix_sin_cos(x_pred)
                    c_dyn = _partial_state_huber(x_pred, x_i, delta=self.RES_DELTA,
                                              state_scale=self.state_scale, slip_lambda=self.SLIP_PRIOR)

                    cost_val += c_dyn

                    # ---- PATH PRIOR (if enabled for this step) -----------------
                    # Prior targets (kinematic reference) newest→older in KP_var
                    # Prior weights (per step) in KW_var; 0.0 disables the term.
                    w_i = self.KW_var[i]

                    def _add_prior_term():
                        x_state = _fix_sin_cos(self.X_var[i, :])  # current var state
                        kp_i = _fix_sin_cos(self.KP_var[i, :])  # prior target
                        # Use TEMP_DELTA for prior's Huber; no slip prior here
                        c_pr = _partial_state_huber(
                            x_state, kp_i,
                            delta=self.TEMP_DELTA,
                            state_scale=self.state_scale,
                            slip_lambda=0.0
                        )
                        return cost_val + w_i * c_pr

                    cost_val = tf.cond(w_i > 0.0, _add_prior_term, lambda: cost_val)

                # --- Robust temporal smoothness (mild) ---
                for i in tf.range(1, T):
                    x_cur  = _fix_sin_cos(self.X_var[i, :])
                    x_prev = _fix_sin_cos(self.X_var[i-1, :])
                    cost_val += self.REG_TEMPORAL * _partial_state_huber(
                        x_cur, x_prev, delta=self.TEMP_DELTA, state_scale=self.state_scale, slip_lambda=self.SLIP_PRIOR
                    )
                return cost_val

            # bounded adaptive loop (cosine LR, early-stop, grad-clip)
            # allow early exit once relative improvement stalls (with a tiny patience).
            max_iters = tf.constant(int(self.ADAM_ITERATIONS_LIMIT), tf.int32)
            min_iters = tf.constant(2, tf.int32)  # always take a couple of steps to settle
            patience = tf.constant(2, tf.int32)  # tolerate a couple of no-improve iterations
            rel_tol = tf.constant(5e-4, tf.float32)  # ~0.05% relative improvement threshold
            clip_norm = tf.constant(1.0, tf.float32)  # small global clip for stability
            pi = tf.constant(np.pi, tf.float32)

            def _cosine_lr(t, Tm):
                # Anneal smoothly from LR1 → LR2 over the short schedule (0..Tm-1)
                x = tf.cast(t, tf.float32) / tf.maximum(1.0, tf.cast(Tm - 1, tf.float32))
                return tf.convert_to_tensor(self.LR2, tf.float32) + \
                    0.5 * (tf.convert_to_tensor(self.LR1, tf.float32) - tf.convert_to_tensor(self.LR2, tf.float32)) * \
                    (1.0 + tf.cos(pi * x))

            i0 = tf.constant(0, tf.int32)
            best0 = tf.constant(np.inf, tf.float32)
            prev0 = tf.constant(np.inf, tf.float32)
            noimp0 = tf.constant(0, tf.int32)

            def _cond(i, best, prev, noimp):
                # Bound runtime and permit early stop after min_iters if no progress.
                base = i < max_iters
                after_min = i >= min_iters
                give_up = tf.logical_and(after_min, noimp >= patience)
                return tf.logical_and(base, tf.logical_not(give_up))

            def _body(i, best, prev, noimp):
                lr = _cosine_lr(i, max_iters)
                try:
                    self.base_optimizer.learning_rate.assign(lr)
                except Exception:
                    pass

                with tf.GradientTape() as tape:
                    c = cost()
                g = tape.gradient(c, [self.X_var])[0]
                g = tf.clip_by_norm(g, clip_norm)  # cheap safety against spikes
                self.base_optimizer.apply_gradients([(g, self.X_var)])

                # Relative improvement vs previous iteration. On the very first step
                # 'prev' is inf → rel becomes NaN → counts as "no improvement", which is fine
                # because we enforce 'min_iters'.
                rel = (prev - c) / (tf.abs(prev) + 1e-8)
                improved = rel > rel_tol

                prev_new = c
                best_new = tf.minimum(best, c)
                noimp_new = tf.where(improved, tf.constant(0, tf.int32), noimp + 1)
                return i + 1, best_new, prev_new, noimp_new

            _ = tf.while_loop(_cond, _body, (i0, best0, prev0, noimp0))

    # --- prior API (called by ProgressiveWindowRefiner) ---
    def set_path_prior(self, prior_targets: np.ndarray, prior_weights: np.ndarray):
        T = int(prior_targets.shape[0])
        dev_ctx = tf.device(CPU_DEVICE) if PIN_REFINER_TO_CPU else contextlib.nullcontext()
        with dev_ctx:
            kp_full = np.zeros((self.MAX_T, self.state_dim), dtype=np.float32)
            kw_full = np.zeros((self.MAX_T,), dtype=np.float32)
            kp_full[:T, :] = prior_targets.astype(np.float32)
            kw_full[:T] = prior_weights.astype(np.float32)
            self.KP_var.assign(kp_full)
            self.KW_var.assign(kw_full)

    def clear_path_prior(self):
        dev_ctx = tf.device(CPU_DEVICE) if PIN_REFINER_TO_CPU else contextlib.nullcontext()
        with dev_ctx:
            self.KW_var.assign(tf.zeros_like(self.KW_var))

    # --- trace counter (to verify retracing isn’t happening repeatedly) ---
    def tracing_count(self) -> Optional[int]:
        try:
            return self._optimize_compiled.experimental_get_tracing_count()
        except Exception:
            return None


# ======================================================================================
# Progressive overlapping windows (refine-only)
# ======================================================================================

class ProgressiveWindowRefiner:
    """
    Refine-only, progressive-grow with overlap (no LM).
    - Build full newest→older kinematic seed once.
    - For k_end = W, W+(W-O), ...: refine the prefix pinned at the measured newest state.
    """

    def __init__(self,
                 mu: Optional[float]=None,
                 controls_are_pid: bool=True,
                 dt: Optional[float]=None,
                 window_size: int = 30,
                 overlap: int = 10,
                 # --- NEW: smoothing pass controls ---
                 smoothing_window: Optional[int] = None,
                 smoothing_overlap: Optional[int] = None,
                 # --- NEW: path-prior schedule ---
                 prior_weight0: float = 1e-3,
                 prior_decay: float = 0.5,
                 prior_min: float = 1e-5,
                 prior_kind: Optional[str] = None,
                 state_scale_std_map: Optional[Dict[str,float]]=None):

        if smoothing_window is not None:
            smoothing_window = min(smoothing_window, window_size)
        if overlap < 0 or window_size <= 0 or overlap >= window_size:
            raise ValueError("window_size>0 and 0 <= overlap < window_size required.")
        self.refiner = TrajectoryRefiner(mu=mu, controls_are_pid=controls_are_pid, dt=dt,
                                         state_scale_std_map=state_scale_std_map)
        # Blend windows a bit more strongly
        self.refiner.REG_TEMPORAL = 1e-2
        self.window_size = int(window_size)
        self.overlap     = int(overlap)
        self.dt = self.refiner.dt

        self.prior_kind = (prior_kind or ID_PRIOR_KIND)
        self.prior = make_prior(self.prior_kind)
        self.prior_w0 = max(0.0, float(prior_weight0))
        self.prior_decay = float(prior_decay)  # allow 0..1 (or >1 if you want)
        self.prior_min = max(0.0, float(prior_min))

        # --- Debug/render hooks (new): last used prior over the full horizon ---
        self._last_prior_newest_first = None  # np.ndarray [T+1,10], newest→older (row 0 is x_T)
        self._last_prior_mask = None          # np.ndarray [T], True where prior existed for that step


        # --- Smoothing normalization ---
        if smoothing_window is None or int(smoothing_window) <= 0:
            self.smooth_W = None
            self.smooth_O = None
        else:
            w = int(min(int(smoothing_window), self.window_size))  # 1..window_size
            # default overlap = half-window
            o = int(smoothing_overlap) if smoothing_overlap is not None else (w // 2)
            # clamp to 0 <= o < w
            o = max(0, min(o, w - 1))
            self.smooth_W, self.smooth_O = w, o


    def _progressive_refine_core(
        self,
        x_T: np.ndarray,
        Q: np.ndarray,
        *,
        collect_stats: bool,
        collect_prior: bool,
    ):
        import time
        x_T = np.asarray(x_T, np.float32); Q = np.asarray(Q, np.float32)
        T = int(Q.shape[0]); W = self.window_size; O = self.overlap
        if x_T.ndim == 1: x_T = x_T[None, :]
        if x_T.shape != (1, self.refiner.state_dim): raise ValueError("x_T bad shape")
        if Q.ndim != 2 or Q.shape[1] != self.refiner.control_dim: raise ValueError("Q bad shape")
        if T == 0:
            empty = np.zeros((0,), dtype=bool)
            return x_T.copy(), empty, ({} if collect_stats else None)

        # Optional stats bag
        stats = {} if collect_stats else None
        if collect_stats:
            t0 = time.time()
            stats["T"] = float(T); stats["W"] = float(W); stats["O"] = float(O)
            stats["smooth_W"] = float(self.smooth_W or 0); stats["smooth_O"] = float(self.smooth_O or 0)

        # Initial seed for the first window only
        k_end = min(W, T)
        refined = np.zeros((T + 1, x_T.shape[1]), dtype=np.float32)
        refined[0] = x_T[0]
        seed_prefix = self.prior.generate(x_T[0], Q[T - k_end: T], self.dt).astype(np.float32)
        refined[1: k_end + 1] = seed_prefix

        # Prior capture (for overlays) only if asked
        if collect_prior:
            prior_accum = np.full((T, refined.shape[1]), np.nan, dtype=np.float32)
            filled = np.zeros((T,), dtype=bool)

        prev_k_end = 0
        lam = self.prior_w0
        t_inv_sum = 0.0

        while True:
            Q_seg  = Q[T - k_end : T]
            X_init = refined[1 : k_end + 1]

            grow_sz = k_end - prev_k_end if prev_k_end > 0 else k_end
            prior_targets = np.zeros((k_end, refined.shape[1]), dtype=np.float32)
            prior_weights = np.zeros((k_end,), dtype=np.float32)
            if grow_sz > 0:
                x_boundary = refined[prev_k_end].astype(np.float32)
                Q_tail = Q_seg[-grow_sz:].astype(np.float32)
                seed_tail = self.prior.generate(x_boundary, Q_tail, self.dt).astype(np.float32)
                prior_targets[-grow_sz:, :] = seed_tail
                prior_weights[-grow_sz:] = lam
                if collect_prior:
                    start = k_end - grow_sz
                    prior_accum[start:k_end, :] = seed_tail
                    filled[start:k_end] = True

            self.refiner.set_path_prior(prior_targets, prior_weights)
            if collect_stats: import time as _t; t0i = _t.time()
            states_w, _ = self.refiner.inverse_entire_trajectory(x_T=x_T, Q=Q_seg, X_init=X_init)
            if collect_stats: t_inv_sum += (_t.time() - t0i) * 1000.0
            self.refiner.clear_path_prior()

            refined[:k_end+1] = states_w
            prev_k_end = k_end
            if k_end == T: break
            lam = max(self.prior_min, lam * self.prior_decay)
            k_end = min(k_end + (W - O), T)

        if collect_stats: stats["inv_ms"] = t_inv_sum

        # Optional smoothing pass (same behavior as before)
        t_smooth_ms = 0.0
        if self.smooth_W is not None:
            W2 = self.smooth_W; O2 = self.smooth_O; step2 = max(1, W2 - O2)
            k2 = min(W2, T)
            while True:
                Q_seg  = Q[T - k2 : T]
                X_init = refined[1 : k2 + 1]
                self.refiner.clear_path_prior()
                if collect_stats: import time as _t; s0 = _t.time()
                states_w, _ = self.refiner.inverse_entire_trajectory(x_T=x_T, Q=Q_seg, X_init=X_init)
                if collect_stats: t_smooth_ms += (_t.time() - s0) * 1000.0
                refined[:k2+1] = states_w
                if k2 == T: break
                k2 = min(k2 + step2, T)
        if collect_stats: stats["smooth_ms"] = t_smooth_ms

        # Flags (unchanged)
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

        # Publish prior overlay only if requested
        if collect_prior:
            self._last_prior_newest_first = np.vstack([x_T[0], prior_accum.astype(np.float32)])
            self._last_prior_mask = filled

        # Finalize stats
        if collect_stats:
            low = getattr(self.refiner, "debug_last_timings", {})
            for k, v in low.items():
                stats[f"last_{k}"] = float(v)
            import time as _t
            stats["total_ms"] = (_t.time() - t0) * 1000.0

        return refined, flags, (stats if collect_stats else None)


    def refine(self, x_T: np.ndarray, Q: np.ndarray):
        states, flags, _ = self._progressive_refine_core(
            x_T, Q, collect_stats=False, collect_prior=False
        )
        return states, flags


    def refine_stats(self, x_T: np.ndarray, Q: np.ndarray):
        states, flags, stats = self._progressive_refine_core(
            x_T, Q, collect_stats=True, collect_prior=True
        )
        return states, flags, stats

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
