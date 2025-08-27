# utilities/prior_trajectories.py
from __future__ import annotations
from typing import Protocol
import numpy as np

from SI_Toolkit.Predictors.predictor_autoregressive_neural import predictor_autoregressive_neural


from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
    POSE_THETA_COS_IDX,
)

STATE_DIM = 10  # keep in sync with your project

def _fix_sin_cos_np(x: np.ndarray) -> np.ndarray:
    """Return a copy of x with sin/cos consistent with heading."""
    y = np.asarray(x, dtype=np.float32).copy()
    th = float(y[POSE_THETA_IDX])
    y[POSE_THETA_SIN_IDX] = np.sin(th)
    y[POSE_THETA_COS_IDX] = np.cos(th)
    return y

class Prior(Protocol):
    """
    Generate a newest→older prior for a segment specified by:
      - x_anchor: newest state (shape [STATE_DIM])
      - Q_old_to_new: controls for that segment (shape [T, 2], in old→new order)
      - dt: integration step (float)
    Return: np.ndarray shape [T, STATE_DIM], newest→older.
    """
    def generate(self, x_anchor: np.ndarray, Q_old_to_new: np.ndarray, dt: float) -> np.ndarray: ...

# -----------------------------------------------------------------------------
# 1) Kinematic back-prop prior (your previous default)
# -----------------------------------------------------------------------------
class KinematicBackpropPrior:
    def generate(self, x_anchor: np.ndarray, Q_old_to_new: np.ndarray, dt: float) -> np.ndarray:
        x_anchor = np.asarray(x_anchor, np.float32)
        Q_old_to_new = np.asarray(Q_old_to_new, np.float32)
        T = int(Q_old_to_new.shape[0])
        out = np.zeros((T, STATE_DIM), dtype=np.float32)
        x_cur = x_anchor.copy()

        for j in range(T):
            # Walk backwards: use last control first
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

            x_prev_vec = np.array([
                wz_prev, vx_prev, vy_prev, psi_prev,
                0.0, 0.0,  # filled by _fix_sin_cos_np
                x_prev, y_prev, beta_prev, delta_prev
            ], dtype=np.float32)
            out[j] = _fix_sin_cos_np(x_prev_vec)
            x_cur  = out[j]
        return out  # newest→older

# -----------------------------------------------------------------------------
# 2) Zero-motion prior (freeze state; still keep sin/cos consistent)
# -----------------------------------------------------------------------------
class ZeroMotionPrior:
    def generate(self, x_anchor: np.ndarray, Q_old_to_new: np.ndarray, dt: float) -> np.ndarray:
        T = int(np.asarray(Q_old_to_new).shape[0])
        x0 = _fix_sin_cos_np(np.asarray(x_anchor, np.float32))
        return np.repeat(x0[None, :], T, axis=0).astype(np.float32)  # newest→older


# Hardcoded predictor parameters (adjust to your project)
_NEURAL_MODEL_NAME = "Dense-8IN-128H1-128H2-8OUT-0"         # e.g. "models/car/backprop_net"
_NEURAL_PATH_TO_MODELS = "SI_Toolkit_ASF/Experiments/04_08_RCA1_noise_reversed/Models"    # folder containing the model file
_NEURAL_DT = 0.02                            # seconds; must match the net if rnn
_NEURAL_BATCH_SIZE = 1                       # hardcoded as requested

class NeuralBackpropPrior:
    """
    Builds the predictor on first use, then rolls back from x_anchor using the
    last control first (newest→older). Output shape: [T, STATE_DIM].
    """
    def __init__(self):
        self._predictor = None

    def _ensure_predictor(self):
        if self._predictor is None:
            # Only non-trivial choices are hardcoded here as requested.
            self._predictor = predictor_autoregressive_neural(
                model_name=_NEURAL_MODEL_NAME,
                path_to_model=_NEURAL_PATH_TO_MODELS,
                dt=_NEURAL_DT,
                batch_size=_NEURAL_BATCH_SIZE,
                update_before_predicting=True,
                hls=False,
                input_quantization="float",
                disable_individual_compilation=False,
                mode=None,
            )

    def generate(self, x_anchor: np.ndarray, Q_old_to_new: np.ndarray, dt: float) -> np.ndarray:
        # Guard: user-supplied dt should match the predictor's dt (avoid silent mismatch).
        if abs(abs(float(dt)) - abs(float(_NEURAL_DT))) > 1e-9:
            raise ValueError(f"dt mismatch: got {dt}, expected {_NEURAL_DT}")

        self._ensure_predictor()

        x_anchor = np.asarray(x_anchor, np.float32)          # [STATE_DIM]
        Q_old_to_new = np.asarray(Q_old_to_new, np.float32)  # [T, C]
        if Q_old_to_new.ndim != 2:
            raise ValueError("Q_old_to_new must be shape [T, C]")
        T = int(Q_old_to_new.shape[0])
        if T == 0:
            return np.zeros((0, STATE_DIM), dtype=np.float32)

        # Newest→older rollout: use the **last** control first.
        Q_rev = Q_old_to_new[::1][None, :, :]               # [1, T, C]
        s0 = x_anchor[None, :]                                # [1, STATE_DIM]

        # Predictor returns [1, T+1, STATE_DIM] (anchor prepended at index 0).
        pred = self._predictor.predict(s0, Q_rev)[0]         # [T+1, STATE_DIM]
        traj = pred[1:, :]                                    # drop anchor → [T, STATE_DIM]

        # Keep sin/cos consistent with heading (vectorised).
        th = traj[:, POSE_THETA_IDX]
        traj[:, POSE_THETA_SIN_IDX] = np.sin(th)
        traj[:, POSE_THETA_COS_IDX] = np.cos(th)
        return traj.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def make_prior(kind: str) -> Prior:
    k = (kind or "kinematic").strip().lower()
    if k in ("kin", "kinematic", "kinematic_backprop"):
        return KinematicBackpropPrior()
    if k in ("zero", "freeze", "zeromotion", "hold"):
        return ZeroMotionPrior()
    if k in ("neural", "nn", "net", "ml"):
        return NeuralBackpropPrior()
    raise ValueError(f"Unknown prior kind: {kind!r}")
