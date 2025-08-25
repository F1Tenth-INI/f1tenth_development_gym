# utilities/prior_trajectories.py
from __future__ import annotations
from typing import Protocol
import numpy as np

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

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def make_prior(kind: str) -> Prior:
    k = (kind or "kinematic").strip().lower()
    if k in ("kin", "kinematic", "kinematic_backprop"):
        return KinematicBackpropPrior()
    if k in ("zero", "freeze", "zeromotion", "hold"):
        return ZeroMotionPrior()
    raise ValueError(f"Unknown prior kind: {kind!r}")
