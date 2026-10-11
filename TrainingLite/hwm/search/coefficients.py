"""Feedback law the gradient planner optimizes, instead of a list of actions.

Fifteen coefficients in ``[-1, 1]``. The first nine scale steering features, the last
six scale acceleration features. Features are dimensionless and about order 1 in
normal driving (see ``control_features``); ``HWM_COEFF_DENORM`` is the gain on each
one. The same coefficients are held for the whole plan. At every state, live or
predicted, the features are recomputed against the target waypoint that was tracked
when the plan started, and the mapping below turns them into the network action the
LLD and the car already use (``±1`` = ``±HWM_ACTION_DENORM``).

Steering is a desired wheel angle relative to the measured wheel angle, with access to
the full physical range. The simulator/real steering servo, not this policy, enforces
the wheel's rate limit. Acceleration is an absolute command. Both go through ``tanh``,
so a coefficient sitting on its limit still has a gradient.
"""

from __future__ import annotations

import math

import torch

from TrainingLite.hwm.shared.state_indices import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    STEERING_ANGLE_IDX,
)
from TrainingLite.hwm.shared.targets import build_target_states, target_index
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings

# Steering, then acceleration. Order is the coefficient vector's order.
STEER_NAMES = (
    "bias",
    "lateral",
    "heading",
    "yaw_rate",
    "steering",
    "lat_vel",
    "wall_side",
    "wall_push",
    "clearance",
)
ACCEL_NAMES = (
    "bias",
    "speed_error",
    "speed",
    "turn",
    "urgency",
    "clearance",
)
COEFF_NAMES = tuple(f"steer.{name}" for name in STEER_NAMES) + tuple(f"accel.{name}" for name in ACCEL_NAMES)
N_STEER = len(STEER_NAMES)
N_ACCEL = len(ACCEL_NAMES)
COEFF_DIM = N_STEER + N_ACCEL

# Speed features: 1 at a 2 m/s error, and 1 at 2 m/s forward. Heading is already / π.
SPEED_FEATURE_MPS = 2.0


def steer_lock_rad() -> float:
    """Wheel angle at a network steering command of ±1.

    The network command is multiplied by ``HWM_ACTION_DENORM[0]`` and then by the car's
    ``HWM_STEERING_MULTIPLIER`` (``s_max / steering_multiplier`` is the denorm).
    """
    return float(Settings.HWM_ACTION_DENORM[0]) * float(Settings.HWM_STEERING_MULTIPLIER)


def frozen_reference(state: torch.Tensor, wall_geometry: WallGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """``(target (B, state_dim), lookahead (B,))`` for the waypoint tracked at ``state``.

    Both are detached. The target stays the one chosen here for the whole plan, and the
    lookahead distance (meters, at least 0.5) is the scale of the lateral feature, so
    that scale does not blow up as the car reaches the waypoint.
    """
    raceline = wall_geometry.raceline_abs.to(device=state.device, dtype=state.dtype)
    targets = build_target_states(raceline)
    index = target_index(state[:, [POSE_X_IDX, POSE_Y_IDX]], raceline)
    target = targets[index].detach()
    delta = target[:, [POSE_X_IDX, POSE_Y_IDX]] - state[:, [POSE_X_IDX, POSE_Y_IDX]]
    lookahead = torch.linalg.norm(delta, dim=-1).clamp_min(0.5).detach()
    return target, lookahead


def pure_pursuit_coefficients(device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Hand-set gains: steer toward the lateral target error, unwind the current wheel angle, P on speed.

    The lateral feature shifts the desired wheel angle toward the target. The steering
    feature damps that correction as the measured wheel turns, and speed error is a
    proportional acceleration command.
    """
    coeffs = torch.zeros(COEFF_DIM, device=device, dtype=dtype)
    coeffs[STEER_NAMES.index("lateral")] = 1.0
    coeffs[STEER_NAMES.index("steering")] = -1.0
    coeffs[N_STEER + ACCEL_NAMES.index("speed_error")] = 1.0
    return coeffs


def coefficient_denorm(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``(COEFF_DIM,)`` gains from ``HWM_COEFF_DENORM``."""
    denorm = torch.as_tensor(Settings.HWM_COEFF_DENORM, device=device, dtype=dtype).reshape(-1)
    if int(denorm.shape[0]) != COEFF_DIM:
        raise ValueError(f"HWM_COEFF_DENORM must have length {COEFF_DIM}, got {int(denorm.shape[0])}")
    return denorm


def control_features(
    state: torch.Tensor,
    target: torch.Tensor,
    lookahead: torch.Tensor,
    wall_geometry: WallGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(steering features (B, 9), acceleration features (B, 6))`` for raw ``state``.

    The target and lookahead come from ``frozen_reference`` at the plan start. Wall
    features are detached: the clearance field is spiky near corners, and the gradient
    the planner needs is through the coefficients and the dynamics, not through the map.
    """
    theta = state[:, POSE_THETA_IDX]
    dx = target[:, POSE_X_IDX] - state[:, POSE_X_IDX]
    dy = target[:, POSE_Y_IDX] - state[:, POSE_Y_IDX]
    lateral = (-torch.sin(theta) * dx + torch.cos(theta) * dy) / lookahead.reshape(-1)
    heading = target[:, POSE_THETA_IDX] - theta
    heading = torch.atan2(torch.sin(heading), torch.cos(heading)) / math.pi
    yaw = state[:, ANGULAR_VEL_Z_IDX] / float(Settings.HWM_STATE_SCALE[ANGULAR_VEL_Z_IDX])
    wheel = state[:, STEERING_ANGLE_IDX] / steer_lock_rad()
    lat_vel = state[:, LINEAR_VEL_Y_IDX] / float(Settings.HWM_STATE_SCALE[LINEAR_VEL_Y_IDX])
    wall_side, urgency, clearance = wall_geometry.controller_wall_features(state)
    steer = torch.stack(
        (
            torch.ones_like(lateral),
            lateral,
            heading,
            yaw,
            wheel,
            lat_vel,
            wall_side,
            wall_side * urgency,
            clearance,
        ),
        dim=-1,
    )
    speed_error = (target[:, LINEAR_VEL_X_IDX] - state[:, LINEAR_VEL_X_IDX]) / SPEED_FEATURE_MPS
    speed = state[:, LINEAR_VEL_X_IDX] / SPEED_FEATURE_MPS
    turn = torch.maximum(heading.abs(), wheel.abs())
    accel = torch.stack(
        (
            torch.ones_like(speed),
            speed_error,
            speed,
            turn,
            urgency,
            clearance,
        ),
        dim=-1,
    )
    return steer, accel


def actions_from_coefficients(
    coefficients: torch.Tensor,
    state: torch.Tensor,
    target: torch.Tensor,
    lookahead: torch.Tensor,
    wall_geometry: WallGeometry,
) -> torch.Tensor:
    """Network action ``(B, 2)`` for coefficients ``(B, COEFF_DIM)`` at raw ``state``.

    ``steer_arg = (c * denorm * φ).sum / sqrt(n)`` shifts the desired wheel angle
    relative to the measured wheel. The measured wheel fraction is first mapped through
    ``atanh``, so a zero shift reproduces it exactly; ``tanh`` of the shifted value can
    request the complete ``[-1, 1]`` steering range. The simulator's steering servo
    applies its physical rate limit. Acceleration is ``tanh`` of the same kind of sum,
    already in network units.
    """
    phi_s, phi_a = control_features(state, target, lookahead, wall_geometry)
    denorm = coefficient_denorm(state.device, state.dtype)
    c_s = coefficients[:, :N_STEER]
    c_a = coefficients[:, N_STEER:]
    steer_arg = (c_s * denorm[:N_STEER] * phi_s).sum(dim=-1) / math.sqrt(N_STEER)
    accel_arg = (c_a * denorm[N_STEER:] * phi_a).sum(dim=-1) / math.sqrt(N_ACCEL)
    lock = steer_lock_rad()
    measured = (state[:, STEERING_ANGLE_IDX] / lock).clamp(-0.999, 0.999)
    steer_cmd = torch.tanh(torch.atanh(measured) + steer_arg)
    accel_cmd = torch.tanh(accel_arg)
    return torch.stack((steer_cmd, accel_cmd), dim=-1)
