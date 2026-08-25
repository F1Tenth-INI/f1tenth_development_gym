"""Fast recoverability checks for the predictive safety filter.

Certify the base controller while clearance is comfortable, or while one
more nominal step still leaves a cheap emergency brake manoeuvre.  Full
multi-rollout probes run only in a narrow margin band near the track edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .psf_track_relative import (
    car_state_to_xr_from_waypoints,
    current_border_clearance,
    probe_track_at_xy,
    trajectory_track_metrics,
    wrap_angle,
)
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    control_limits_high,
    control_limits_low,
)


@dataclass
class RecoveryProbeResult:
    recoverable: bool
    u_hold: np.ndarray | None
    trajectory: np.ndarray | None
    min_border_clearance: float
    terminal_speed: float
    n_evals: int


def _rollout_pacejka(
    s0: np.ndarray,
    Q: np.ndarray,
    car_params_jax,
    dt: float,
    state_history,
    control_history,
) -> np.ndarray:
    import jax.numpy as jnp
    from sim.f110_sim.envs.car_model_jax import car_steps_sequential_jax

    traj = car_steps_sequential_jax(
        jnp.asarray(s0, dtype=jnp.float32),
        jnp.asarray(Q, dtype=jnp.float32),
        car_params_jax,
        float(dt),
        int(Q.shape[0]),
        model_type="pacejka",
        state_history=state_history,
        control_history=control_history,
    )
    return np.asarray(traj, dtype=np.float64)


def rollout_one_step(
    s0: np.ndarray,
    u0: np.ndarray,
    car_params_jax,
    dt: float,
    state_history,
    control_history,
) -> np.ndarray:
    Q = np.tile(np.asarray(u0, dtype=np.float64).reshape(1, 2), (1, 1))
    return _rollout_pacejka(s0, Q, car_params_jax, dt, state_history, control_history)[-1]


def terminal_speed_on_track(trajectory: np.ndarray, waypoints: np.ndarray) -> float:
    xr, _ = car_state_to_xr_from_waypoints(trajectory[-1], waypoints)
    return float(np.hypot(xr[2], xr[3]))


def trajectory_recovery_metrics(
    trajectory: np.ndarray,
    waypoints: np.ndarray,
    lf: float,
    width: float,
    margin: float,
    terminal_stop_speed: float,
) -> dict[str, float]:
    metrics = trajectory_track_metrics(trajectory, waypoints, lf, width, margin)
    speed = terminal_speed_on_track(trajectory, waypoints)
    path_ok = float(metrics["min_border_clearance"]) >= 0.0
    # Safe terminal preferred, but path-feasible emergency hold is enough to certify.
    return {
        "min_border_clearance": float(metrics["min_border_clearance"]),
        "terminal_speed": speed,
        "path_ok": float(path_ok),
        "terminal_safe": float(path_ok and speed <= terminal_stop_speed),
        "recoverable": float(path_ok),
    }


def _emergency_holds(steer_bias: float = 0.0) -> list[np.ndarray]:
    low = np.asarray(control_limits_low, dtype=np.float64)
    high = np.asarray(control_limits_high, dtype=np.float64)
    steers = np.unique(
        np.clip(
            np.array(
                [
                    steer_bias,
                    steer_bias - 0.15,
                    steer_bias + 0.15,
                    steer_bias - 0.35,
                    steer_bias + 0.35,
                    0.0,
                    float(low[0]),
                    float(high[0]),
                ],
                dtype=np.float64,
            ),
            low[0],
            high[0],
        )
    )
    accels = np.unique(
        np.clip(
            np.array([float(low[1]), -8.0, -4.0, -1.0], dtype=np.float64),
            low[1],
            high[1],
        )
    )
    return [np.array([float(s), float(a)], dtype=np.float64) for a in accels for s in steers]


def probe_recovery_from_state(
    s0: np.ndarray,
    waypoints: np.ndarray,
    car_params_jax,
    dt: float,
    horizon: int,
    lf: float,
    width: float,
    margin: float,
    terminal_stop_speed: float,
    state_history,
    control_history,
    *,
    steer_bias: float = 0.0,
    max_evals: int = 6,
) -> RecoveryProbeResult:
    """Return True if some constant hold stays on track and ends slow."""
    best: RecoveryProbeResult | None = None
    n_evals = 0
    for u_hold in _emergency_holds(steer_bias=steer_bias):
        if n_evals >= int(max_evals):
            break
        n_evals += 1
        Q = np.tile(u_hold.reshape(1, 2), (int(horizon), 1))
        traj = _rollout_pacejka(s0, Q, car_params_jax, dt, state_history, control_history)
        met = trajectory_recovery_metrics(
            traj, waypoints, lf, width, margin, terminal_stop_speed
        )
        cand = RecoveryProbeResult(
            recoverable=bool(met["recoverable"]),
            u_hold=u_hold.copy(),
            trajectory=traj,
            min_border_clearance=float(met["min_border_clearance"]),
            terminal_speed=float(met["terminal_speed"]),
            n_evals=n_evals,
        )
        if cand.recoverable:
            return cand
        if best is None or cand.min_border_clearance > best.min_border_clearance:
            best = cand

    if best is None:
        return RecoveryProbeResult(
            recoverable=False,
            u_hold=None,
            trajectory=None,
            min_border_clearance=float("-inf"),
            terminal_speed=float("inf"),
            n_evals=n_evals,
        )
    return best


def should_certify_u_d(
    s0: np.ndarray,
    u_d: np.ndarray,
    waypoints: np.ndarray,
    car_params_jax,
    dt: float,
    horizon: int,
    lf: float,
    width: float,
    margin: float,
    terminal_stop_speed: float,
    state_history,
    control_history,
    *,
    probe_clearance: float = 0.12,
    max_probe_evals: int = 6,
    recovery_horizon: int | None = None,
) -> tuple[bool, str, float, int]:
    """Certify ``u_d`` only if a short open-loop hold still leaves a recovery option.

    Uses a short multi-step hold of ``u_d`` (not the full horizon).  Feedback
    controllers change ``u`` every tick, but if even a few identical steps leave
    the track, the current command is already unsafe.
    """
    H = int(recovery_horizon or horizon)
    n_evals = 0
    s0 = np.asarray(s0, dtype=np.float64).reshape(-1)
    c0 = current_border_clearance(s0, waypoints, lf, width, margin)
    if c0 < 0.0:
        return False, "off_track_now", c0, n_evals

    u_d = np.clip(
        np.asarray(u_d, dtype=np.float64).reshape(2),
        control_limits_low,
        control_limits_high,
    )

    vx = float(abs(s0[LINEAR_VEL_X_IDX]))
    dyn_probe = max(float(probe_clearance), 0.08 + 0.06 * vx)
    # Look ahead ~0.3–0.5 s of constant u_d (enough to catch wall approaches).
    H_ud = int(np.clip(round(0.40 / max(dt, 1e-3)), 6, min(12, H)))

    Q_ud = np.tile(u_d.reshape(1, 2), (H_ud, 1))
    traj_ud = _rollout_pacejka(s0, Q_ud, car_params_jax, dt, state_history, control_history)
    n_evals += 1
    met_ud = trajectory_recovery_metrics(
        traj_ud, waypoints, lf, width, margin, terminal_stop_speed
    )
    c_path = float(met_ud["min_border_clearance"])
    if c_path < 0.0:
        return False, "ud_leaves_track", c_path, n_evals

    if c0 >= dyn_probe and c_path >= 0.5 * dyn_probe:
        return True, "ud_path_ok", c_path, n_evals

    s1 = traj_ud[0]
    track = probe_track_at_xy(waypoints, float(s1[POSE_X_IDX]), float(s1[POSE_Y_IDX]))
    mu = float(wrap_angle(s1[POSE_THETA_IDX] - track["psi_t"]))
    steer_bias = float(np.clip(-1.5 * track["e_lat"] - 0.8 * mu, -0.4, 0.4))
    probe = probe_recovery_from_state(
        s1,
        waypoints,
        car_params_jax,
        dt,
        H,
        lf,
        width,
        margin,
        terminal_stop_speed,
        state_history,
        control_history,
        steer_bias=steer_bias,
        max_evals=max_probe_evals,
    )
    n_evals += probe.n_evals
    return bool(probe.recoverable), (
        "probe_ok" if probe.recoverable else "unrecoverable"
    ), c_path, n_evals


def find_minimal_recovery_u0(
    s0: np.ndarray,
    u_d: np.ndarray,
    waypoints: np.ndarray,
    car_params_jax,
    dt: float,
    horizon: int,
    lf: float,
    width: float,
    margin: float,
    terminal_stop_speed: float,
    state_history,
    control_history,
    *,
    steer_bias: float = 0.0,
    max_evals: int = 10,
    recovery_horizon: int | None = None,
) -> tuple[np.ndarray | None, RecoveryProbeResult | None, int]:
    """Smallest change to ``u_d`` that keeps the next state recoverable."""
    H = int(recovery_horizon or horizon)
    u_d = np.clip(
        np.asarray(u_d, dtype=np.float64).reshape(2),
        control_limits_low,
        control_limits_high,
    )
    low = np.asarray(control_limits_low, dtype=np.float64)
    high = np.asarray(control_limits_high, dtype=np.float64)

    near: list[np.ndarray] = [
        np.clip(u_d + np.array([0.0, -2.0]), low, high),
        np.clip(u_d + np.array([0.0, -5.0]), low, high),
        np.clip(u_d + np.array([0.0, float(low[1])]), low, high),
    ]
    for seed in _emergency_holds(steer_bias=steer_bias)[:4]:
        for alpha in (0.35, 0.7, 1.0):
            near.append(np.clip((1.0 - alpha) * u_d + alpha * seed, low, high))

    best_u: np.ndarray | None = None
    best_probe: RecoveryProbeResult | None = None
    n_evals = 0

    def _score(u0: np.ndarray) -> tuple[float, float]:
        return (float(np.linalg.norm(u0 - u_d)), abs(float(u0[0] - u_d[0])))

    for u0 in near:
        if n_evals >= int(max_evals):
            break
        n_evals += 1
        s1 = rollout_one_step(s0, u0, car_params_jax, dt, state_history, control_history)
        probe = probe_recovery_from_state(
            s1,
            waypoints,
            car_params_jax,
            dt,
            H,
            lf,
            width,
            margin,
            terminal_stop_speed,
            state_history,
            control_history,
            steer_bias=steer_bias,
            max_evals=4,
        )
        n_evals += probe.n_evals
        if not probe.recoverable:
            continue
        if best_u is None or _score(u0) < _score(best_u):
            best_u = u0.copy()
            best_probe = probe

    return best_u, best_probe, n_evals


def nominal_step_still_recoverable(
    s0: np.ndarray,
    u_d: np.ndarray,
    waypoints: np.ndarray,
    car_params_jax,
    dt: float,
    horizon: int,
    lf: float,
    width: float,
    margin: float,
    terminal_stop_speed: float,
    state_history,
    control_history,
    *,
    max_evals: int = 6,
    recovery_horizon: int | None = None,
    probe_clearance: float = 0.12,
) -> tuple[bool, RecoveryProbeResult]:
    ok, _, _, _ = should_certify_u_d(
        s0,
        u_d,
        waypoints,
        car_params_jax,
        dt,
        horizon,
        lf,
        width,
        margin,
        terminal_stop_speed,
        state_history,
        control_history,
        probe_clearance=probe_clearance,
        max_probe_evals=max_evals,
        recovery_horizon=recovery_horizon,
    )
    return ok, RecoveryProbeResult(
        recoverable=ok,
        u_hold=None,
        trajectory=None,
        min_border_clearance=0.0,
        terminal_speed=0.0,
        n_evals=0,
    )
