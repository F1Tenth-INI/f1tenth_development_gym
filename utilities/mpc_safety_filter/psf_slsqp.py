"""Realtime Tearle predictive safety filter solver (paper eqs. 3–5).

Philosophy
----------
1. **Certify-first**: roll ``u_d`` (base controller) unchanged. If the predicted
   path stays inside the track, return ``u_d`` immediately.
2. **Recovery only when needed**: search for a minimally invasive ``u_0`` close
   to ``u_d`` that restores clearance. Do *not* preemptively rewrite throttle
   or impose speed schedules on a healthy command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .psf_track_relative import (
    car_state_to_xr_from_waypoints,
    front_corner_errors,
    half_track_widths,
    probe_track_batch,
    wrap_angle,
)
from .psf_recoverability import (
    find_minimal_recovery_u0,
    should_certify_u_d,
)
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    control_limits_high,
    control_limits_low,
)


@dataclass
class PSFSolveResult:
    u0: np.ndarray
    Q: np.ndarray
    trajectory: np.ndarray
    cost: float
    certified: bool
    success: bool
    message: str
    min_border_clearance: float
    terminal_violation: float
    max_abs_vy: float
    max_abs_wz: float
    nfev: int


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


def _path_terminal_metrics(
    trajectory: np.ndarray,
    waypoints: np.ndarray,
    lf: float,
    width: float,
    margin: float,
    terminal_violation_fn: Callable[[np.ndarray, float], float],
    c_term: float,
) -> dict[str, float]:
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim == 3:
        traj = traj[0]
    pts = traj[:, [POSE_X_IDX, POSE_Y_IDX]]
    batch = probe_track_batch(waypoints, pts)
    mu = wrap_angle(traj[:, POSE_THETA_IDX] - batch["psi_t"])

    clearances = np.empty(len(traj), dtype=np.float64)
    for i in range(len(traj)):
        e_lat_i = float(batch["e_lat"][i])
        mu_i = float(mu[i])
        e_lf, e_rf = front_corner_errors(e_lat_i, mu_i, lf, width)
        t_left, t_right = half_track_widths(
            float(batch["d_left"][i]), float(batch["d_right"][i]), margin
        )
        corner_clear = min(t_left - e_lf, t_right + e_lf, t_left - e_rf, t_right + e_rf)
        center_clear = min(t_left - e_lat_i, t_right + e_lat_i)
        clearances[i] = min(corner_clear, center_clear)

    xr_T, _ = car_state_to_xr_from_waypoints(traj[-1], waypoints)
    term_viol = float(terminal_violation_fn(xr_T, c_term))
    return {
        "min_border_clearance": float(np.min(clearances)),
        "path_violation": float(max(0.0, -np.min(clearances))),
        "terminal_violation": term_viol,
        "terminal_e_lat": float(xr_T[0]),
        "terminal_mu": float(xr_T[1]),
        "max_abs_vy": float(np.max(np.abs(traj[:, LINEAR_VEL_Y_IDX]))),
        "max_abs_wz": float(np.max(np.abs(traj[:, ANGULAR_VEL_Z_IDX]))),
        "terminal_vx": float(abs(traj[-1, LINEAR_VEL_X_IDX])),
    }


def shift_warm_start(Q_prev: np.ndarray | None, horizon: int, u_d: np.ndarray) -> np.ndarray:
    if Q_prev is None or len(Q_prev) == 0:
        return np.tile(np.asarray(u_d, dtype=np.float64).reshape(1, 2), (horizon, 1))
    Q_prev = np.asarray(Q_prev, dtype=np.float64)
    if Q_prev.ndim != 2:
        Q_prev = Q_prev.reshape(-1, 2)
    H = int(horizon)
    if Q_prev.shape[0] >= 2:
        shifted = np.vstack([Q_prev[1:], Q_prev[-1:]])
    else:
        shifted = Q_prev.copy()
    if shifted.shape[0] < H:
        pad = np.tile(shifted[-1:], (H - shifted.shape[0], 1))
        shifted = np.vstack([shifted, pad])
    return shifted[:H].copy()


def _psf_scalar_cost(Q: np.ndarray, u_d: np.ndarray, u_prev: np.ndarray, w_u: float, w_du: float) -> float:
    track = w_u * float(np.sum((Q[0] - u_d) ** 2))
    rate = w_du * float(np.sum((Q[0] - u_prev) ** 2))
    if len(Q) > 1:
        rate += w_du * float(np.sum(np.diff(Q, axis=0) ** 2))
    return track + rate


def _path_constraints_ok(
    metrics: dict[str, float],
    *,
    min_path_clearance: float,
    max_vy: float,
    max_yaw_rate: float,
) -> bool:
    """Layer-1: every rollout state inside the margin-shrunk track corridor."""
    return (
        metrics["min_border_clearance"] >= float(min_path_clearance)
        and metrics["max_abs_vy"] <= max_vy
        and metrics["max_abs_wz"] <= max_yaw_rate
    )


def solve_psf_slsqp(
    *,
    s0: np.ndarray,
    u_d: np.ndarray,
    u_prev: np.ndarray,
    waypoints: np.ndarray,
    car_params_jax,
    car_params_np: np.ndarray,
    dt: float,
    horizon: int,
    w_u: float,
    w_du: float,
    slack_penalty: float,
    lf: float,
    width: float,
    margin: float,
    c_term: float,
    terminal_violation_fn: Callable[[np.ndarray, float], float],
    Q_init: np.ndarray | None,
    state_history=None,
    control_history=None,
    maxiter: int = 40,
    certify_eps: float = 1e-3,
    max_vy: float = 5.0,
    max_yaw_rate: float = 14.0,
    max_accel: float | None = None,
    terminal_tol: float = 0.5,
    recovery_terminal_tol: float = 0.05,
    w_terminal: float = 30.0,
    max_recovery_evals: int = 14,
    equilibrium_u: np.ndarray | None = None,
    recovery_hold_steps: int = 3,
    num_blend_steps: int = 6,
    e_lat: float = 0.0,
    mu: float = 0.0,
    min_path_clearance: float = 0.0,
    track_s: float | None = None,
    terminal_stop_speed: float = 0.5,
    recovery_probe_evals: int = 6,
    probe_clearance: float = 0.12,
    recovery_horizon: int | None = None,
) -> PSFSolveResult:
    """Certify ``u_d`` or compute a minimally invasive safe ``u_0``."""
    import jax.numpy as jnp

    del car_params_np, track_s, max_accel
    H = int(horizon)
    u_d = np.asarray(u_d, dtype=np.float64).reshape(2)
    u_prev = np.asarray(u_prev, dtype=np.float64).reshape(2)
    s0 = np.asarray(s0, dtype=np.float64).reshape(-1)

    if state_history is None:
        state_history = jnp.zeros((10, 10), dtype=jnp.float32)
    if control_history is None:
        control_history = jnp.zeros((10, 2), dtype=jnp.float32)

    low = np.asarray(control_limits_low, dtype=np.float64)
    high = np.asarray(control_limits_high, dtype=np.float64)

    nfev = 0
    vy0 = float(s0[LINEAR_VEL_Y_IDX])
    wz0 = float(s0[ANGULAR_VEL_Z_IDX])

    now = probe_track_batch(waypoints, s0[None, [POSE_X_IDX, POSE_Y_IDX]])
    mu_now = float(wrap_angle(s0[POSE_THETA_IDX] - now["psi_t"][0]))
    e_lf, e_rf = front_corner_errors(float(now["e_lat"][0]), mu_now, lf, width)
    t_l, t_r = half_track_widths(float(now["d_left"][0]), float(now["d_right"][0]), margin)
    corner_clear0 = float(min(t_l - e_lf, t_r + e_lf, t_l - e_rf, t_r + e_rf))
    center_clear0 = float(min(t_l - float(now["e_lat"][0]), t_r + float(now["e_lat"][0])))
    clearance0 = min(corner_clear0, center_clear0)

    # Only treat as emergency when already leaving the track / spinning hard.
    emergency = clearance0 < 0.0 or abs(vy0) > max_vy or abs(wz0) > max_yaw_rate

    def eval_Q(Q: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        nonlocal nfev
        nfev += 1
        traj = _rollout_pacejka(s0, Q, car_params_jax, dt, state_history, control_history)
        metrics = _path_terminal_metrics(
            traj, waypoints, lf, width, margin, terminal_violation_fn, c_term
        )
        return traj, metrics

    u_eq = None
    if equilibrium_u is not None:
        u_eq = np.clip(np.asarray(equilibrium_u, dtype=np.float64).reshape(2), low, high)

    def make_Q(u0: np.ndarray, *, recovery: bool = False) -> np.ndarray:
        """Hold ``u0`` for the horizon.

        Previously faded recovery toward raceline ``u_eq``, which re-applied
        throttle near walls and caused failed recoveries.  For a safety filter
        the first applied input matters most; keep a constant hold.
        """
        del recovery
        u0 = np.clip(np.asarray(u0, dtype=np.float64).reshape(2), low, high)
        return np.tile(u0.reshape(1, 2), (H, 1))

    # ---- 1) Classic Tearle certify on a SHORT hold of u_d ----
    # Full-horizon constant hold is too strict for feedback controllers (PP).
    H_cert = int(np.clip(round(0.80 / max(dt, 1e-3)), 15, min(22, H)))
    Q_try = np.tile(np.clip(u_d, low, high).reshape(1, 2), (H_cert, 1))
    traj, metrics = eval_Q(Q_try)
    if (
        not emergency
        and _path_constraints_ok(
            metrics, min_path_clearance=min_path_clearance, max_vy=max_vy, max_yaw_rate=max_yaw_rate
        )
    ):
        Q_full = make_Q(u_d, recovery=False)
        return PSFSolveResult(
            u0=u_d.copy(),
            Q=Q_full,
            trajectory=traj,
            cost=0.0,
            certified=True,
            success=True,
            message="certified",
            min_border_clearance=metrics["min_border_clearance"],
            terminal_violation=metrics["terminal_violation"],
            max_abs_vy=metrics["max_abs_vy"],
            max_abs_wz=metrics["max_abs_wz"],
            nfev=nfev,
        )

    # ---- 2) Recovery: emergency brake / steer grid (no throttle) ----
    near_right = float(now["d_right"][0]) < float(now["d_left"][0])
    # Centerline steer + hard brake; allow stronger steer when near border.
    delta_c = float(np.clip(-1.5 * e_lat - 0.8 * mu, low[0], high[0]))
    accel_c = float(low[1])
    u_brake = np.array([delta_c, float(low[1])], dtype=np.float64)
    open_steer = float(high[0] if near_right else low[0])
    u_center = np.array([delta_c, float(low[1])], dtype=np.float64)

    best: PSFSolveResult | None = None
    recovery_evals = 0
    stop_search = False
    u_grid_seed = None

    def consider(u0: np.ndarray, label: str, *, blend_alpha: float = 1.0) -> bool:
        """Evaluate a recovery candidate. Return True to stop searching."""
        nonlocal best, recovery_evals, stop_search
        if stop_search or recovery_evals >= int(max_recovery_evals):
            return True
        recovery_evals += 1
        u0 = np.clip(np.asarray(u0, dtype=np.float64).reshape(2), low, high)
        Q = make_Q(u0, recovery=True)
        traj_i, met = eval_Q(Q)
        path_ok = _path_constraints_ok(
            met, min_path_clearance=min_path_clearance, max_vy=max_vy, max_yaw_rate=max_yaw_rate
        )
        term_viol = float(met["terminal_violation"])
        # Lexicographic: path feasible first, then prefer braking, then small Δu.
        clearance = float(met["min_border_clearance"])
        J = 0.0
        if not path_ok:
            J += slack_penalty * (
                25.0 * max(0.0, min_path_clearance - clearance) ** 2
                + 25.0 * max(0.0, -clearance) ** 2
                + max(0.0, term_viol - recovery_terminal_tol) ** 2
                + 0.1 * max(0.0, met["max_abs_vy"] - max_vy) ** 2
            )
        else:
            # Feasible recoveries: brake hard, change steering only as needed.
            J += 5.0 * max(0.0, float(u0[1])) ** 2
            J += 0.05 * float(u0[1] - low[1]) ** 2  # prefer stronger brake
            J += w_u * float((u0[0] - u_d[0]) ** 2)
            J += 0.05 * w_u * float((u0[1] - u_d[1]) ** 2)
            J += w_terminal * float(term_viol**2)

        cand = PSFSolveResult(
            u0=u0,
            Q=Q,
            trajectory=traj_i,
            cost=J,
            certified=False,
            success=path_ok,
            message=label,
            min_border_clearance=met["min_border_clearance"],
            terminal_violation=term_viol,
            max_abs_vy=met["max_abs_vy"],
            max_abs_wz=met["max_abs_wz"],
            nfev=nfev,
        )
        if best is None:
            best = cand
        elif path_ok and not best.success:
            best = cand
        elif path_ok == best.success and J < best.cost:
            best = cand

        if path_ok:
            stop_search = True
            return True
        return recovery_evals >= int(max_recovery_evals)

    if u_grid_seed is not None:
        consider(u_grid_seed, "grid_seed")
    # Try emergency brake seeds before blending toward u_d.
    for seed in (
        u_brake,
        np.array([open_steer, float(low[1])], dtype=np.float64),
        u_center,
    ):
        if consider(seed, "emergency", blend_alpha=1.0):
            break
    if not stop_search and (best is None or not best.success):
        for seed in (u_center, u_brake, np.array([open_steer, float(low[1])], dtype=np.float64)):
            for a in np.linspace(0.0, 1.0, max(2, int(num_blend_steps) + 1)):
                if consider(
                    (1.0 - float(a)) * u_d + float(a) * seed,
                    "blend",
                    blend_alpha=float(a),
                ):
                    break
            if stop_search or (best is not None and best.success):
                break

    if not stop_search and (best is None or not best.success):
        # Sparse recovery grid, still ranked by proximity to u_d.
        steers = np.unique(
            np.clip(
                np.array([float(u_d[0]), delta_c, open_steer, 0.0], dtype=np.float64),
                low[0],
                high[0],
            )
        )
        accels = np.unique(
            np.clip(
                np.array(
                    [float(u_d[1]), 0.0, -2.0, -5.0, float(low[1]), accel_c],
                    dtype=np.float64,
                ),
                low[1],
                high[1],
            )
        )
        for steer in steers:
            for accel in accels:
                if consider(np.array([steer, accel], dtype=np.float64), "grid"):
                    break
            if stop_search or (best is not None and best.success):
                break

    if not stop_search and (best is None or not best.success) and maxiter > 0:
        u0_init = u_d.copy() if best is None else best.u0.copy()

        def objective(z: np.ndarray) -> float:
            Q = make_Q(z, recovery=True)
            _, met = eval_Q(Q)
            J = _psf_scalar_cost(Q, u_d, u_prev, w_u, w_du)
            J += w_terminal * float(met["terminal_violation"] ** 2)
            clearance = float(met["min_border_clearance"])
            J += slack_penalty * (
                25.0 * max(0.0, min_path_clearance - clearance) ** 2
                + 25.0 * max(0.0, -clearance) ** 2
                + max(0.0, met["terminal_violation"] - recovery_terminal_tol) ** 2
            )
            return J

        try:
            res = minimize(
                objective,
                u0_init,
                method="SLSQP",
                bounds=[
                    (float(low[0]), float(high[0])),
                    (float(low[1]), float(high[1])),
                ],
                options={"maxiter": int(max(1, maxiter)), "ftol": 1e-4, "disp": False},
            )
            consider(np.asarray(res.x, dtype=np.float64), "slsqp_u0")
        except Exception:
            pass

    if best is None:
        Q = make_Q(u_brake, recovery=True)
        traj, metrics = eval_Q(Q)
        best = PSFSolveResult(
            u0=u_brake,
            Q=Q,
            trajectory=traj,
            cost=float("nan"),
            certified=False,
            success=False,
            message="fallback_brake",
            min_border_clearance=metrics["min_border_clearance"],
            terminal_violation=metrics["terminal_violation"],
            max_abs_vy=metrics["max_abs_vy"],
            max_abs_wz=metrics["max_abs_wz"],
            nfev=nfev,
        )
    elif not best.success:
        # Never apply a failed near-u_d blend that still leaves the track.
        Q = make_Q(u_brake, recovery=True)
        traj, metrics = eval_Q(Q)
        best = PSFSolveResult(
            u0=u_brake.copy(),
            Q=Q,
            trajectory=traj,
            cost=_psf_scalar_cost(Q, u_d, u_prev, w_u, w_du),
            certified=False,
            success=_path_constraints_ok(
                metrics, min_path_clearance=min_path_clearance, max_vy=max_vy, max_yaw_rate=max_yaw_rate
            ),
            message="fallback_brake",
            min_border_clearance=metrics["min_border_clearance"],
            terminal_violation=metrics["terminal_violation"],
            max_abs_vy=metrics["max_abs_vy"],
            max_abs_wz=metrics["max_abs_wz"],
            nfev=nfev,
        )

    best.nfev = nfev
    best.certified = bool(
        np.linalg.norm(best.u0 - u_d) <= certify_eps and best.success
    )
    if best.certified:
        best.message = "certified"
        best.u0 = u_d.copy()
    return best
