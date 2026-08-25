"""Track-relative utilities for the predictive safety filter (Tearle et al.).

Track-relative state (paper Sec. V-A)::

    x_r = [e_lat, mu, v_x, v_y, r]

where ``e_lat`` / ``mu`` are lateral and heading errors w.r.t. the centerline,
and ``v_x``, ``v_y``, ``r`` are body-frame velocities / yaw rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)
from utilities.waypoint_utils import (
    WP_D_LEFT_IDX,
    WP_D_RIGHT_IDX,
    WP_KAPPA_IDX,
    WP_PSI_IDX,
    WP_S_IDX,
    WP_X_IDX,
    WP_Y_IDX,
)

# x_r layout
XR_ELAT = 0
XR_MU = 1
XR_VX = 2
XR_VY = 3
XR_R = 4
XR_DIM = 5


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def probe_track_batch(
    waypoints: np.ndarray,
    xy_points: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized nearest-segment probe for many ``(x, y)`` points.

    Returns arrays of length ``M`` for ``psi_t, e_lat, c, s, d_left, d_right``.
    """
    wps = np.asarray(waypoints, dtype=np.float64)
    pts = np.asarray(xy_points, dtype=np.float64).reshape(-1, 2)
    if wps.ndim != 2 or wps.shape[0] < 2:
        raise ValueError("waypoints must be (N, >=10)")
    m = pts.shape[0]
    if m == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "psi_t": empty,
            "e_lat": empty,
            "c": empty,
            "s": empty,
            "d_left": empty,
            "d_right": empty,
            "seg_idx": empty.astype(np.int64),
            "t": empty,
        }

    xy = wps[:, [WP_X_IDX, WP_Y_IDX]]
    p0 = xy[:-1]
    p1 = xy[1:]
    v = p1 - p0
    seg_len2 = np.einsum("ij,ij->i", v, v) + 1e-12

    # (M, S)
    w = pts[:, None, :] - p0[None, :, :]
    t = np.einsum("msi,si->ms", w, v) / seg_len2[None, :]
    t = np.clip(t, 0.0, 1.0)
    proj = p0[None, :, :] + t[:, :, None] * v[None, :, :]
    diff = pts[:, None, :] - proj
    dist2 = np.einsum("msi,msi->ms", diff, diff)
    seg_idx = np.argmin(dist2, axis=1)
    rows = np.arange(m)
    t_best = t[rows, seg_idx]
    closest = proj[rows, seg_idx]

    v_seg = v[seg_idx]
    seg_len = np.sqrt(seg_len2[seg_idx])
    psi = np.arctan2(v_seg[:, 1], v_seg[:, 0])
    degenerate = seg_len < 1e-6
    if np.any(degenerate):
        psi = psi.copy()
        psi[degenerate] = wps[seg_idx[degenerate], WP_PSI_IDX]

    n_x = -np.sin(psi)
    n_y = np.cos(psi)
    offset = pts - closest
    e_lat = offset[:, 0] * n_x + offset[:, 1] * n_y

    s_i = wps[seg_idx, WP_S_IDX]
    s_ip1 = wps[np.minimum(seg_idx + 1, len(wps) - 1), WP_S_IDX]
    s = s_i + t_best * (s_ip1 - s_i)

    k_i = wps[seg_idx, WP_KAPPA_IDX]
    k_ip1 = wps[np.minimum(seg_idx + 1, len(wps) - 1), WP_KAPPA_IDX]
    c = k_i + t_best * (k_ip1 - k_i)

    d_left = (1.0 - t_best) * wps[seg_idx, WP_D_LEFT_IDX] + t_best * wps[
        np.minimum(seg_idx + 1, len(wps) - 1), WP_D_LEFT_IDX
    ]
    d_right = (1.0 - t_best) * wps[seg_idx, WP_D_RIGHT_IDX] + t_best * wps[
        np.minimum(seg_idx + 1, len(wps) - 1), WP_D_RIGHT_IDX
    ]

    return {
        "psi_t": psi.astype(np.float64),
        "e_lat": e_lat.astype(np.float64),
        "c": c.astype(np.float64),
        "s": s.astype(np.float64),
        "d_left": d_left.astype(np.float64),
        "d_right": d_right.astype(np.float64),
        "seg_idx": seg_idx.astype(np.int64),
        "t": t_best.astype(np.float64),
    }


def probe_track_at_xy(
    waypoints: np.ndarray,
    x: float,
    y: float,
) -> dict[str, float]:
    """Nearest-segment probe of the raceline polyline at (x, y)."""
    batch = probe_track_batch(waypoints, np.array([[x, y]], dtype=np.float64))
    return {
        "xt": float("nan"),  # not needed by callers; kept for API stability
        "yt": float("nan"),
        "psi_t": float(batch["psi_t"][0]),
        "c": float(batch["c"][0]),
        "s": float(batch["s"][0]),
        "e_lat": float(batch["e_lat"][0]),
        "d_left": float(batch["d_left"][0]),
        "d_right": float(batch["d_right"][0]),
        "seg_idx": float(batch["seg_idx"][0]),
        "t": float(batch["t"][0]),
    }


def curvature_at_s(waypoints: np.ndarray, s_query: float) -> float:
    """Interpolate track curvature at arc-length ``s_query`` (wrapped into range)."""
    wps = np.asarray(waypoints, dtype=np.float64)
    s = wps[:, WP_S_IDX]
    kappa = wps[:, WP_KAPPA_IDX]
    s0 = float(s[0])
    s_max = float(s[-1] - s0)
    if s_max < 1e-9:
        return float(kappa[0])
    s_rel = (float(s_query) - s0) % s_max
    s_local = s - s0
    return float(np.interp(s_rel, s_local, kappa))


def probe_track_at_s(waypoints: np.ndarray, s_query: float) -> dict[str, float]:
    """Probe raceline pose / width at arc-length ``s_query``."""
    wps = np.asarray(waypoints, dtype=np.float64)
    s = wps[:, WP_S_IDX]
    s0 = float(s[0])
    s_max = float(s[-1] - s0)
    if s_max < 1e-9:
        i = 0
        t = 0.0
    else:
        s_rel = (float(s_query) - s0) % s_max
        s_local = s - s0
        i = int(np.searchsorted(s_local, s_rel, side="right") - 1)
        i = int(np.clip(i, 0, len(s_local) - 2))
        ds = s_local[i + 1] - s_local[i]
        t = 0.0 if abs(ds) < 1e-12 else float((s_rel - s_local[i]) / ds)

    def _lerp(col: int) -> float:
        return float((1.0 - t) * wps[i, col] + t * wps[i + 1, col])

    psi_a = float(wps[i, WP_PSI_IDX])
    psi_b = float(wps[i + 1, WP_PSI_IDX])
    psi = float(wrap_angle(psi_a + t * wrap_angle(psi_b - psi_a)))

    return {
        "xt": _lerp(WP_X_IDX),
        "yt": _lerp(WP_Y_IDX),
        "psi_t": psi,
        "c": _lerp(WP_KAPPA_IDX),
        "s": float(s0 + (s_local[i] + t * (s_local[i + 1] - s_local[i]))),
        "d_left": _lerp(WP_D_LEFT_IDX),
        "d_right": _lerp(WP_D_RIGHT_IDX),
        "seg_idx": float(i),
        "t": t,
    }


def global_state_to_xr(car_state: np.ndarray, track: dict[str, float]) -> np.ndarray:
    """Convert gym car state + track probe into ``x_r``."""
    s = np.asarray(car_state, dtype=np.float64).reshape(-1)
    mu = float(wrap_angle(s[POSE_THETA_IDX] - track["psi_t"]))
    return np.array(
        [
            float(track["e_lat"]),
            mu,
            float(s[LINEAR_VEL_X_IDX]),
            float(s[LINEAR_VEL_Y_IDX]),
            float(s[ANGULAR_VEL_Z_IDX]),
        ],
        dtype=np.float64,
    )


def car_state_to_xr_from_waypoints(car_state: np.ndarray, waypoints: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    s = np.asarray(car_state, dtype=np.float64).reshape(-1)
    track = probe_track_at_xy(waypoints, float(s[POSE_X_IDX]), float(s[POSE_Y_IDX]))
    # Prefer signed e_lat from probe (already computed).
    xr = global_state_to_xr(s, track)
    xr[XR_ELAT] = track["e_lat"]
    return xr, track


def front_corner_errors(e_lat: float, mu: float, lf: float, width: float) -> tuple[float, float]:
    """Paper eq. (9): front-left / front-right corner lateral errors."""
    half_w = 0.5 * float(width)
    e_lf = float(e_lat + lf * np.sin(mu) + half_w * np.cos(mu))
    e_rf = float(e_lat + lf * np.sin(mu) - half_w * np.cos(mu))
    return e_lf, e_rf


def half_track_widths(d_left: float, d_right: float, margin: float = 0.0) -> tuple[float, float]:
    """Usable half-widths after boundary margin (left positive, right positive)."""
    t_left = max(float(d_left) - float(margin), 1e-3)
    t_right = max(float(d_right) - float(margin), 1e-3)
    return t_left, t_right


def track_border_violations(
    e_lat: float,
    mu: float,
    lf: float,
    width: float,
    d_left: float,
    d_right: float,
    margin: float = 0.0,
) -> dict[str, float]:
    """Signed clearances of front corners; negative means outside the track."""
    e_lf, e_rf = front_corner_errors(e_lat, mu, lf, width)
    t_left, t_right = half_track_widths(d_left, d_right, margin)
    # Left border is +e, right is -e in our Frenet convention.
    clear_lf_left = t_left - e_lf
    clear_lf_right = t_right + e_lf
    clear_rf_left = t_left - e_rf
    clear_rf_right = t_right + e_rf
    min_clear = float(min(clear_lf_left, clear_lf_right, clear_rf_left, clear_rf_right))
    return {
        "e_lf": e_lf,
        "e_rf": e_rf,
        "min_clearance": min_clear,
        "t_left": t_left,
        "t_right": t_right,
    }


def xr_state_polytope(
    t_half: float,
    mu_max: float = 0.5 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Polytopic ``H x_r ≤ h`` on (|e_lat| ≤ t_half, |μ| ≤ mu_max).

    Velocity components are unconstrained here (paper uses only these for X_r
    in the terminal SDP state constraint block).
    """
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    h = np.array([t_half, t_half, mu_max, mu_max], dtype=np.float64)
    return H, h


def input_polytope(
    delta_max: float = 0.4,
    accel_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Polytope ``G u ≤ g`` for ``u = [δ, τ]``."""
    G = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )
    g = np.array([delta_max, delta_max, accel_max, accel_max], dtype=np.float64)
    return G, g


def lookahead_s_ahead(
    u_d_accel: float,
    horizon: int,
    dt: float,
    v_ref: float,
    accel_gain: float = 0.5,
) -> float:
    """Heuristic lookahead distance for terminal curvature (paper Sec. V).

    ``s_ahead ≈ v_ref * t_N + gain * max(u_d,τ, 0) * t_N^2``.
    """
    t_n = float(horizon) * float(dt)
    return float(max(v_ref, 0.0) * t_n + accel_gain * max(float(u_d_accel), 0.0) * t_n * t_n)


def terminal_curvature(
    waypoints: np.ndarray,
    s_now: float,
    u_d_accel: float,
    horizon: int,
    dt: float,
    v_ref: float,
) -> float:
    s_ahead = lookahead_s_ahead(u_d_accel, horizon, dt, v_ref)
    return curvature_at_s(waypoints, s_now + s_ahead)


def current_border_clearance(
    car_state: np.ndarray,
    waypoints: np.ndarray,
    lf: float,
    width: float,
    margin: float = 0.0,
) -> float:
    """Signed front-corner clearance at the current pose (one probe, no rollout)."""
    s = np.asarray(car_state, dtype=np.float64).reshape(-1)
    track = probe_track_at_xy(waypoints, float(s[POSE_X_IDX]), float(s[POSE_Y_IDX]))
    mu = float(wrap_angle(s[POSE_THETA_IDX] - track["psi_t"]))
    info = track_border_violations(
        float(track["e_lat"]),
        mu,
        lf,
        width,
        float(track["d_left"]),
        float(track["d_right"]),
        margin,
    )
    return float(info["min_clearance"])


def trajectory_track_metrics(
    trajectory: np.ndarray,
    waypoints: np.ndarray,
    lf: float,
    width: float,
    margin: float = 0.0,
) -> dict[str, Any]:
    """Evaluate track clearances along a gym-state trajectory."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim == 3:
        traj = traj[0]
    clearances = []
    e_lats = []
    mus = []
    for i in range(len(traj)):
        track = probe_track_at_xy(waypoints, float(traj[i, POSE_X_IDX]), float(traj[i, POSE_Y_IDX]))
        mu = float(wrap_angle(traj[i, POSE_THETA_IDX] - track["psi_t"]))
        info = track_border_violations(
            track["e_lat"], mu, lf, width, track["d_left"], track["d_right"], margin
        )
        clearances.append(info["min_clearance"])
        e_lats.append(track["e_lat"])
        mus.append(mu)
    clearances = np.asarray(clearances, dtype=np.float64)
    return {
        "min_border_clearance": float(np.min(clearances)) if len(clearances) else float("nan"),
        "clearances": clearances,
        "e_lats": np.asarray(e_lats, dtype=np.float64),
        "mus": np.asarray(mus, dtype=np.float64),
        "terminal_e_lat": float(e_lats[-1]) if e_lats else float("nan"),
        "terminal_mu": float(mus[-1]) if mus else float("nan"),
    }


# ---------------------------------------------------------------------------
# Continuous / discrete track-relative dynamics (constant curvature)
# ---------------------------------------------------------------------------

def pacejka_body_derivatives(
    v_x: float,
    v_y: float,
    r: float,
    delta: float,
    accel: float,
    car_params: np.ndarray,
) -> tuple[float, float, float]:
    """Body-frame Pacejka rates matching ``_pacejka_step`` (simplified, no servo).

    ``accel`` is commanded longitudinal acceleration (our control τ).
    Returns ``(v_x_dot, v_y_dot, r_dot)``.
    """
    p = np.asarray(car_params, dtype=np.float64)
    mu_s, lf, lr, h_cg, m, I_z, g_ = p[0:7]
    B_f, C_f, D_f, E_f = p[7:11]
    B_r, C_r, D_r, E_r = p[11:15]
    a_min, a_max = float(p[20]), float(p[21])
    v_min, v_max, v_switch = float(p[22]), float(p[23]), float(p[24])

    v_x_safe = max(float(v_x), 1e-3)
    alpha_f = -np.arctan((v_y + r * lf) / v_x_safe) + delta
    alpha_r = -np.arctan((v_y - r * lr) / v_x_safe)

    # Approximate normal loads (ignore load transfer coupling to accel for steady-state).
    F_zf = m * g_ * lr / (lr + lf)
    F_zr = m * g_ * lf / (lr + lf)

    F_yf = mu_s * F_zf * D_f * np.sin(
        C_f * np.arctan(B_f * alpha_f - E_f * (B_f * alpha_f - np.arctan(B_f * alpha_f)))
    )
    F_yr = mu_s * F_zr * D_r * np.sin(
        C_r * np.arctan(B_r * alpha_r - E_r * (B_r * alpha_r - np.arctan(B_r * alpha_r)))
    )

    # Longitudinal accel with simple clips (no rolling resistance for equilibrium).
    vel = v_x_safe
    pos_limit = a_max * v_switch / vel if vel > v_switch else a_max
    v_x_dot = float(np.clip(accel, a_min, pos_limit))
    max_a = mu_s * g_
    v_x_dot = float(np.clip(v_x_dot, -max_a, max_a))
    if v_x < v_min and v_x_dot < 0.0:
        v_x_dot = 0.0
    if v_x > v_max and v_x_dot > 0.0:
        v_x_dot = 0.0

    v_y_dot = (F_yr + F_yf) / m - v_x * r
    # Match paper/sign convention used in jax model: (-lr * F_yr + lf * F_yf) / I_z
    # Jax F_yf already uses steered alpha; cos(delta) omitted as in jax _pacejka_step.
    r_dot = (-lr * F_yr + lf * F_yf) / I_z
    return float(v_x_dot), float(v_y_dot), float(r_dot)


def track_relative_continuous_f(
    x_r: np.ndarray,
    u: np.ndarray,
    c: float,
    car_params: np.ndarray,
) -> np.ndarray:
    """Continuous ``ẋ_r = f_r(x_r, u; c)`` (paper eqs. 11 + Pacejka body)."""
    e_lat, mu, v_x, v_y, r = [float(v) for v in np.asarray(x_r, dtype=np.float64).reshape(XR_DIM)]
    delta, accel = [float(v) for v in np.asarray(u, dtype=np.float64).reshape(2)]
    c = float(c)

    e_lat_dot = v_x * np.sin(mu) + v_y * np.cos(mu)
    denom = max(1.0 - c * e_lat, 1e-3)
    mu_dot = r - c * (v_x * np.cos(mu) - v_y * np.sin(mu)) / denom
    v_x_dot, v_y_dot, r_dot = pacejka_body_derivatives(v_x, v_y, r, delta, accel, car_params)
    return np.array([e_lat_dot, mu_dot, v_x_dot, v_y_dot, r_dot], dtype=np.float64)


def track_relative_euler_step(
    x_r: np.ndarray,
    u: np.ndarray,
    c: float,
    car_params: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Forward-Euler discretization of track-relative dynamics."""
    x_r = np.asarray(x_r, dtype=np.float64).reshape(XR_DIM)
    f = track_relative_continuous_f(x_r, u, c, car_params)
    x_next = x_r + float(dt) * f
    x_next[XR_MU] = float(wrap_angle(x_next[XR_MU]))
    return x_next
