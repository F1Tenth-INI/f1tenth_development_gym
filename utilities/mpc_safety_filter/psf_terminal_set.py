"""Terminal safe sets for the predictive safety filter (Tearle et al. Sec. V).

Two modes
---------
1. ``steady_state`` — conservative: terminal state must stay near the curvature-
   parameterised raceline equilibrium ``x_e(c)``.
2. ``fine_ellipsoid`` — paper SDP (18) + nonlinear verification (19) yielding
   an ellipsoidal invariant set ``{x̄ | x̄ᵀ P x̄ ≤ 1}`` with feedback ``K``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares

from .psf_track_relative import (
    XR_DIM,
    XR_ELAT,
    XR_MU,
    XR_R,
    XR_VX,
    XR_VY,
    input_polytope,
    track_relative_continuous_f,
    track_relative_euler_step,
    xr_state_polytope,
)

SfMode = Literal["steady_state", "fine_ellipsoid"]


@dataclass
class SteadyStateEquilibrium:
    c: float
    x_e: np.ndarray  # (5,)
    u_e: np.ndarray  # (2,)
    residual: float


@dataclass
class FineSafeEllipsoid:
    P: np.ndarray  # (5, 5)
    K: np.ndarray  # (2, 5)
    c_grid: np.ndarray
    x_e_grid: np.ndarray  # (n_c, 5)
    u_e_grid: np.ndarray  # (n_c, 2)
    v_x: float
    t_half: float
    verified_objective: float
    scale: float


def _equilibrium_guess(c: float, v_x: float) -> np.ndarray:
    """Rough kinematic guess: [μ, v_y, r, δ, τ]."""
    # For small slip: r ≈ c * v_x, μ ≈ 0, δ ≈ atan((lf+lr)*c) ~ c * L for small c
    r = float(c) * float(v_x)
    mu = 0.0
    vy = 0.0
    delta = float(np.clip(1.5 * c, -0.35, 0.35))
    tau = 0.0
    return np.array([mu, vy, r, delta, tau], dtype=np.float64)


def find_steady_state(
    c: float,
    v_x: float,
    car_params: np.ndarray,
    *,
    max_nfev: int = 200,
) -> SteadyStateEquilibrium:
    """Solve ``f_r(x_e, u_e; c) = 0`` with ``e_lat=0`` and fixed ``v_x``.

    Free variables: ``[μ, v_y, r, δ, τ]``.
    """
    c = float(c)
    v_x = float(v_x)

    def residual(z: np.ndarray) -> np.ndarray:
        mu, vy, r, delta, tau = z
        x_r = np.array([0.0, mu, v_x, vy, r], dtype=np.float64)
        u = np.array([delta, tau], dtype=np.float64)
        return track_relative_continuous_f(x_r, u, c, car_params)

    best = None
    for scale in (1.0, 0.6, 1.4, 0.3):
        z0 = _equilibrium_guess(c, v_x)
        z0[2] *= scale
        z0[3] = float(np.clip(z0[3] * scale, -0.4, 0.4))
        bounds = (
            np.array([-0.5 * np.pi, -4.0, -12.0, -0.4, -10.0]),
            np.array([0.5 * np.pi, 4.0, 12.0, 0.4, 10.0]),
        )
        sol = least_squares(
            residual, z0, bounds=bounds, max_nfev=max_nfev, ftol=1e-12, xtol=1e-12
        )
        mu, vy, r, delta, tau = sol.x
        eq = SteadyStateEquilibrium(
            c=c,
            x_e=np.array([0.0, mu, v_x, vy, r], dtype=np.float64),
            u_e=np.array([delta, tau], dtype=np.float64),
            residual=float(np.linalg.norm(sol.fun)),
        )
        if best is None or eq.residual < best.residual:
            best = eq
        if eq.residual < 1e-8:
            break
    assert best is not None
    return best


def find_feasible_vx_for_curvature_range(
    c_max: float,
    v_x_target: float,
    car_params: np.ndarray,
    *,
    n_check: int = 5,
    residual_tol: float = 1e-4,
) -> float:
    """Reduce ``v_x`` until equilibria exist across ``[-c_max, c_max]``."""
    for v_try in np.linspace(v_x_target, max(0.8, 0.35 * v_x_target), 8):
        ok = True
        for c in np.linspace(-c_max, c_max, n_check):
            eq = find_steady_state(float(c), float(v_try), car_params)
            if eq.residual > residual_tol:
                ok = False
                break
        if ok:
            return float(v_try)
    return float(max(0.8, 0.35 * v_x_target))


def find_steady_state_grid(
    c_grid: np.ndarray,
    v_x: float,
    car_params: np.ndarray,
) -> list[SteadyStateEquilibrium]:
    return [find_steady_state(float(c), v_x, car_params) for c in np.asarray(c_grid, dtype=np.float64)]


def linearize_track_relative(
    x_e: np.ndarray,
    u_e: np.ndarray,
    c: float,
    car_params: np.ndarray,
    dt: float,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time linearization ``x̄⁺ = A x̄ + B ū`` via finite differences."""
    x_e = np.asarray(x_e, dtype=np.float64).reshape(XR_DIM)
    u_e = np.asarray(u_e, dtype=np.float64).reshape(2)
    f0 = track_relative_euler_step(x_e, u_e, c, car_params, dt)

    A = np.zeros((XR_DIM, XR_DIM), dtype=np.float64)
    for i in range(XR_DIM):
        dx = np.zeros(XR_DIM, dtype=np.float64)
        dx[i] = eps
        fp = track_relative_euler_step(x_e + dx, u_e, c, car_params, dt)
        fm = track_relative_euler_step(x_e - dx, u_e, c, car_params, dt)
        A[:, i] = (fp - fm) / (2.0 * eps)

    B = np.zeros((XR_DIM, 2), dtype=np.float64)
    for j in range(2):
        du = np.zeros(2, dtype=np.float64)
        du[j] = eps
        fp = track_relative_euler_step(x_e, u_e + du, c, car_params, dt)
        fm = track_relative_euler_step(x_e, u_e - du, c, car_params, dt)
        B[:, j] = (fp - fm) / (2.0 * eps)

    # Shift so deviation dynamics are around the fixed point (f0 ≈ x_e).
    # Euler map is already absolute; A,B above are correct Jacobians of f_d.
    _ = f0
    return A, B


def _solve_terminal_sdp(
    A_list: list[np.ndarray],
    B_list: list[np.ndarray],
    x_e_list: list[np.ndarray],
    u_e_list: list[np.ndarray],
    H: np.ndarray,
    h: np.ndarray,
    G: np.ndarray,
    g: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper SDP (18): maximal-volume ellipsoid with common K for all curvatures."""
    import cvxpy as cp

    n = XR_DIM
    m = 2
    E = cp.Variable((n, n), PSD=True)
    Y = cp.Variable((m, n))

    constraints = [E >> 1e-6 * np.eye(n)]

    Q_sqrt = np.linalg.cholesky(Q + 1e-12 * np.eye(n))
    R_sqrt = np.linalg.cholesky(R + 1e-12 * np.eye(m))
    Z_nn = np.zeros((n, n))
    Z_nm = np.zeros((n, m))
    Z_mn = np.zeros((m, n))
    I_n = np.eye(n)
    I_m = np.eye(m)

    for x_e, u_e, A, B in zip(x_e_list, u_e_list, A_list, B_list):
        # State constraints around each equilibrium (18c)
        for j in range(H.shape[0]):
            Hj = H[j : j + 1, :]
            margin = float(h[j] - Hj @ x_e)
            if margin <= 1e-6:
                continue
            constraints.append(
                cp.bmat(
                    [
                        [np.array([[margin**2]]), Hj @ E],
                        [E @ Hj.T, E],
                    ]
                )
                >> 0
            )
        # Input constraints (18d)
        for l in range(G.shape[0]):
            Gl = G[l : l + 1, :]
            margin = float(g[l] - Gl @ u_e)
            if margin <= 1e-6:
                continue
            # [[m^2, G Y]; [Y^T G^T, E]]
            constraints.append(
                cp.bmat(
                    [
                        [np.array([[margin**2]]), Gl @ Y],
                        [Y.T @ Gl.T, E],
                    ]
                )
                >> 0
            )
        # Lyapunov LMI (18e): size (3n+m)
        AE_BY = A @ E + B @ Y
        constraints.append(
            cp.bmat(
                [
                    [E, AE_BY.T, E @ Q_sqrt.T, Y.T @ R_sqrt.T],
                    [AE_BY, E, Z_nn, Z_nm],
                    [Q_sqrt @ E, Z_nn, I_n, Z_nm],
                    [R_sqrt @ Y, Z_mn, Z_mn, I_m],
                ]
            )
            >> 0
        )

    prob = cp.Problem(cp.Maximize(cp.log_det(E)), constraints)
    solved = False
    for solver_name, kwargs in (
        ("SCS", dict(max_iters=10000, eps=1e-4)),
        ("CLARABEL", {}),
        ("MOSEK", {}),
    ):
        try:
            solver = getattr(cp, solver_name)
            prob.solve(solver=solver, verbose=False, **kwargs)
            if E.value is not None and Y.value is not None and np.all(np.isfinite(E.value)):
                solved = True
                break
        except Exception:
            continue
    if not solved or E.value is None or Y.value is None:
        raise RuntimeError(f"Terminal SDP failed (status={prob.status})")

    E_val = 0.5 * (E.value + E.value.T)
    # Reject clearly broken SDP solutions
    evals_E = np.linalg.eigvalsh(E_val)
    if np.min(evals_E) < 1e-10 or np.max(evals_E) / max(np.min(evals_E), 1e-16) > 1e10:
        raise RuntimeError("Terminal SDP produced ill-conditioned E")
    Y_val = Y.value
    P = np.linalg.inv(E_val)
    P = 0.5 * (P + P.T)
    K = Y_val @ np.linalg.inv(E_val)
    return P.astype(np.float64), K.astype(np.float64)


def _terminal_control(x_bar: np.ndarray, K: np.ndarray, u_e: np.ndarray) -> np.ndarray:
    u = u_e + K @ x_bar
    return np.clip(u, [-0.4, -10.0], [0.4, 10.0])


def verify_ellipsoid_invariance(
    P: np.ndarray,
    K: np.ndarray,
    c_min: float,
    c_max: float,
    v_x: float,
    car_params: np.ndarray,
    dt: float,
    *,
    n_restarts: int = 200,
    x_e_grid: np.ndarray | None = None,
    u_e_grid: np.ndarray | None = None,
    c_grid: np.ndarray | None = None,
) -> float:
    """Sample-based check of paper (19): worst next Lyapunov value on the ellipsoid.

    Draws random states in ``{x̄ᵀ P x̄ ≤ 1}`` and curvatures in ``[c_min, c_max]``,
    applies ``κ_f``, and returns ``max x̄⁺ᵀ P x̄⁺``. Values ≤ 1 certify approximate
    invariance (same strategy as the paper's multistart experiments).
    """
    P = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    K = np.asarray(K, dtype=np.float64)
    evals = np.linalg.eigvalsh(P)
    if np.min(evals) <= 1e-10 or not np.all(np.isfinite(evals)):
        return float("inf")

    evals, evecs = np.linalg.eigh(P)
    L_inv = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12)))

    def _eq_at(c: float) -> tuple[np.ndarray, np.ndarray]:
        if x_e_grid is not None and u_e_grid is not None and c_grid is not None:
            return interpolate_equilibrium(c, c_grid, x_e_grid, u_e_grid)
        eq = find_steady_state(c, v_x, car_params, max_nfev=60)
        return eq.x_e, eq.u_e

    best = 0.0
    rng = np.random.default_rng(0)
    for _ in range(int(n_restarts)):
        z = rng.normal(size=XR_DIM)
        z /= np.linalg.norm(z) + 1e-12
        radius = float(rng.uniform(0.0, 1.0)) ** (1.0 / XR_DIM)  # fill ellipsoid volume
        x_bar = radius * (L_inv @ z)
        c = float(rng.uniform(c_min, c_max))
        x_e, u_e = _eq_at(c)
        u = _terminal_control(x_bar, K, u_e)
        x_next = track_relative_euler_step(x_e + x_bar, u, c, car_params, dt)
        x_bar_next = x_next - x_e
        x_bar_next[XR_MU] = (x_bar_next[XR_MU] + np.pi) % (2 * np.pi) - np.pi
        if not np.all(np.isfinite(x_bar_next)):
            return float("inf")
        val = float(x_bar_next @ P @ x_bar_next)
        if val > best:
            best = val
    return float(best)


def _lqr_terminal_ellipsoid(
    A_list: list[np.ndarray],
    B_list: list[np.ndarray],
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: discrete LQR at mid curvature + averaged Lyapunov matrix."""
    from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

    mid = len(A_list) // 2
    A, B = A_list[mid], B_list[mid]
    try:
        P_lqr = solve_discrete_are(A, B, Q, R)
        K = -np.linalg.solve(R + B.T @ P_lqr @ B, B.T @ P_lqr @ A)
    except Exception:
        K = np.zeros((2, XR_DIM), dtype=np.float64)
        K[0, XR_ELAT] = -1.5
        K[0, XR_MU] = -2.0
        K[0, XR_R] = -0.4
        K[1, XR_VX] = -0.8
        P_lqr = np.eye(XR_DIM)

    P_sum = np.zeros((XR_DIM, XR_DIM), dtype=np.float64)
    for A_i, B_i in zip(A_list, B_list):
        A_cl = A_i + B_i @ K
        try:
            rho = float(np.max(np.abs(np.linalg.eigvals(A_cl))))
            if rho < 0.999:
                P_i = solve_discrete_lyapunov(A_cl.T, np.eye(XR_DIM))
                P_sum += P_i
            else:
                P_sum += P_lqr
        except Exception:
            P_sum += np.eye(XR_DIM)
    P = P_sum / max(len(A_list), 1)
    P = 0.5 * (P + P.T)
    x_typ = np.array([0.1, 0.1, 0.3, 0.15, 0.3], dtype=np.float64)
    v_typ = float(x_typ @ P @ x_typ)
    if v_typ > 1e-9:
        P = P / v_typ
    return P.astype(np.float64), K.astype(np.float64)


def compute_fine_safe_ellipsoid(
    car_params: np.ndarray,
    *,
    v_x: float,
    c_max: float,
    n_curvature: int,
    dt: float,
    t_half: float,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    verify_restarts: int = 40,
    max_shrink_iters: int = 12,
) -> FineSafeEllipsoid:
    """Offline pipeline: equilibria → SDP → verify/shrink."""
    v_x_use = find_feasible_vx_for_curvature_range(c_max, v_x, car_params)
    if abs(v_x_use - v_x) > 1e-3:
        print(f"[FineSafeEllipsoid] reducing v_x {v_x:.2f} → {v_x_use:.2f} for c_max={c_max}")

    c_grid = np.linspace(-float(c_max), float(c_max), int(n_curvature))
    eqs = find_steady_state_grid(c_grid, v_x_use, car_params)
    for eq in eqs:
        if eq.residual > 1e-4:
            print(f"[FineSafeEllipsoid] warning: residual {eq.residual:.3e} at c={eq.c:.3f}")

    A_list, B_list, x_e_list, u_e_list = [], [], [], []
    for eq in eqs:
        A, B = linearize_track_relative(eq.x_e, eq.u_e, eq.c, car_params, dt)
        A_list.append(A)
        B_list.append(B)
        x_e_list.append(eq.x_e)
        u_e_list.append(eq.u_e)

    H, h = xr_state_polytope(t_half=t_half)
    G, g = input_polytope()
    if Q is None:
        Q = np.diag([10.0, 5.0, 0.1, 0.5, 0.5])
    if R is None:
        R = np.diag([1.0, 0.1])

    # Try SDP; fall back to discrete LQR ellipsoid if it fails / is ill-conditioned.
    try:
        P, K = _solve_terminal_sdp(A_list, B_list, x_e_list, u_e_list, H, h, G, g, Q, R)
        print("[FineSafeEllipsoid] SDP succeeded")
    except Exception as exc:
        print(f"[FineSafeEllipsoid] SDP failed ({exc}); using LQR fallback")
        P, K = _lqr_terminal_ellipsoid(A_list, B_list, Q, R)

    scale = 1.0
    x_e_mat = np.stack(x_e_list, axis=0)
    u_e_mat = np.stack(u_e_list, axis=0)
    verified = verify_ellipsoid_invariance(
        P,
        K,
        -float(c_max),
        float(c_max),
        v_x_use,
        car_params,
        dt,
        n_restarts=verify_restarts,
        x_e_grid=x_e_mat,
        u_e_grid=u_e_mat,
        c_grid=c_grid,
    )
    print(f"[FineSafeEllipsoid] initial verified={verified:.4f}")

    best = (verified, P.copy(), K.copy(), scale)
    for _ in range(max_shrink_iters):
        if verified <= 1.0 + 1e-3:
            break
        if not np.isfinite(verified):
            P, K = _lqr_terminal_ellipsoid(A_list, B_list, Q, R)
            P = P * 4.0
            scale = 0.25
            verified = verify_ellipsoid_invariance(
                P, K, -float(c_max), float(c_max), v_x_use, car_params, dt,
                n_restarts=verify_restarts, x_e_grid=x_e_mat, u_e_grid=u_e_mat, c_grid=c_grid,
            )
            if verified < best[0]:
                best = (verified, P.copy(), K.copy(), scale)
            continue
        beta = float(min(max(verified * 1.02, 1.05), 2.0))
        P = P * beta
        scale /= beta
        verified = verify_ellipsoid_invariance(
            P,
            K,
            -float(c_max),
            float(c_max),
            v_x_use,
            car_params,
            dt,
            n_restarts=verify_restarts,
            x_e_grid=x_e_mat,
            u_e_grid=u_e_mat,
            c_grid=c_grid,
        )
        if np.isfinite(verified) and verified < best[0]:
            best = (verified, P.copy(), K.copy(), scale)

    verified, P, K, scale = best
    print(f"[FineSafeEllipsoid] best verified={verified:.4f} scale={scale:.4e}")
    if not np.isfinite(verified):
        P, K = _lqr_terminal_ellipsoid(A_list, B_list, Q, R)
        scale = 1.0
        verified = verify_ellipsoid_invariance(
            P, K, -float(c_max), float(c_max), v_x_use, car_params, dt,
            n_restarts=verify_restarts, x_e_grid=x_e_mat, u_e_grid=u_e_mat, c_grid=c_grid,
        )
        print(f"[FineSafeEllipsoid] LQR emergency verified={verified:.4f}")
    # Online PSF softens S_f anyway (paper Sec. VI-A); keep best practicable set.

    return FineSafeEllipsoid(
        P=P,
        K=K,
        c_grid=c_grid.astype(np.float64),
        x_e_grid=np.stack(x_e_list, axis=0),
        u_e_grid=np.stack(u_e_list, axis=0),
        v_x=float(v_x_use),
        t_half=float(t_half),
        verified_objective=float(verified),
        scale=float(scale),
    )


def save_fine_safe_ellipsoid(path: str, ell: FineSafeEllipsoid) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        P=ell.P,
        K=ell.K,
        c_grid=ell.c_grid,
        x_e_grid=ell.x_e_grid,
        u_e_grid=ell.u_e_grid,
        v_x=np.array([ell.v_x]),
        t_half=np.array([ell.t_half]),
        verified_objective=np.array([ell.verified_objective]),
        scale=np.array([ell.scale]),
    )


def load_fine_safe_ellipsoid(path: str) -> FineSafeEllipsoid:
    data = np.load(path)
    return FineSafeEllipsoid(
        P=np.asarray(data["P"], dtype=np.float64),
        K=np.asarray(data["K"], dtype=np.float64),
        c_grid=np.asarray(data["c_grid"], dtype=np.float64),
        x_e_grid=np.asarray(data["x_e_grid"], dtype=np.float64),
        u_e_grid=np.asarray(data["u_e_grid"], dtype=np.float64),
        v_x=float(np.asarray(data["v_x"]).reshape(-1)[0]),
        t_half=float(np.asarray(data["t_half"]).reshape(-1)[0]),
        verified_objective=float(np.asarray(data["verified_objective"]).reshape(-1)[0]),
        scale=float(np.asarray(data["scale"]).reshape(-1)[0]),
    )


def interpolate_equilibrium(
    c: float,
    c_grid: np.ndarray,
    x_e_grid: np.ndarray,
    u_e_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of equilibria over the curvature grid."""
    c = float(np.clip(c, float(c_grid[0]), float(c_grid[-1])))
    x_e = np.array([np.interp(c, c_grid, x_e_grid[:, i]) for i in range(XR_DIM)], dtype=np.float64)
    u_e = np.array([np.interp(c, c_grid, u_e_grid[:, i]) for i in range(2)], dtype=np.float64)
    return x_e, u_e


class TerminalSafeSet:
    """Online terminal constraint evaluator for PSF."""

    def __init__(
        self,
        mode: SfMode,
        car_params: np.ndarray,
        *,
        v_x: float = 3.0,
        c_max: float = 2.5,
        n_curvature: int = 21,
        dt: float = 0.04,
        t_half: float = 0.35,
        eps_ss: float = 0.15,
        ellipsoid_path: str | None = None,
        recompute_ellipsoid: bool = False,
        verify_restarts: int = 25,
    ):
        self.mode = mode
        self.car_params = np.asarray(car_params, dtype=np.float64)
        self.v_x = float(v_x)
        self.c_max = float(c_max)
        self.n_curvature = int(n_curvature)
        self.dt = float(dt)
        self.t_half = float(t_half)
        self.eps_ss = float(eps_ss)
        self.ellipsoid: FineSafeEllipsoid | None = None
        self._eq_cache: dict[float, SteadyStateEquilibrium] = {}

        if mode == "fine_ellipsoid":
            self._ensure_ellipsoid(ellipsoid_path, recompute_ellipsoid, verify_restarts)
        else:
            v_x_use = find_feasible_vx_for_curvature_range(self.c_max, self.v_x, self.car_params)
            if abs(v_x_use - self.v_x) > 1e-3:
                print(f"[TerminalSafeSet] reducing v_x {self.v_x:.2f} → {v_x_use:.2f} for c_max={self.c_max}")
            self.v_x = v_x_use
            c_grid = np.linspace(-self.c_max, self.c_max, self.n_curvature)
            eqs = find_steady_state_grid(c_grid, self.v_x, self.car_params)
            self._ss_c_grid = c_grid
            self._ss_x_grid = np.stack([e.x_e for e in eqs], axis=0)
            self._ss_u_grid = np.stack([e.u_e for e in eqs], axis=0)

    def _ensure_ellipsoid(
        self,
        path: str | None,
        recompute: bool,
        verify_restarts: int,
    ) -> None:
        if path and (not recompute) and os.path.isfile(path):
            self.ellipsoid = load_fine_safe_ellipsoid(path)
            self.v_x = float(self.ellipsoid.v_x)
            print(
                f"[TerminalSafeSet] Loaded FineSafeEllipsoid from {path} "
                f"(verified={self.ellipsoid.verified_objective:.4f}, scale={self.ellipsoid.scale:.4f})"
            )
            return
        print("[TerminalSafeSet] Computing FineSafeEllipsoid offline (SDP + verify) ...")
        self.ellipsoid = compute_fine_safe_ellipsoid(
            self.car_params,
            v_x=self.v_x,
            c_max=self.c_max,
            n_curvature=self.n_curvature,
            dt=self.dt,
            t_half=self.t_half,
            verify_restarts=verify_restarts,
        )
        self.v_x = float(self.ellipsoid.v_x)
        print(
            f"[TerminalSafeSet] Ellipsoid ready "
            f"(verified={self.ellipsoid.verified_objective:.4f}, scale={self.ellipsoid.scale:.4f})"
        )
        if path:
            save_fine_safe_ellipsoid(path, self.ellipsoid)
            print(f"[TerminalSafeSet] Saved to {path}")

    def equilibrium_at(self, c: float) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "fine_ellipsoid" and self.ellipsoid is not None:
            return interpolate_equilibrium(
                c, self.ellipsoid.c_grid, self.ellipsoid.x_e_grid, self.ellipsoid.u_e_grid
            )
        return interpolate_equilibrium(c, self._ss_c_grid, self._ss_x_grid, self._ss_u_grid)

    def terminal_violation(self, x_r: np.ndarray, c: float) -> float:
        """Soft violation (≥0). Zero means inside S_f."""
        x_r = np.asarray(x_r, dtype=np.float64).reshape(XR_DIM)
        x_e, _ = self.equilibrium_at(c)
        x_bar = x_r - x_e
        # wrap heading error component
        x_bar[XR_MU] = (x_bar[XR_MU] + np.pi) % (2 * np.pi) - np.pi

        if self.mode == "steady_state":
            # Weighted ball around equilibrium
            w = np.array([1.0, 1.0, 0.05, 0.2, 0.2], dtype=np.float64)
            return float(max(0.0, np.linalg.norm(w * x_bar) - self.eps_ss))

        assert self.ellipsoid is not None
        val = float(x_bar @ self.ellipsoid.P @ x_bar)
        return float(max(0.0, val - 1.0))

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "v_x": self.v_x, "c_max": self.c_max}
        if self.ellipsoid is not None:
            out["verified_objective"] = self.ellipsoid.verified_objective
            out["scale"] = self.ellipsoid.scale
        return out
