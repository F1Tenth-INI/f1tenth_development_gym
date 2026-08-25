"""Predictive safety filter (Tearle et al., arXiv:2102.11907).

Paper idea
----------
Pair *any* desired input ``u_d`` (PP, RL, ...) with an MPC that either

1. **certifies** ``u_d`` as safe and applies it unchanged, or
2. returns a **minimally invasive** alternative ``u_0*`` that keeps the car
   inside the track.

Formally (paper eqs. (3)–(5))::

    min_{x,u}  ||u_d - u_0||_W^2  +  sum_i ||Δu_i||_R^2
    s.t.       x_0 = x(k)
               x_{i+1} = f(x_i, u_i)
               x_i in X          (track constraints)
               u_i in U
               x_N in S_f        (terminal invariant set)

    Then  π_S(x, u_d) = u_0*.

Implementation
--------------
Online: SciPy SLSQP on Pacejka JAX rollouts (``psf_slsqp.py``).
Terminal set ``S_f`` (``psf_terminal_set.py``):

* ``steady_state`` — conservative raceline equilibrium ball
* ``fine_ellipsoid`` — paper SDP + nonlinear verification

Enable via ``Settings.MPC_SAFETY_FILTER``; edit tunables below.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from utilities.Settings import Settings
from utilities.state_utilities import (
    ANGULAR_CONTROL_IDX,
    TRANSLATIONAL_CONTROL_IDX,
    control_limits_high,
    control_limits_low,
)
from .psf_slsqp import solve_psf_slsqp
from .psf_terminal_set import TerminalSafeSet
from .psf_track_relative import (
    car_state_to_xr_from_waypoints,
    terminal_curvature,
    trajectory_track_metrics,
)

# ---------------------------------------------------------------------------
# Tunables (Settings only exposes MPC_SAFETY_FILTER as the master switch)
# ---------------------------------------------------------------------------
BACKEND = "rpgd"  # "rpgd" (full-horizon recovery) | "grid" (fast emergency)
RPGD_BATCH_SIZE = 8
RPGD_GRADIENT_STEPS = 8
SF_MODE = "steady_state"  # "steady_state" | "fine_ellipsoid"
PROBE_CLEARANCE = 0.20  # [m] base threshold; grows with speed
RECOVERY_HORIZON = 35  # emergency-hold horizon (~0.8 s)
TERMINAL_STOP_SPEED = 0.5  # [m/s] preferred terminal speed (soft)
RECOVERY_PROBE_EVALS = 12  # emergency-hold rollouts in the margin band
BOUNDARY_MARGIN = 0.10  # [m] shrink track corridor (≈ clearance to walls)
MIN_PATH_CLEARANCE = 0.0  # [m] extra clearance on top of margin (0 ⇒ at margin)
MAX_VY = 5.0  # [m/s] path spin limit during recovery rollouts
MAX_YAW_RATE = 12.0  # [rad/s] path yaw-rate limit
W_U = 100.0  # strong preference to keep base-controller u_d
W_DU = 0.05  # light input-rate regularisation in recovery
DT = 0.04  # Timestep for rollouts
HORIZON = 40  # Recovery horizon N (~1.20 s)
# No accel cap on the base controller — only used if explicitly set.
MAX_ACCEL = None
# Slew limits apply only during recovery (not when certified).
MAX_STEER_DELTA = 0.15
MAX_ACCEL_DELTA = 2.0
MAX_BRAKE_DELTA = 4.0
RENDER_SMOOTHING = 0.0  # show true predicted recovery path
VX_SS = 3.5  # [m/s] terminal equilibrium speed
C_MAX = 0.7  # [1/m] curvature grid for S_f
N_CURVATURE = 7  # number of curvature samples for equilibria / SDP
T_HALF = 0.40  # [m] |e_lat| half-width used in terminal SDP X_r
EPS_SS = 0.45  # S_f ball radius (~raceline lateral / heading)
TERMINAL_TOL = 3.0  # soft certify check — do not block PP
RECOVERY_TERMINAL_TOL = 0.05  # recovery must end in S_f
W_TERMINAL = 30.0  # rank recovery by raceline terminal quality
MAX_RECOVERY_EVALS = 14  # cap JAX rollouts per recovery step
RECOVERY_HOLD_STEPS = 3  # hold u0 briefly, then fade to u_eq
SLSQP_MAXITER = 0  # blend/grid recovery is enough at realtime
SLACK_PENALTY = 2.0e3  # soft track / S_f slack weight
CERTIFY_EPS = 1.0e-3  # ||u0 - ud|| below this ⇒ certified
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ELLIPSOID_PATH = os.path.join(_PKG_DIR, "fine_safe_ellipsoid.npz")
RECOMPUTE_ELLIPSOID = False  # force offline SDP recompute on load
BLEND_STEPS = 8  # fine blends u_d → recovery (prefer small α)


class MPCSafetyFilter:
    """Tearle-style predictive SF with SLSQP backup MPC + terminal S_f."""

    def __init__(self):
        self.backend = str(BACKEND).lower()
        self.rpgd_batch_size = int(RPGD_BATCH_SIZE)
        self.rpgd_gradient_steps = int(RPGD_GRADIENT_STEPS)
        self.boundary_margin = float(BOUNDARY_MARGIN)
        self.min_path_clearance = float(MIN_PATH_CLEARANCE)
        self.max_vy = float(MAX_VY)
        self.max_yaw_rate = float(MAX_YAW_RATE)
        self.w_u = float(W_U)
        self.w_du = float(W_DU)
        self.dt = float(DT if DT is not None else Settings.TIMESTEP_CONTROL)
        self.horizon = int(HORIZON)
        self.max_accel = (
            float(MAX_ACCEL)
            if MAX_ACCEL is not None and np.isfinite(float(MAX_ACCEL))
            else None
        )
        self.max_steer_delta = float(MAX_STEER_DELTA)
        self.max_accel_delta = float(MAX_ACCEL_DELTA)
        self.max_brake_delta = float(MAX_BRAKE_DELTA)
        self.render_smoothing = float(RENDER_SMOOTHING)
        self.sf_mode = str(SF_MODE)
        self.v_x_ss = float(VX_SS)
        self.c_max = float(C_MAX)
        self.n_curvature = int(N_CURVATURE)
        self.t_half = float(T_HALF)
        self.eps_ss = float(EPS_SS)
        self.slsqp_maxiter = int(SLSQP_MAXITER)
        self.slack_penalty = float(SLACK_PENALTY)
        self.certify_eps = float(CERTIFY_EPS)
        self.terminal_tol = float(TERMINAL_TOL)
        self.recovery_terminal_tol = float(RECOVERY_TERMINAL_TOL)
        self.w_terminal = float(W_TERMINAL)
        self.terminal_stop_speed = float(TERMINAL_STOP_SPEED)
        self.max_recovery_evals = int(MAX_RECOVERY_EVALS)
        self.recovery_probe_evals = int(RECOVERY_PROBE_EVALS)
        self.probe_clearance = float(PROBE_CLEARANCE)
        self.recovery_horizon = int(RECOVERY_HORIZON)
        self.recovery_hold_steps = int(RECOVERY_HOLD_STEPS)
        self.num_blend_steps = int(BLEND_STEPS)
        self.ellipsoid_path = str(ELLIPSOID_PATH)
        self.recompute_ellipsoid = bool(RECOMPUTE_ELLIPSOID)

        self._u_prev = np.zeros(2, dtype=np.float64)
        self._Q_prev: np.ndarray | None = None
        self._car_params_jax = None
        self._car_params_np: np.ndarray | None = None
        self._state_history = None
        self._control_history = None
        self._terminal: TerminalSafeSet | None = None
        self._rpgd = None
        self._render_traj_prev: np.ndarray | None = None
        self._lf = 0.16
        self._width = 0.25
        self._load_error: str | None = None
        self.render_utils = None
        self.last_info: dict[str, Any] = self._empty_info()

    def attach_render_utils(self, render_utils) -> None:
        self.render_utils = render_utils

    # ------------------------------------------------------------------ API
    def filter_from_observation(self, u_nom, observation):
        """Certify ``u_nom`` or return the closest safe backup input from SLSQP."""
        u_d = np.asarray(u_nom, dtype=np.float64).reshape(2)
        car_state = observation.get("car_state")
        next_wp = observation.get("next_waypoints")
        # Prefer the full map so predicted rollouts near wraps/walls stay well posed.
        waypoints_full = observation.get("waypoints", next_wp)
        if waypoints_full is None or len(waypoints_full) == 0:
            waypoints_full = next_wp
        waypoints_probe = waypoints_full

        if car_state is None or waypoints_probe is None or len(waypoints_probe) == 0:
            return u_d.astype(np.float32), self._passthrough_info("missing_obs")

        if not self._ensure_loaded():
            return u_d.astype(np.float32), self._passthrough_info(
                f"solver_unavailable: {self._load_error}"
            )

        xr, track = car_state_to_xr_from_waypoints(car_state, waypoints_full)
        c_term = terminal_curvature(
            waypoints_full,
            track["s"],
            float(u_d[1]),
            self.horizon,
            self.dt,
            self.v_x_ss,
        )
        c_term = float(np.clip(c_term, -self.c_max, self.c_max))
        if self.backend == "rpgd":
            return self._filter_rpgd(
                u_d,
                observation,
                np.asarray(waypoints_full, dtype=np.float64),
                xr,
                c_term,
            )

        _, u_eq = self._terminal.equilibrium_at(c_term)

        result = solve_psf_slsqp(
            s0=np.asarray(car_state, dtype=np.float64),
            u_d=u_d,
            u_prev=self._u_prev,
            waypoints=np.asarray(waypoints_probe, dtype=np.float64),
            car_params_jax=self._car_params_jax,
            car_params_np=self._car_params_np,
            dt=self.dt,
            horizon=self.horizon,
            w_u=self.w_u,
            w_du=self.w_du,
            slack_penalty=self.slack_penalty,
            lf=self._lf,
            width=self._width,
            margin=self.boundary_margin,
            c_term=c_term,
            terminal_violation_fn=self._terminal.terminal_violation,
            Q_init=self._Q_prev,
            state_history=self._state_history,
            control_history=self._control_history,
            maxiter=self.slsqp_maxiter,
            certify_eps=self.certify_eps,
            max_vy=self.max_vy,
            max_yaw_rate=self.max_yaw_rate,
            max_accel=self.max_accel,
            terminal_tol=self.terminal_tol,
            recovery_terminal_tol=self.recovery_terminal_tol,
            w_terminal=self.w_terminal,
            max_recovery_evals=self.max_recovery_evals,
            equilibrium_u=u_eq,
            recovery_hold_steps=self.recovery_hold_steps,
            num_blend_steps=self.num_blend_steps,
            e_lat=float(xr[0]),
            mu=float(xr[1]),
            min_path_clearance=self.min_path_clearance,
            track_s=float(track["s"]),
            terminal_stop_speed=self.terminal_stop_speed,
            recovery_probe_evals=self.recovery_probe_evals,
            probe_clearance=self.probe_clearance,
            recovery_horizon=self.recovery_horizon,
        )

        u_safe = np.clip(
            result.u0,
            [control_limits_low[ANGULAR_CONTROL_IDX], control_limits_low[TRANSLATIONAL_CONTROL_IDX]],
            [control_limits_high[ANGULAR_CONTROL_IDX], control_limits_high[TRANSLATIONAL_CONTROL_IDX]],
        )
        # Do not slew-limit recovery: emergency brake/steer must apply immediately.
        # Slew was delaying hard stops until after the car was already off-track.

        self._Q_prev = result.Q.copy()
        self._u_prev = u_safe.copy()
        self._update_renderer(result.trajectory)

        # Compare against actuator-limited nominal so clip-to-bounds is not counted
        # as an SF intervention.
        u_d_lim = np.clip(
            u_d,
            [control_limits_low[ANGULAR_CONTROL_IDX], control_limits_low[TRANSLATIONAL_CONTROL_IDX]],
            [control_limits_high[ANGULAR_CONTROL_IDX], control_limits_high[TRANSLATIONAL_CONTROL_IDX]],
        )
        intervene = not result.certified
        intervention = float(np.linalg.norm(u_safe - u_d_lim))
        reason = (
            "certified"
            if result.certified
            else ("recovery" if result.success else f"fail:{result.message}")
        )

        info = {
            "active": intervene and intervention > 1e-4,
            "intervene": bool(intervene),
            "reason": reason,
            "intervention": intervention,
            "recoverable_now": bool(result.min_border_clearance >= 0.0),
            "recoverable_after_nom": bool(result.certified),
            "certified": bool(result.certified),
            "max_dev_now": float("nan"),
            "max_dev_after_nom": float("nan"),
            "min_border_clearance_now": float(result.min_border_clearance),
            "min_border_clearance_after_nom": float(result.min_border_clearance),
            "min_cbf_margin_after_nom": float("nan"),
            "max_abs_vy_now": float(result.max_abs_vy),
            "max_abs_vy_after_nom": float(result.max_abs_vy),
            "max_abs_wz_now": float(result.max_abs_wz),
            "max_abs_wz_after_nom": float(result.max_abs_wz),
            "max_deviation_limit": float(self.eps_ss),
            "boundary_margin": float(self.boundary_margin),
            "max_vy_limit": float(self.max_vy),
            "max_yaw_rate_limit": float(self.max_yaw_rate),
            "alpha": float("nan"),
            "psf_cost": float(result.cost),
            "backup_cost": float(result.cost),
            "delta_nom": float(u_d[0]),
            "accel_nom": float(u_d[1]),
            "delta_safe": float(u_safe[0]),
            "accel_safe": float(u_safe[1]),
            "delta_backup": float(u_safe[0]),
            "accel_backup": float(u_safe[1]),
            "sf_mode": self.sf_mode,
            "c_term": c_term,
            "terminal_violation": float(result.terminal_violation),
            "slsqp_success": bool(result.success),
            "slsqp_nfev": int(result.nfev),
            "e_lat": float(xr[0]),
            "mu": float(xr[1]),
        }
        self.last_info = info
        return u_safe.astype(np.float32), info

    def _filter_rpgd(self, u_d, observation, waypoints, xr, c_term):
        """Tearle certify-first, then full-horizon RPGD recovery when needed."""
        if self._rpgd is None:
            return u_d.astype(np.float32), self._passthrough_info("rpgd_unavailable")

        car_state = observation.get("car_state")
        u_nom_limited = np.clip(
            u_d,
            [control_limits_low[ANGULAR_CONTROL_IDX], control_limits_low[TRANSLATIONAL_CONTROL_IDX]],
            [control_limits_high[ANGULAR_CONTROL_IDX], control_limits_high[TRANSLATIONAL_CONTROL_IDX]],
        )
        wp_np = np.asarray(waypoints, dtype=np.float64)
        car_state_np = np.asarray(car_state, dtype=np.float64)

        # Cheap Tearle certify: hold u_d for a short horizon; if path-safe, pass through.
        H_cert = int(np.clip(round(0.80 / max(self.dt, 1e-3)), 15, min(22, self.horizon)))
        Q_cert = np.tile(u_nom_limited.reshape(1, 2), (H_cert, 1))
        certify_traj = self._rollout_pacejka_Q(car_state_np, Q_cert)
        certify_metrics = trajectory_track_metrics(
            certify_traj, wp_np, self._lf, self._width, self.boundary_margin
        )
        if float(certify_metrics["min_border_clearance"]) >= self.min_path_clearance:
            return self._build_rpgd_result(
                u_safe=u_nom_limited,
                u_d=u_d,
                trajectory=certify_traj,
                metrics=certify_metrics,
                waypoints=waypoints,
                xr=xr,
                c_term=c_term,
                certified=True,
                reason="certified",
                nfev=1,
            )

        # Full-horizon RPGD recovery (the path that previously worked well).
        planner_observation = dict(observation)
        planner_observation["next_waypoints"] = np.asarray(waypoints, dtype=np.float32)
        planner_observation["nominal_control"] = np.asarray(u_d, dtype=np.float32)
        self._rpgd.process_observation(planner_observation)

        candidate_trajectories = np.asarray(
            self._rpgd.rollout_trajectories, dtype=np.float64
        )
        candidate_controls = np.asarray(
            self._rpgd.candidate_control_sequences, dtype=np.float64
        )
        candidate_clearances = np.asarray(
            [
                float(
                    trajectory_track_metrics(
                        candidate_trajectories[i],
                        wp_np,
                        self._lf,
                        self._width,
                        self.boundary_margin,
                    )["min_border_clearance"]
                )
                for i in range(candidate_trajectories.shape[0])
            ],
            dtype=np.float64,
        )
        path_feasible = [
            i
            for i, clearance in enumerate(candidate_clearances)
            if float(clearance) >= self.min_path_clearance
        ]
        execute_idx = min(
            int(Settings.CONTROL_DELAY / self.dt), candidate_controls.shape[1] - 1
        )

        def _score(i: int) -> tuple[float, float, float]:
            u0 = candidate_controls[i, execute_idx]
            # Prefer path-feasible, then minimal intervention, then less throttle.
            return (
                float(np.linalg.norm(u0 - u_nom_limited)),
                abs(float(u0[0] - u_nom_limited[0])),
                max(0.0, float(u0[1])),
            )

        if path_feasible:
            selected = min(path_feasible, key=_score)
        else:
            selected = int(np.argmax(candidate_clearances))

        steer, accel = candidate_controls[selected, execute_idx]
        u_safe = np.clip(
            np.array([steer, accel], dtype=np.float64),
            [control_limits_low[ANGULAR_CONTROL_IDX], control_limits_low[TRANSLATIONAL_CONTROL_IDX]],
            [control_limits_high[ANGULAR_CONTROL_IDX], control_limits_high[TRANSLATIONAL_CONTROL_IDX]],
        )
        # If RPGD still leaves the track, fall back to hard brake toward centerline.
        trajectory = candidate_trajectories[selected]
        metrics = trajectory_track_metrics(
            trajectory, wp_np, self._lf, self._width, self.boundary_margin
        )
        path_safe = float(metrics["min_border_clearance"]) >= self.min_path_clearance
        if not path_safe:
            delta_c = float(np.clip(-1.5 * xr[0] - 0.8 * xr[1], -0.4, 0.4))
            u_safe = np.array(
                [delta_c, float(control_limits_low[TRANSLATIONAL_CONTROL_IDX])],
                dtype=np.float64,
            )
            Q_brake = np.tile(u_safe.reshape(1, 2), (self.horizon, 1))
            trajectory = self._rollout_pacejka_Q(car_state_np, Q_brake)
            metrics = trajectory_track_metrics(
                trajectory, wp_np, self._lf, self._width, self.boundary_margin
            )
            path_safe = float(metrics["min_border_clearance"]) >= self.min_path_clearance

        return self._build_rpgd_result(
            u_safe=u_safe,
            u_d=u_d,
            trajectory=trajectory,
            metrics=metrics,
            waypoints=waypoints,
            xr=xr,
            c_term=c_term,
            certified=False,
            reason="rpgd_recovery" if path_safe else "rpgd_soft_constraint_violation",
            nfev=self.rpgd_batch_size * self.rpgd_gradient_steps,
        )

    def _rollout_pacejka_Q(self, car_state, Q: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp
        from sim.f110_sim.envs.car_model_jax import car_steps_sequential_jax

        traj = car_steps_sequential_jax(
            jnp.asarray(car_state, dtype=jnp.float32),
            jnp.asarray(Q, dtype=jnp.float32),
            self._car_params_jax,
            float(self.dt),
            int(Q.shape[0]),
            model_type="pacejka",
            state_history=self._state_history,
            control_history=self._control_history,
        )
        return np.asarray(traj, dtype=np.float64)

    def _rollout_constant_control(self, car_state, u0: np.ndarray) -> np.ndarray:
        Q = np.tile(np.asarray(u0, dtype=np.float32).reshape(1, 2), (self.horizon, 1))
        return self._rollout_pacejka_Q(car_state, Q)

    def _build_rpgd_result(
        self,
        *,
        u_safe: np.ndarray,
        u_d: np.ndarray,
        trajectory: np.ndarray,
        metrics: dict,
        waypoints: np.ndarray,
        xr: np.ndarray,
        c_term: float,
        certified: bool,
        reason: str,
        nfev: int,
    ):
        u_safe = np.clip(
            np.asarray(u_safe, dtype=np.float64).reshape(2),
            [control_limits_low[ANGULAR_CONTROL_IDX], control_limits_low[TRANSLATIONAL_CONTROL_IDX]],
            [control_limits_high[ANGULAR_CONTROL_IDX], control_limits_high[TRANSLATIONAL_CONTROL_IDX]],
        )
        u_nom_limited = np.clip(
            u_d,
            [control_limits_low[ANGULAR_CONTROL_IDX], control_limits_low[TRANSLATIONAL_CONTROL_IDX]],
            [control_limits_high[ANGULAR_CONTROL_IDX], control_limits_high[TRANSLATIONAL_CONTROL_IDX]],
        )
        intervention = float(np.linalg.norm(u_safe - u_nom_limited))
        min_clearance = float(metrics["min_border_clearance"])
        path_safe = min_clearance >= self.min_path_clearance
        xr_terminal, _ = car_state_to_xr_from_waypoints(trajectory[-1], waypoints)
        terminal_speed = float(np.hypot(xr_terminal[2], xr_terminal[3])) if self._terminal else 0.0
        terminal_violation = max(0.0, terminal_speed - self.terminal_stop_speed)

        self._Q_prev = np.tile(u_safe.reshape(1, 2), (self.horizon, 1))
        self._u_prev = u_safe.copy()
        self._update_renderer(trajectory)

        info = self._empty_info()
        info.update(
            {
                "active": (not certified) and intervention > 1e-4,
                "intervene": (not certified) and intervention > 1e-4,
                "reason": reason,
                "intervention": intervention,
                "recoverable_now": path_safe,
                "recoverable_after_nom": path_safe,
                "certified": certified,
                "min_border_clearance_now": min_clearance,
                "min_border_clearance_after_nom": min_clearance,
                "boundary_margin": self.boundary_margin,
                "psf_cost": 0.0 if certified else float("nan"),
                "backup_cost": 0.0 if certified else float("nan"),
                "delta_nom": float(u_d[0]),
                "accel_nom": float(u_d[1]),
                "delta_safe": float(u_safe[0]),
                "accel_safe": float(u_safe[1]),
                "delta_backup": float(u_safe[0]),
                "accel_backup": float(u_safe[1]),
                "sf_mode": f"{self.sf_mode}:rpgd",
                "c_term": c_term,
                "terminal_violation": terminal_violation,
                "slsqp_success": path_safe,
                "slsqp_nfev": int(nfev),
                "e_lat": float(xr[0]),
                "mu": float(xr[1]),
            }
        )
        self.last_info = info
        return u_safe.astype(np.float32), info

    # ----------------------------------------------------------- internals
    def _ensure_loaded(self) -> bool:
        if self._car_params_jax is not None and self._terminal is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import jax.numpy as jnp
            from utilities.car_files.vehicle_parameters import VehicleParameters

            print(f"[MPCSafetyFilter] Loading Pacejka PSF (S_f mode={self.sf_mode}) ...")
            veh = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
            params = veh.to_np_array().astype(np.float32)
            self._car_params_np = params.astype(np.float64)
            self._car_params_jax = jnp.asarray(params)
            self._lf = float(veh.lf)
            self._width = float(veh.width)
            self._state_history = jnp.zeros((10, 10), dtype=jnp.float32)
            self._control_history = jnp.zeros((10, 2), dtype=jnp.float32)

            mode = self.sf_mode if self.sf_mode in ("steady_state", "fine_ellipsoid") else "steady_state"
            self._terminal = TerminalSafeSet(
                mode=mode,  # type: ignore[arg-type]
                car_params=self._car_params_np,
                v_x=self.v_x_ss,
                c_max=self.c_max,
                n_curvature=self.n_curvature,
                dt=self.dt,
                t_half=self.t_half,
                eps_ss=self.eps_ss,
                ellipsoid_path=self.ellipsoid_path if mode == "fine_ellipsoid" else None,
                recompute_ellipsoid=self.recompute_ellipsoid,
            )
            if self.backend == "rpgd":
                from .rpgd_jax_safety_filter import RPGDJaxSafetyFilter

                self._rpgd = RPGDJaxSafetyFilter(
                    horizon=self.horizon,
                    batch_size=self.rpgd_batch_size,
                    gradient_steps=self.rpgd_gradient_steps,
                    control_smoothing_alpha=1.0,
                    safety_margin=self.boundary_margin,
                    w_u=self.w_u,
                    quiet=True,
                )
            print("[MPCSafetyFilter] Ready:", self._terminal.info())
            return True
        except Exception as exc:  # pragma: no cover
            self._load_error = str(exc)
            print(f"[MPCSafetyFilter] Failed to load: {exc}")
            return False

    def _update_renderer(self, traj) -> None:
        render_utils = self.render_utils
        if render_utils is None or not hasattr(render_utils, "update_mpc"):
            return
        if traj is None or len(np.asarray(traj)) == 0:
            return
        highlight = np.asarray(traj, dtype=np.float32)
        if highlight.ndim == 2:
            highlight = highlight[None, ...]
        if (
            self.render_smoothing > 0.0
            and self._render_traj_prev is not None
            and self._render_traj_prev.shape == highlight.shape
        ):
            alpha = float(np.clip(self.render_smoothing, 0.0, 0.95))
            highlight = alpha * self._render_traj_prev + (1.0 - alpha) * highlight
        self._render_traj_prev = highlight.copy()
        render_utils.update_mpc(
            rollout_trajectory=np.zeros((0, 0, 0), dtype=np.float32),
            optimal_trajectory=highlight,
        )

    def _slew_limit_control(self, u_safe: np.ndarray, result) -> np.ndarray:
        """Rate-limit the applied safety-filter command to avoid chatter."""
        u = np.asarray(u_safe, dtype=np.float64).reshape(2).copy()
        prev = np.asarray(self._u_prev, dtype=np.float64).reshape(2)

        if np.isfinite(self.max_steer_delta):
            steer_step = float(np.clip(u[0] - prev[0], -self.max_steer_delta, self.max_steer_delta))
            u[0] = prev[0] + steer_step

        if np.isfinite(self.max_accel_delta) or np.isfinite(self.max_brake_delta):
            accel_step = float(u[1] - prev[1])
            emergency = (
                float(getattr(result, "min_border_clearance", 1.0)) < 0.08
                or float(getattr(result, "max_abs_vy", 0.0)) > self.max_vy
            )
            max_up = self.max_accel_delta if np.isfinite(self.max_accel_delta) else np.inf
            max_down = (
                4.0 * self.max_brake_delta
                if emergency and np.isfinite(self.max_brake_delta)
                else self.max_brake_delta
            )
            if accel_step >= 0.0:
                accel_step = min(accel_step, float(max_up))
            else:
                accel_step = max(accel_step, -float(max_down))
            u[1] = prev[1] + accel_step

        return np.clip(
            u,
            [control_limits_low[ANGULAR_CONTROL_IDX], control_limits_low[TRANSLATIONAL_CONTROL_IDX]],
            [control_limits_high[ANGULAR_CONTROL_IDX], control_limits_high[TRANSLATIONAL_CONTROL_IDX]],
        )

    def _empty_info(self) -> dict[str, Any]:
        return {
            "active": False,
            "intervene": False,
            "reason": "init",
            "intervention": 0.0,
            "recoverable_now": True,
            "recoverable_after_nom": True,
            "certified": False,
            "max_dev_now": 0.0,
            "max_dev_after_nom": 0.0,
            "min_border_clearance_now": 0.0,
            "min_border_clearance_after_nom": 0.0,
            "min_cbf_margin_after_nom": float("nan"),
            "max_abs_vy_now": 0.0,
            "max_abs_vy_after_nom": 0.0,
            "max_abs_wz_now": 0.0,
            "max_abs_wz_after_nom": 0.0,
            "max_deviation_limit": float(self.eps_ss),
            "boundary_margin": float(self.boundary_margin),
            "max_vy_limit": float(self.max_vy),
            "max_yaw_rate_limit": float(self.max_yaw_rate),
            "alpha": float("nan"),
            "psf_cost": float("nan"),
            "backup_cost": float("nan"),
            "delta_nom": float("nan"),
            "accel_nom": float("nan"),
            "delta_safe": float("nan"),
            "accel_safe": float("nan"),
            "delta_backup": float("nan"),
            "accel_backup": float("nan"),
            "sf_mode": self.sf_mode,
            "c_term": float("nan"),
            "terminal_violation": float("nan"),
            "slsqp_success": False,
            "slsqp_nfev": 0,
            "e_lat": float("nan"),
            "mu": float("nan"),
        }

    def _passthrough_info(self, reason: str) -> dict[str, Any]:
        info = self._empty_info()
        info["reason"] = reason
        self.last_info = info
        return info
