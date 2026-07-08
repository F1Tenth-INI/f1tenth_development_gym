"""Control-Barrier-Function (CBF) safety filter for the F1TENTH car.

This is a controller-agnostic safety layer that lives at the ``CarSystem`` level.
It takes the *nominal* control produced by any planner (MPC, MPPI, RL/SAC, FTG,
pure pursuit, ...) and minimally modifies it so the closed loop stays inside the
track and below the lateral-grip limit:

    u_safe = argmin_u ||u - u_nom||_W^2 + rho * xi^2
             s.t.  CBF constraints (relaxable by slack xi >= 0)
                   u in [u_min, u_max]

Two barriers are used (single car, no opponents):

1. Track-boundary barrier (relative degree 2 w.r.t. steering -> HOCBF).
   Uses the Frenet lateral deviation ``d`` and the per-waypoint distances to the
   left/right track border. Keeps ``-(d_right - margin) <= d <= (d_left - margin)``.

2. Friction-circle / speed barrier (relative degree 1 w.r.t. acceleration).
   Keeps the demanded lateral acceleration ``v^2 * kappa_eff`` below
   ``CBF_GRIP_FACTOR * CBF_SPEED_MARGIN * mu * g``. Upcoming curvature is a
   discount-weighted average over the look-ahead window (near waypoints weigh
   more than far ones); see :func:`discounted_kappa_ahead`.

Model
-----
A kinematic single-track model in the Frenet frame is used for the barrier
derivatives (standard for F1TENTH CBF work):

    d_dot   = v * sin(e)
    e_dot   = psi_dot - kappa * s_dot,   psi_dot = (v / L) * tan(delta)
    s_dot   = v * cos(e) / (1 - kappa * d)
    v_dot   = a

with ``v = v_x`` (forward body velocity) and ``L`` the wheelbase. ``tan(delta)``
is small-angle linearised (``tan d ~ d``) so every barrier row is affine in the
control and the whole thing is a tiny convex QP solved with ``quadprog``.

Control semantics
-----------------
``angular_control`` is a desired steering angle [rad] and ``translational_control``
is a longitudinal acceleration [m/s^2] (the default, ``MOTOR_PID_IN_CAR_MODEL =
False``). If ``MOTOR_PID_IN_CAR_MODEL`` is True the second channel is a desired
speed; in that case the acceleration channel is left untouched (fixed to nominal)
and the speed barrier is disabled, so only the steering channel is filtered.
"""

from __future__ import annotations

import numpy as np

from utilities.Settings import Settings
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    ANGULAR_CONTROL_IDX,
    TRANSLATIONAL_CONTROL_IDX,
    control_limits_low,
    control_limits_high,
)
from utilities.waypoint_utils import WP_KAPPA_IDX, WP_D_LEFT_IDX, WP_D_RIGHT_IDX

try:
    import quadprog  # small, fast strictly-convex QP solver (in requirements.txt)
    _HAS_QUADPROG = True
except Exception:  # pragma: no cover - solver missing -> filter becomes a pass-through
    _HAS_QUADPROG = False


def discounted_kappa_ahead(
    next_waypoints: np.ndarray,
    discount: float = 0.92,
    max_steps: int = 0,
) -> float:
    """Discount-weighted mean of |kappa| over upcoming waypoints.

    Waypoint index ``i`` receives weight ``discount**i`` (``i=0`` is the nearest
    upcoming point). This reacts to imminent curvature while still looking ahead,
    without the over-conservatism of ``max(|kappa|)`` over the whole window.

    Args:
        next_waypoints: look-ahead waypoint array (N, >= WP_KAPPA_IDX+1).
        discount: per-step decay in (0, 1]; 1.0 = uniform average.
        max_steps: if > 0, only the first ``max_steps`` waypoints are used;
            0 = use the full array.

    Returns:
        Effective curvature [1/m] for the speed CBF.
    """
    if next_waypoints is None or len(next_waypoints) == 0:
        return 0.0

    kappas = np.abs(np.asarray(next_waypoints[:, WP_KAPPA_IDX], dtype=np.float64))
    if max_steps > 0:
        kappas = kappas[:max_steps]
    n = len(kappas)
    if n == 0:
        return 0.0

    gamma = float(np.clip(discount, 1e-6, 1.0))
    weights = gamma ** np.arange(n, dtype=np.float64)
    return float(np.dot(weights, kappas) / np.sum(weights))


class CBFSafetyFilter:
    """Minimally-invasive CBF-QP safety filter, independent of the controller."""

    def __init__(
        self,
        wheelbase: float | None = None,
        mu: float | None = None,
        g: float = 9.81,
        margin: float | None = None,
        alpha_1: float | None = None,
        alpha_2: float | None = None,
        alpha_v: float | None = None,
        grip_factor: float | None = None,
        weight_steering: float | None = None,
        weight_accel: float | None = None,
        slack_penalty: float | None = None,
        enable_speed_barrier: bool | None = None,
        v_eps: float = 0.3,
    ):
        # Vehicle geometry / limits (fall back to the configured car parameter file).
        if wheelbase is None or mu is None:
            try:
                from utilities.car_files.vehicle_parameters import VehicleParameters

                params = VehicleParameters(Settings.ENV_CAR_PARAMETER_FILE)
                wheelbase = wheelbase if wheelbase is not None else float(params.lf + params.lr)
                mu = mu if mu is not None else float(params.mu)
                g = float(getattr(params, "g", g))
            except Exception:
                wheelbase = wheelbase if wheelbase is not None else 0.25
                mu = mu if mu is not None else 1.0

        self.L = float(wheelbase)
        self.mu = float(mu)
        self.g = float(g)

        # Tunables (Settings override defaults so experiments stay reproducible).
        self.margin = _cfg(margin, "CBF_BOUNDARY_MARGIN", 0.20)
        self.alpha_1 = _cfg(alpha_1, "CBF_ALPHA_1", 2.5)
        self.alpha_2 = _cfg(alpha_2, "CBF_ALPHA_2", 2.5)
        self.alpha_v = _cfg(alpha_v, "CBF_ALPHA_V", 3.0)
        self.grip_factor = _cfg(grip_factor, "CBF_GRIP_FACTOR", 0.9)
        if self.grip_factor > 1.0:
            print(
                f"[CBFSafetyFilter] CBF_GRIP_FACTOR={self.grip_factor} > 1 is non-physical; "
                "use CBF_SPEED_MARGIN for tuning headroom instead."
            )
            self.grip_factor = min(self.grip_factor, 1.0)
        self.speed_margin = _cfg(None, "CBF_SPEED_MARGIN", 1.0)
        self.kappa_discount = float(getattr(Settings, "CBF_KAPPA_DISCOUNT", 0.92))
        self.kappa_max_steps = int(getattr(Settings, "CBF_KAPPA_MAX_STEPS", 0))
        self.slack_penalty = _cfg(slack_penalty, "CBF_SLACK_PENALTY", 1.0e4)
        self.enable_speed_barrier = bool(
            enable_speed_barrier
            if enable_speed_barrier is not None
            else getattr(Settings, "CBF_ENABLE_SPEED_BARRIER", True)
        )
        self.v_eps = float(v_eps)

        # Control box limits (canonical clip from state_utilities).
        self.delta_lb = float(control_limits_low[ANGULAR_CONTROL_IDX])
        self.delta_ub = float(control_limits_high[ANGULAR_CONTROL_IDX])
        self.accel_lb = float(control_limits_low[TRANSLATIONAL_CONTROL_IDX])
        self.accel_ub = float(control_limits_high[TRANSLATIONAL_CONTROL_IDX])

        # Objective weights: normalise deviations by control range so steering and
        # acceleration corrections are comparably "expensive".
        delta_range = max(self.delta_ub - self.delta_lb, 1e-3)
        accel_range = max(self.accel_ub - self.accel_lb, 1e-3)
        self.w_steering = _cfg(weight_steering, "CBF_WEIGHT_STEERING", 1.0) / (delta_range ** 2)
        self.w_accel = _cfg(weight_accel, "CBF_WEIGHT_ACCEL", 1.0) / (accel_range ** 2)

        self._pid_speed_mode = bool(getattr(Settings, "MOTOR_PID_IN_CAR_MODEL", False))

        # Debug / render info from the last call.
        self.last_info: dict = {
            "active": False,
            "intervention": 0.0,
            "slack": 0.0,
            "h_left": np.inf,
            "h_right": np.inf,
            "h_speed": np.inf,
        }

    # ------------------------------------------------------------------ API
    def filter_from_observation(self, u_nom, observation):
        """Apply the safety filter using a controller observation dict.

        Extracts everything the filter needs (car state, Frenet coordinates,
        track-border distances and upcoming curvature) from the standard
        controller-observation schema, then delegates to :meth:`filter`.

        Args:
            u_nom: (2,) nominal control [steering_angle, acceleration].
            observation: controller observation dict (as built by
                ``CarSystem._build_controller_observation``). Must contain
                ``car_state``, ``frenet_coordinates`` and ``next_waypoints``.

        Returns:
            (u_safe (np.ndarray shape (2,)), info dict).
        """
        u_nom_arr = np.asarray(u_nom, dtype=np.float64)
        car_state = observation.get("car_state")
        frenet = observation.get("frenet_coordinates")
        next_wp = observation.get("next_waypoints")

        if car_state is None or frenet is None or next_wp is None or len(next_wp) == 0:
            return u_nom_arr, self._passthrough_info()

        # Border distances at the car (first upcoming waypoint) and discount-weighted
        # upcoming curvature for the speed barrier.
        d_left = float(next_wp[0, WP_D_LEFT_IDX])
        d_right = float(next_wp[0, WP_D_RIGHT_IDX])
        kappa_ahead = discounted_kappa_ahead(
            next_wp, self.kappa_discount, self.kappa_max_steps
        )

        return self.filter(u_nom_arr, car_state, frenet, d_left, d_right, kappa_ahead)

    def filter(
        self,
        u_nom,
        car_state,
        frenet,
        d_left: float,
        d_right: float,
        kappa_ahead: float,
    ):
        """Return a safe control close to ``u_nom``.

        This is a thin orchestrator: it evaluates the CBF constraints via
        :meth:`compute_cbf_constraints` and then solves the safety QP via
        :meth:`solve_qp`.

        Args:
            u_nom: (2,) nominal control [steering_angle, acceleration].
            car_state: full car state vector (STATE_VARIABLES ordering).
            frenet: (s, d, e, kappa) from ``WaypointUtils.get_frenet_coordinates``.
            d_left: distance from the raceline to the left border at the car [m] (>=0).
            d_right: distance from the raceline to the right border at the car [m] (>=0).
            kappa_ahead: representative |curvature| of the upcoming track [1/m].

        Returns:
            (u_safe (np.ndarray shape (2,)), info dict).
        """
        delta_nom = float(u_nom[0])
        accel_nom = float(u_nom[1])
        u_nom_arr = np.array([delta_nom, accel_nom], dtype=np.float64)

        # Bail out to nominal if we cannot build a meaningful barrier.
        if not _HAS_QUADPROG or frenet is None or car_state is None:
            return u_nom_arr, self._passthrough_info()
        d, e, k_frenet = float(frenet[1]), float(frenet[2]), float(frenet[3])
        if not np.all(np.isfinite([d, e, k_frenet, d_left, d_right, kappa_ahead])):
            return u_nom_arr, self._passthrough_info()

        # --- The CBF: barrier values and affine constraint rows ---------------
        cbf_rows, cbf_b, h_values = self.compute_cbf_constraints(
            car_state, frenet, d_left, d_right, kappa_ahead
        )

        # --- The filter: solve the safety QP ----------------------------------
        u_safe, slack, solved = self.solve_qp(
            u_nom_arr, cbf_rows, cbf_b, fix_accel=self._pid_speed_mode
        )
        if not solved:
            u_safe = np.clip(u_nom_arr, [self.delta_lb, self.accel_lb], [self.delta_ub, self.accel_ub])
            slack = 0.0

        intervention = float(np.linalg.norm(u_safe - u_nom_arr))
        info = {
            "active": intervention > 1e-4,
            "intervention": intervention,
            "slack": float(slack),
            "h_left": float(h_values["h_left"]),
            "h_right": float(h_values["h_right"]),
            "h_speed": float(h_values["h_speed"]),
            "kappa_ahead": float(kappa_ahead),
            "delta_nom": delta_nom,
            "accel_nom": accel_nom,
            "delta_safe": float(u_safe[0]),
            "accel_safe": float(u_safe[1]),
        }
        self.last_info = info
        return u_safe.astype(np.float32), info

    # -------------------------------------------------------------- the CBF
    def compute_cbf_constraints(self, car_state, frenet, d_left, d_right, kappa_ahead):
        """Evaluate the control barrier functions and their QP constraint rows.

        Builds, for the current state, the affine-in-control constraints

            A @ [delta, accel] + slack >= b

        that encode the safety barriers (track boundary + friction circle), using
        a kinematic single-track model in the Frenet frame with a small-angle
        ``tan(delta) ~ delta`` linearisation. This function contains all barrier
        definitions; it does no optimisation.

        Args:
            car_state: full car state vector (STATE_VARIABLES ordering).
            frenet: (s, d, e, kappa) Frenet coordinates.
            d_left, d_right: distances to the left/right track border at the car [m].
            kappa_ahead: representative |curvature| of the upcoming track [1/m].

        Returns:
            (rows, b_vals, h_values):
                rows: list of [coef_delta, coef_accel] constraint gradients.
                b_vals: list of right-hand sides (one per row).
                h_values: dict of current barrier values {h_left, h_right, h_speed}.
        """
        d, e, kappa = float(frenet[1]), float(frenet[2]), float(frenet[3])

        v_x = float(car_state[LINEAR_VEL_X_IDX])
        v_y = float(car_state[LINEAR_VEL_Y_IDX])
        v = float(np.hypot(v_x, v_y))

        sin_e = np.sin(e)
        cos_e = np.cos(e)

        # s_dot with a guarded (1 - kappa*d) denominator.
        denom = 1.0 - kappa * d
        denom = denom if abs(denom) > 1e-3 else (1e-3 if denom >= 0 else -1e-3)
        s_dot = v * cos_e / denom

        # Corridor half-extents (signed d is positive to the left of the raceline).
        D_left = max(d_left - self.margin, 1e-3)
        D_right = max(d_right - self.margin, 1e-3)

        # Barrier values (>= 0 is safe).
        h_left = D_left - d          # d must stay below the left border
        h_right = d + D_right        # d must stay above the (negative) right border
        a_lat_max = self.grip_factor * self.speed_margin * self.mu * self.g
        kappa_eff = abs(kappa_ahead)
        h_speed = a_lat_max - (v * v) * kappa_eff

        a1, a2 = self.alpha_1, self.alpha_2
        # Shared steering authority term: d/dt of (v cos e * psi_dot) w.r.t. delta.
        steer_gain = (v * v) * cos_e / self.L  # coefficient of delta (small-angle tan)

        rows = []   # each: [coef_delta, coef_accel] for coef . u + slack >= b
        b_vals = []

        # --- Track-boundary HOCBF (relative degree 2 in steering) --------------
        # h_ddot + (a1+a2) h_dot + a1 a2 h >= 0, with slack relaxation.
        # Left:  h = D_left - d,  h_dot = -v sin e
        left_const = v * cos_e * kappa * s_dot - (a1 + a2) * v * sin_e + a1 * a2 * h_left
        rows.append([-steer_gain, -sin_e])
        b_vals.append(-left_const)
        # Right: h = d + D_right,  h_dot = +v sin e
        right_const = -v * cos_e * kappa * s_dot + (a1 + a2) * v * sin_e + a1 * a2 * h_right
        rows.append([steer_gain, sin_e])
        b_vals.append(-right_const)

        # --- Friction-circle / speed CBF (relative degree 1 in acceleration) ---
        if self.enable_speed_barrier and not self._pid_speed_mode:
            # h_dot + alpha_v h >= 0,  h_dot = -2 v kappa_eff a
            speed_const = self.alpha_v * h_speed
            rows.append([0.0, -2.0 * v * kappa_eff])
            b_vals.append(-speed_const)

        h_values = {"h_left": h_left, "h_right": h_right, "h_speed": h_speed}
        return rows, b_vals, h_values

    # ---------------------------------------------------------- the QP / filter
    def solve_qp(self, u_nom, cbf_rows, cbf_b, fix_accel: bool):
        """Solve the minimally-invasive safety QP.

        Decision variable x = [delta, accel, slack]:

            min 0.5 w_d (delta-delta_nom)^2 + 0.5 w_a (accel-accel_nom)^2
                + 0.5 rho slack^2
            s.t.  cbf_rows @ [delta, accel] + slack >= cbf_b
                  control box limits, slack >= 0

        Args:
            u_nom: (2,) nominal control [delta, accel].
            cbf_rows: iterable of [coef_delta, coef_accel] constraint gradients.
            cbf_b: iterable of right-hand sides (one per CBF row).
            fix_accel: if True, pin acceleration to nominal (speed-command mode).

        Returns:
            (u_safe (2,), slack, solved_flag).
        """
        # 0.5 x^T G x - a^T x  ==  0.5 w_d (d-d0)^2 + 0.5 w_a (a-a0)^2 + 0.5 rho slack^2
        G = np.diag([self.w_steering, self.w_accel, self.slack_penalty]).astype(np.float64)
        a_lin = np.array(
            [self.w_steering * u_nom[0], self.w_accel * u_nom[1], 0.0], dtype=np.float64
        )

        C_cols = []
        b_list = []

        # CBF constraints: [coef_delta, coef_accel, 1(slack)] . x >= rhs
        for row, rhs in zip(cbf_rows, cbf_b):
            C_cols.append([row[0], row[1], 1.0])
            b_list.append(rhs)

        # Box constraints on the controls.
        C_cols.append([1.0, 0.0, 0.0]);  b_list.append(self.delta_lb)     # delta >= lb
        C_cols.append([-1.0, 0.0, 0.0]); b_list.append(-self.delta_ub)    # delta <= ub
        if fix_accel:
            # Pin acceleration to nominal (channel is a speed command, not filtered).
            C_cols.append([0.0, 1.0, 0.0]);  b_list.append(u_nom[1] - 1e-6)
            C_cols.append([0.0, -1.0, 0.0]); b_list.append(-(u_nom[1] + 1e-6))
        else:
            C_cols.append([0.0, 1.0, 0.0]);  b_list.append(self.accel_lb)   # accel >= lb
            C_cols.append([0.0, -1.0, 0.0]); b_list.append(-self.accel_ub)  # accel <= ub
        C_cols.append([0.0, 0.0, 1.0]); b_list.append(0.0)                # slack >= 0

        C = np.array(C_cols, dtype=np.float64).T  # shape (n_var, n_constraints)
        b = np.array(b_list, dtype=np.float64)

        try:
            x = quadprog.solve_qp(G, a_lin, C, b, 0)[0]
            return np.array([x[0], x[1]]), x[2], True
        except Exception:
            return np.asarray(u_nom, dtype=np.float64).copy(), 0.0, False

    def _passthrough_info(self):
        info = dict(self.last_info)
        info.update({"active": False, "intervention": 0.0, "slack": 0.0})
        return info


def _cfg(explicit, settings_name: str, default):
    """Resolve a parameter: explicit arg > Settings attribute > hard default."""
    if explicit is not None:
        return float(explicit)
    return float(getattr(Settings, settings_name, default))
