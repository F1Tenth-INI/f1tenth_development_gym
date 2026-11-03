"""Frenet-frame Pacejka model and acados OCP for the safe MPC.

State   x = [s, n, mu, v_x, v_y, r, delta]
Control u = [steer_cmd, accel]

``steer_cmd`` and ``accel`` are exactly what the simulator receives. ``accel``
is the commanded longitudinal acceleration before rolling resistance, the
friction ellipse and the steered-tyre drag. ``brake_multiplier`` and
``accel_multiplier`` are applied only in the simulator, as a sim-to-real scale
for a mistuned ESC; the controller does not cancel them.  The dynamics
replicate ``car_dynamics_pacejka_jax`` for ``Settings.ODE_MODEL_OF_CAR_DYNAMICS``:

* Pacejka slip angles use ``|v_x|`` and ``sign(v_x) * delta`` (valid in reverse);
* a friction ellipse derates both axles by ``sqrt(1 - (a_x / (mu g))^2)``;
* body-frame rates include ``F_yf sin/cos(delta)`` and the ``v_y r``, ``-v_x r`` terms;
* ``ODE:ks`` integrates speed along the kinematic sideslip and resets ``v_x``, ``v_y`` and yaw rate from the steering angle;
* ``ODE:ks_pacejka`` uses the dynamic model when the car is fast or a tyre is sliding, and the kinematic model near standstill.

Integration is the simulator's explicit Euler sub-steps.  Constraints keep the
plan where the simulator's clips (servo rate, steering limit, friction limit)
are inactive, so model and plant coincide there apart from the ESC scale.

Terminal-set MPC:

* every car corner stays ``BORDER_MARGIN`` inside the track at every stage;
* terminal set: inside the track, ``v_x <= TERMINAL_SPEED_FACTOR * v_wp(s)`` (waypoint speed) and
  ``|mu| <= TERMINAL_MAX_HEADING_DEG`` (heading relative to the raceline);
* the previous solution shifted by one stage and extended with one step of a
  raceline-tracking policy is the fallback candidate.  Unlike a standstill
  terminal set, this set is not provably invariant, so the candidate's
  feasibility is checked online instead of holding by construction.
  Track and terminal constraints are exact-penalty slacks: their solution
  equals the hard-constrained one whenever that is feasible.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from typing import NamedTuple

import casadi as ca
import numpy as np

from utilities.Settings import Settings

from . import config as cfg

os.environ.setdefault("ACADOS_SOURCE_DIR", cfg.ACADOS_SOURCE_DIR)

NX_CAR, NU = 7, 2
NX = NX_CAR + 1  # + previous accel command (for the input-rate cost)
IX_S, IX_N, IX_MU, IX_VX, IX_VY, IX_R, IX_DELTA, IX_ACCEL_PREV = range(NX)
IU_STEER, IU_ACCEL = range(NU)
N_BORDER = 8  # left/right border for each car corner


class OcpFunctions(NamedTuple):
    """CasADi functions of the OCP model: discrete dynamics and constraint residuals (feasible iff all <= 0)."""
    f_disc: ca.Function      # (x, u) -> x_next
    g_stage0: ca.Function    # (x0, u0) -> stage-0 constraints (x0 itself is fixed)
    g_path: ca.Function      # (x_k, u_k) -> track + input/friction constraints, k = 1..N-1
    g_terminal: ca.Function  # x_N -> track constraints + terminal speed / heading

# Slip-angle floor and ks/Pacejka blend, same constants as dynamic_model_pacejka_jax.
VX_SLIP_FLOOR = 1.0e-3       # [m/s] |v_x| floor in the slip angles
KS_SPEED_STEEPNESS = 4.8
KS_SPEED_MID = 1.75          # [m/s] total speed at which the speed-blend is 0.5
KS_SLIDE_STEEPNESS = 10.0
KS_SLIDE_MID = 0.5           # [m/s] axle sliding speed at which the slide-blend is 0.5


def build_interpolants(track, s_after: float):
    knots, values = track.periodic_grid(s_before=5.0, s_after=s_after)
    return {k: ca.interpolant(k, "bspline", [knots], v) for k, v in values.items()}


def _smooth_sign(v, v_dead):
    return v / ca.sqrt(v * v + v_dead * v_dead)


def slip_angles(veh, vx, vy, r, delta):
    """Pacejka slip angles. ``|v_x|`` and ``sign(v_x) * delta`` stay valid in reverse."""
    vxs = ca.fmax(ca.fabs(vx), VX_SLIP_FLOOR)
    alpha_f = -ca.atan((vy + r * veh.lf) / vxs) + (vx / vxs) * delta
    alpha_r = -ca.atan((vy - r * veh.lr) / vxs)
    return alpha_f, alpha_r


def ks_blend_weight(veh, vx, vy, r, delta):
    """Dynamic-model weight of ``ODE:ks_pacejka``.

    The dynamic model is used when the car is fast or a tyre is sliding
    (a sideways slide must not be stopped by the kinematic model). Near
    standstill with little slide the weight is about 0.
    """
    speed = ca.sqrt(vx * vx + vy * vy)
    slide_f = (vy + veh.lf * r) * ca.cos(delta) - vx * ca.sin(delta)
    slide_r = vy - veh.lr * r
    slide = ca.fmax(ca.fabs(slide_f), ca.fabs(slide_r))
    w_speed = 1.0 / (1.0 + ca.exp(-KS_SPEED_STEEPNESS * (speed - KS_SPEED_MID)))
    w_slide = 1.0 / (1.0 + ca.exp(-KS_SLIDE_STEEPNESS * (slide - KS_SLIDE_MID)))
    return 1.0 - (1.0 - w_speed) * (1.0 - w_slide)


def body_dynamics(veh, vx, vy, r, delta, steer_cmd, accel):
    """Continuous body-frame rates of one Pacejka step.

    Returns ``(vx_dot, vy_dot, r_dot, delta_dot, a_long, cornering)``.
    ``a_long`` is the command plus rolling resistance; the simulator clips that,
    then clips it again after ``cornering``.  ``F_yf sin/cos(delta)`` and the
    rotating-frame terms sit in the rates and are not part of those clips.
    """
    g = veh.g
    v_dead = max(veh.v_dead, cfg.OCP_V_DEAD)
    rolling = -veh.c_rr * g * _smooth_sign(vx, v_dead)
    a_long = accel + rolling

    alpha_f, alpha_r = slip_angles(veh, vx, vy, r, delta)
    l = veh.lf + veh.lr
    F_zf = veh.m * (-a_long * veh.h + g * veh.lr) / l
    F_zr = veh.m * (a_long * veh.h + g * veh.lf) / l
    Bf, Cf, Df, Ef = veh.C_Pf
    Br, Cr, Dr, Er = veh.C_Pr
    F_yf = veh.mu * F_zf * Df * ca.sin(Cf * ca.atan(Bf * alpha_f - Ef * (Bf * alpha_f - ca.atan(Bf * alpha_f))))
    F_yr = veh.mu * F_zr * Dr * ca.sin(Cr * ca.atan(Br * alpha_r - Er * (Br * alpha_r - ca.atan(Br * alpha_r))))
    # 4WD, load-proportional drive: the same ellipse factor derates both axles.
    lat_scale = ca.sqrt(ca.fmax(1.0 - (a_long / (veh.mu * g)) ** 2, 1.0e-4))
    F_yf, F_yr = F_yf * lat_scale, F_yr * lat_scale

    f_eps = cfg.OCP_LAT_FORCE_SMOOTH
    lat_force = ca.sqrt(F_yf ** 2 + F_yr ** 2 + f_eps ** 2) - f_eps
    cornering = -veh.curve_resistance_factor * lat_force / veh.m * _smooth_sign(vx, v_dead)
    a_x = a_long + cornering
    vx_dot = a_x - F_yf * ca.sin(delta) / veh.m + vy * r
    vy_dot = (F_yr + F_yf * ca.cos(delta)) / veh.m - vx * r
    r_dot = (veh.lf * F_yf * ca.cos(delta) - veh.lr * F_yr) / veh.I_z
    delta_dot = veh.servo_p * (veh.steering_multiplier * steer_cmd - delta)
    return vx_dot, vy_dot, r_dot, delta_dot, a_long, cornering


def frenet_rhs(veh, interp, x, u):
    s, n, mu, vx, vy, r, delta = (x[i] for i in range(NX_CAR))
    steer_cmd, accel = u[0], u[1]
    kappa = interp["kappa"](s)
    vx_dot, vy_dot, r_dot, delta_dot, _, _ = body_dynamics(veh, vx, vy, r, delta, steer_cmd, accel)
    s_dot = (vx * ca.cos(mu) - vy * ca.sin(mu)) / (1.0 - n * kappa)
    n_dot = vx * ca.sin(mu) + vy * ca.cos(mu)
    mu_dot = r - kappa * s_dot
    return ca.vertcat(s_dot, n_dot, mu_dot, vx_dot, vy_dot, r_dot, delta_dot)


def euler_substep(veh, interp, x, u, h, ode_model: str):
    """One simulator sub-step, including the ``ODE:ks_pacejka`` kinematic/dynamic blend."""
    x_dyn = x + h * frenet_rhs(veh, interp, x, u)
    if ode_model == "ODE:pacejka":
        return x_dyn

    s, n, mu, vx, vy, r, delta = (x[i] for i in range(NX_CAR))
    steer_cmd, accel = u[0], u[1]
    kappa = interp["kappa"](s)
    _, _, _, delta_dot, a_long, _ = body_dynamics(veh, vx, vy, r, delta, steer_cmd, accel)
    l_wb = veh.lf + veh.lr
    # CoG kinematic bicycle: speed is integrated along the no-slip direction beta(delta),
    # then v_x, v_y and yaw rate are reset from the steered geometry. Steering changes
    # the direction of the velocity, not its magnitude.
    v = ca.if_else(vx < 0, -1.0, 1.0) * ca.sqrt(vx * vx + vy * vy)
    beta = ca.atan(ca.tan(delta) * veh.lr / l_wb)
    vx_b, vy_b = v * ca.cos(beta), v * ca.sin(beta)
    psi_dot = vx_b * ca.tan(delta) / l_wb
    s_dot = (vx_b * ca.cos(mu) - vy_b * ca.sin(mu)) / (1.0 - n * kappa)
    v1 = v + h * a_long
    delta1 = delta + h * delta_dot
    beta1 = ca.atan(ca.tan(delta1) * veh.lr / l_wb)
    vx1 = v1 * ca.cos(beta1)
    x_kin = ca.vertcat(
        s + h * s_dot,
        n + h * (vx_b * ca.sin(mu) + vy_b * ca.cos(mu)),
        mu + h * (psi_dot - kappa * s_dot),
        vx1,
        v1 * ca.sin(beta1),
        vx1 * ca.tan(delta1) / l_wb,
        delta1,
    )
    if ode_model == "ODE:ks":
        return x_kin
    w = ks_blend_weight(veh, vx, vy, r, delta)
    mixed = (1.0 - w) * x_kin + w * x_dyn
    # Circular blend of the heading error, as ``_blend_states_circular_yaw`` does for yaw.
    mu_b = ca.atan2((1.0 - w) * ca.sin(x_kin[IX_MU]) + w * ca.sin(x_dyn[IX_MU]),
                    (1.0 - w) * ca.cos(x_kin[IX_MU]) + w * ca.cos(x_dyn[IX_MU]))
    return ca.vertcat(mixed[IX_S], mixed[IX_N], mu_b, mixed[IX_VX], mixed[IX_VY], mixed[IX_R], mixed[IX_DELTA])


def corner_offsets(veh):
    half_l, half_w = 0.5 * veh.length, 0.5 * veh.width
    return [(half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)]


def corner_lateral_positions(veh, n, mu):
    return [n + cx * ca.sin(mu) + cy * ca.cos(mu) for cx, cy in corner_offsets(veh)]


def build_ocp(veh, track, dt: float, n_sub: int, name: str):
    from acados_template import AcadosModel, AcadosOcp

    N = cfg.N_HORIZON
    s_after = veh.v_max * N * dt + 5.0
    interp = build_interpolants(track, s_after)

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    s, n, mu, vx, vy, r, delta, accel_prev = (x[i] for i in range(NX))
    steer_cmd, accel = u[0], u[1]

    h_sub = dt / n_sub
    xk = x[:NX_CAR]
    for _ in range(n_sub):
        xk = euler_substep(veh, interp, xk, u, h_sub, Settings.ODE_MODEL_OF_CAR_DYNAMICS)

    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = u
    model.disc_dyn_expr = ca.vertcat(xk, accel)

    kappa = interp["kappa"](s)
    d_left = interp["d_left"](s)
    d_right = interp["d_right"](s)
    v_ref = interp["v_ref"](s)
    yaw_err = r - kappa * vx
    servo_rate = veh.servo_p * (veh.steering_multiplier * steer_cmd - delta)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    # --- cost ---------------------------------------------------------------------
    # Stage: reward progress along the raceline (the speed is limited only by the constraints)
    # plus quadratic regularisation.  The optional speed term uses the relative error, because an
    # absolute error (vx - v_ref)^2 makes progress into fast sections expensive.
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    s_dot = (vx * ca.cos(mu) - vy * ca.sin(mu)) / (1.0 - n * kappa)
    speed_err = 1.0 - vx / ca.fmax(v_ref, 1.0)
    y = ca.vertcat(n, mu, speed_err, yaw_err, vy, servo_rate, accel, accel - accel_prev)
    w = np.array([cfg.W_LATERAL, cfg.W_HEADING, cfg.W_SPEED, cfg.W_YAW_ERR, cfg.W_VY,
                  cfg.W_STEER_RATE, cfg.W_ACCEL, cfg.W_ACCEL_RATE])
    model.cost_expr_ext_cost = 0.5 * ca.dot(y, w * y) - cfg.W_PROGRESS * s_dot
    model.cost_y_expr_e = ca.vertcat(n, mu)
    ocp.cost.W_e = cfg.W_TERMINAL_SCALE * np.diag(w[:2])
    ocp.cost.yref_e = np.zeros(2)

    # --- input / state bounds ------------------------------------------------------
    mu_g = veh.mu * veh.g
    steer_lim_lo = veh.s_min / veh.steering_multiplier
    steer_lim_hi = veh.s_max / veh.steering_multiplier
    ocp.constraints.x0 = np.zeros(NX)
    ocp.constraints.idxbu = np.array([IU_STEER, IU_ACCEL])
    ocp.constraints.lbu = np.array([steer_lim_lo, max(veh.a_min, -mu_g)])
    ocp.constraints.ubu = np.array([steer_lim_hi, min(veh.a_max, mu_g)])
    ocp.constraints.idxbx = np.array([IX_VX, IX_DELTA])
    ocp.constraints.lbx = np.array([0.0, veh.s_min])
    ocp.constraints.ubx = np.array([veh.v_max, veh.s_max])

    # servo rate stays inside [sv_min, sv_max] -> simulator's rate clip never acts
    C = np.zeros((1, NX))
    C[0, IX_DELTA] = -veh.servo_p
    D = np.zeros((1, NU))
    D[0, IU_STEER] = veh.servo_p * veh.steering_multiplier
    ocp.constraints.C = C
    ocp.constraints.D = D
    ocp.constraints.lg = np.array([0.98 * veh.sv_min])
    ocp.constraints.ug = np.array([0.98 * veh.sv_max])

    # --- nonlinear path constraints -------------------------------------------------
    _, _, _, _, a_long, cornering = body_dynamics(veh, vx, vy, r, delta, steer_cmd, accel)
    accel_h = ca.vertcat(
        a_long,                                    # >= -mu g  (rolling clip inactive)
        a_long + cornering,                        # >= -mu g  (cornering clip inactive)
        accel * ca.fmax(vx, veh.v_switch),         # <= a_max * v_switch (power limit inactive)
    )
    accel_lh = np.array([-0.98 * mu_g, -0.98 * mu_g, -1e4])
    accel_uh = np.array([0.98 * mu_g, 0.98 * mu_g, veh.a_max * veh.v_switch])

    corners = corner_lateral_positions(veh, n, mu)
    border_h = ca.vertcat(*[c - d_left for c in corners], *[-c - d_right for c in corners])
    n_border = N_BORDER
    border_uh = -cfg.BORDER_MARGIN * np.ones(n_border)
    border_lh = -1e3 * np.ones(n_border)

    slip_h = ca.vertcat(*slip_angles(veh, vx, vy, r, delta))
    if Settings.ODE_MODEL_OF_CAR_DYNAMICS == "ODE:ks_pacejka":
        # tyres only matter where the simulator uses the dynamic model
        slip_h = slip_h * ks_blend_weight(veh, vx, vy, r, delta)
    elif Settings.ODE_MODEL_OF_CAR_DYNAMICS == "ODE:ks":
        slip_h = 0.0 * slip_h
    slip_ub = np.array([cfg.MAX_SLIP_FRONT, cfg.MAX_SLIP_REAR])

    # The solver gets limits tightened by SOLVER_TIGHTENING; the rollout check uses the nominal
    # ones, so plans that are only approximately converged still pass it.
    tight = 1.0 - cfg.SOLVER_TIGHTENING
    border_uh_ocp = (1.0 + cfg.SOLVER_TIGHTENING) * border_uh

    model.con_h_expr_0 = accel_h
    ocp.constraints.lh_0 = tight * accel_lh
    ocp.constraints.uh_0 = tight * accel_uh

    model.con_h_expr = ca.vertcat(border_h, slip_h, accel_h)
    ocp.constraints.lh = np.concatenate([border_lh, -slip_ub, tight * accel_lh])
    ocp.constraints.uh = np.concatenate([border_uh_ocp, slip_ub, tight * accel_uh])
    ocp.constraints.idxsh = np.arange(n_border + 2)

    # --- terminal set: inside the track, v_x <= waypoint speed, heading near the raceline ----
    mu_max = np.deg2rad(cfg.TERMINAL_MAX_HEADING_DEG)
    v_term = cfg.TERMINAL_SPEED_FACTOR * interp["v_wp"](s)
    terminal_h = ca.vertcat(border_h, vx - v_term, mu)
    terminal_lh = np.concatenate([border_lh, [-1e3, -mu_max]])
    terminal_uh = np.concatenate([border_uh, [0.0, mu_max]])
    model.con_h_expr_e = ca.vertcat(border_h, vx - tight * v_term, mu)
    ocp.constraints.lh_e = np.concatenate([border_lh, [-1e3, -tight * mu_max]])
    ocp.constraints.uh_e = np.concatenate([border_uh_ocp, [0.0, tight * mu_max]])
    ns_e = n_border + 2
    ocp.constraints.idxsh_e = np.arange(ns_e)

    z_path = np.concatenate([cfg.SLACK_L1 * np.ones(n_border), cfg.SLIP_SLACK_L1 * np.ones(2)])
    Z_path = np.concatenate([cfg.SLACK_L2 * np.ones(n_border), cfg.SLIP_SLACK_L2 * np.ones(2)])
    ocp.cost.zl = z_path
    ocp.cost.zu = z_path
    ocp.cost.Zl = Z_path
    ocp.cost.Zu = Z_path
    ocp.cost.zl_e = cfg.SLACK_L1 * np.ones(ns_e)
    ocp.cost.zu_e = cfg.SLACK_L1 * np.ones(ns_e)
    ocp.cost.Zl_e = cfg.SLACK_L2 * np.ones(ns_e)
    ocp.cost.Zu_e = cfg.SLACK_L2 * np.ones(ns_e)

    # --- solver ---------------------------------------------------------------------
    so = ocp.solver_options
    so.integrator_type = "DISCRETE"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_cond_N = max(1, N // cfg.QP_COND_BLOCK)
    so.nlp_solver_tol_eq = cfg.NLP_TOL
    so.nlp_solver_tol_ineq = cfg.NLP_TOL
    so.nlp_solver_tol_stat = cfg.NLP_TOL_STAT
    so.nlp_solver_tol_comp = cfg.NLP_TOL
    so.hessian_approx = cfg.HESSIAN_APPROX
    if cfg.HESSIAN_APPROX == "EXACT":
        so.regularize_method = "PROJECT"
    so.nlp_solver_type = "SQP_RTI" if cfg.SQP_MAX_ITER <= 1 else "SQP"
    so.nlp_solver_max_iter = max(1, cfg.SQP_MAX_ITER)
    if so.nlp_solver_type == "SQP":
        so.globalization = cfg.GLOBALIZATION
    so.qp_solver_iter_max = 100
    so.levenberg_marquardt = cfg.LEVENBERG_MARQUARDT
    so.print_level = 0

    code_dir = os.path.join(cfg.CODEGEN_DIR, name)
    ocp.code_export_directory = code_dir

    # OCP constraints (slip limits excluded, they only condition the NLP) as "<= 0" residuals
    servo = C @ x + D @ u
    x_bounded = x[ocp.constraints.idxbx]
    hard_path = ca.vertcat(accel_lh - accel_h, accel_h - accel_uh, ocp.constraints.lg - servo,
                           servo - ocp.constraints.ug, ocp.constraints.lbx - x_bounded,
                           x_bounded - ocp.constraints.ubx)
    fns = OcpFunctions(
        f_disc=ca.Function("f_disc", [x, u], [model.disc_dyn_expr]),
        g_stage0=ca.Function("g_stage0", [x, u], [hard_path[:-2 * len(ocp.constraints.idxbx)]]),
        g_path=ca.Function("g_path", [x, u], [ca.vertcat(border_h - border_uh, hard_path)]),
        g_terminal=ca.Function("g_terminal", [x], [ca.vertcat(terminal_h - terminal_uh, terminal_lh - terminal_h,
                                                              x_bounded - ocp.constraints.ubx,
                                                              ocp.constraints.lbx - x_bounded)]),
    )
    return ocp, fns


def _preload_acados_libs() -> None:
    """Make libacados' dependencies resolvable without LD_LIBRARY_PATH."""
    lib_dir = os.path.join(cfg.ACADOS_SOURCE_DIR, "lib")
    for lib in ("libblasfeo.so", "libhpipm.so", "libacados.so"):
        path = os.path.join(lib_dir, lib)
        if os.path.isfile(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def create_solver(veh, track, dt: float, n_sub: int):
    from acados_template import AcadosOcpSolver

    _preload_acados_libs()
    tunables = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    key = repr((track.fingerprint, sorted(tunables.items()), sorted(veh.to_dict().items()), dt, n_sub,
                Settings.ODE_MODEL_OF_CAR_DYNAMICS, open(__file__).read()))
    name = "safe_mpc_" + hashlib.sha1(key.encode()).hexdigest()[:12]
    ocp, fns = build_ocp(veh, track, dt, n_sub, name)
    json_file = f"{name}.json"
    code_dir = ocp.code_export_directory
    lib = os.path.join(code_dir, f"libacados_ocp_solver_{name}.so")
    cached = os.path.isfile(os.path.join(code_dir, json_file)) and os.path.isfile(lib)
    solver = AcadosOcpSolver(ocp, json_file=json_file, build=not cached, generate=not cached, verbose=False)
    return solver, fns
