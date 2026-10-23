"""acados NMPC with track constraints and a moving terminal set (see ``model.py``).

Each control step:

1. Predict the state at which the new command takes effect (control delay) with
   the commands still in the delay buffer.
2. Candidate: the previous input sequence shifted by one stage and extended with
   one step of the raceline-tracking policy (full braking if that is infeasible).
3. Solve the OCP (a few SQP iterations).  The solver's input sequence is rolled
   out on the OCP model and evaluated on the OCP constraints; if it is feasible it
   is applied, otherwise whichever of solver output and candidate violates the
   constraints less (suboptimal MPC).
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from Control_Toolkit_ASF.Controllers import template_planner
from utilities.Settings import Settings
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    STEERING_ANGLE_IDX,
)

from . import config as cfg
from .model import IU_ACCEL, IU_STEER, IX_DELTA, IX_VX, NU, create_solver
from .plant import Plant, commands_to_plan, gym_to_frenet, plan_to_commands
from .track import RacelineTrack

STATS_PRINT_EVERY = 250


class SafeMPCPlanner(template_planner):
    def __init__(self):
        super().__init__()
        print("Loading SafeMPCPlanner (acados)")
        self.veh = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
        if Settings.FRICTION_FOR_CONTROLLER is not None:
            self.veh.mu = Settings.FRICTION_FOR_CONTROLLER
        self.dt = float(Settings.TIMESTEP_CONTROL)
        self.n_sub = int(round(Settings.TIMESTEP_CONTROL / Settings.TIMESTEP_SIM))
        self.delay_steps = int(round(Settings.CONTROL_DELAY / Settings.TIMESTEP_SIM))
        self.N = cfg.N_HORIZON

        self.track: RacelineTrack | None = None
        self.plant: Plant | None = None
        self.solver = None
        self.fns = None
        self.optimal_control_sequence = np.zeros((self.N, NU))
        self.stats = {"solver": 0, "shifted": 0, "infeasible": 0, "solve_time": 0.0, "step_time": 0.0, "steps": 0}
        self.last_info: dict = {}
        self.debug: dict = {}
        self.reset()

    def reset(self):
        super().reset()
        self.delay_buffer = deque([np.zeros(2)] * self.delay_steps, maxlen=max(self.delay_steps, 1))
        self.prev_U: np.ndarray | None = None
        self.pending_iterate: tuple[np.ndarray, np.ndarray] | None = None
        self.s_hint: float | None = None

    # ------------------------------------------------------------------ setup
    def _ensure_track(self, waypoints: np.ndarray) -> None:
        track = RacelineTrack(
            waypoints,
            mu=self.veh.mu,
            g=self.veh.g,
            lat_acc_factor_ref=cfg.LAT_ACC_FACTOR_REF,
            kappa_smoothing_window=cfg.KAPPA_SMOOTHING_WINDOW,
        )
        if self.track is not None and track.fingerprint == self.track.fingerprint:
            return
        t0 = time.time()
        print(f"[SafeMPC] building acados solver for track {track.fingerprint} (cached builds are instant) ...")
        self.track = track
        self.plant = Plant(self.veh, Settings.TIMESTEP_SIM, self.n_sub)
        self.solver, self.fns = create_solver(self.veh, track, self.dt, self.n_sub)
        self._rollout_fn = self.fns.f_disc.mapaccum(self.N)
        self._g_path_fn = self.fns.g_path.map(self.N - 1)
        print(f"[SafeMPC] solver ready in {time.time() - t0:.1f}s; horizon {self.N * self.dt:.2f}s")
        self.reset()

    # --------------------------------------------------------------- helpers
    def _frenet_state(self, state: np.ndarray, accel_prev: float) -> np.ndarray:
        s, n, mu = gym_to_frenet(self.track, state, self.s_hint)
        return np.array([s[0], n[0], mu[0], state[LINEAR_VEL_X_IDX], state[LINEAR_VEL_Y_IDX],
                         state[ANGULAR_VEL_Z_IDX], state[STEERING_ANGLE_IDX], accel_prev], dtype=np.float64)

    def _rollout_model(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """OCP-model trajectory of a full-horizon input sequence."""
        return np.vstack([x0, np.asarray(self._rollout_fn(x0, U.T)).T])

    def _violation(self, X: np.ndarray, U: np.ndarray) -> float:
        """Largest OCP constraint violation of a trajectory (<= 0: feasible)."""
        return max(float(np.max(self.fns.g_stage0(X[0], U[0]))),
                   float(np.max(self._g_path_fn(X[1:-1].T, U[1:].T))),
                   float(np.max(self.fns.g_terminal(X[-1]))))

    def _policy_input(self, x: np.ndarray, full_brake: bool = False) -> np.ndarray:
        """Raceline-tracking policy: Stanley-like steering, speed towards the terminal speed bound one stage ahead."""
        veh, track = self.veh, self.track
        s, n, mu, vx, delta = x[0], x[1], x[2], x[IX_VX], x[IX_DELTA]
        a_lo, a_hi = -0.9 * veh.mu * veh.g, min(veh.a_max, 0.9 * veh.mu * veh.g)
        s_next = s + max(vx, 0.0) * self.dt
        v_tgt = min(float(track.v_ref_at(s_next)), cfg.TERMINAL_SPEED_FACTOR * float(track.v_wp_at(s_next)))
        accel = a_lo if full_brake else np.clip(cfg.GUESS_SPEED_GAIN * (v_tgt - vx), a_lo, a_hi)
        d_delta = 0.98 * veh.sv_max * self.dt
        delta_cmd = (np.arctan((veh.lf + veh.lr) * float(track.kappa_at(s))) - cfg.GUESS_K_HEADING * mu
                     - np.arctan(cfg.GUESS_K_LATERAL * n / (abs(vx) + 1.0)))
        delta_cmd = np.clip(delta_cmd, max(veh.s_min, delta - d_delta), min(veh.s_max, delta + d_delta))
        u = np.zeros(NU)
        u[IU_STEER], u[IU_ACCEL] = delta_cmd / veh.steering_multiplier, accel
        return u

    def _candidate(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Previous solution shifted by one stage and extended with one policy step (or a policy rollout)."""
        if self.prev_U is None:
            U, X = self._policy_rollout(x0)
            return U, X, self._violation(X, U)
        U = np.vstack([self.prev_U[1:], self.prev_U[-1:]])
        x_last = self._rollout_model(x0, U)[-2]
        best = None
        for full_brake in (False, True):
            U[-1] = self._policy_input(x_last, full_brake)
            X = self._rollout_model(x0, U)
            v = self._violation(X, U)
            if best is None or v < best[2]:
                best = (U.copy(), X, v)
            if v <= cfg.FEASIBILITY_TOL:
                break
        return best

    def _policy_rollout(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.empty((self.N + 1, len(x0)))
        U = np.empty((self.N, NU))
        X[0] = x0
        for k in range(self.N):
            U[k] = self._policy_input(X[k])
            X[k + 1] = np.asarray(self.fns.f_disc(X[k], U[k])).ravel()
        return U, X

    def _set_guess(self, X: np.ndarray, U: np.ndarray) -> None:
        for i in range(self.N):
            self.solver.set(i, "x", X[i])
            self.solver.set(i, "u", U[i])
        self.solver.set(self.N, "x", X[self.N])

    # ------------------------------------------------------------------- main
    def process_observation(self, controller_observation):
        t_step = time.perf_counter()
        state = np.asarray(controller_observation["car_state"], dtype=np.float64)
        self._ensure_track(np.asarray(controller_observation["waypoints"]))

        if self.delay_steps > 0:
            x_pred = self.plant.rollout(state, np.asarray(self.delay_buffer), n_sub=1)[-1].astype(np.float64)
        else:
            x_pred = state
        last_command = self.delay_buffer[-1] if self.delay_steps > 0 else np.zeros(2)
        accel_prev = float(commands_to_plan(last_command, self.veh)[0, IU_ACCEL])
        x0 = self._frenet_state(x_pred, accel_prev)
        self.s_hint = float(x0[0])

        U_cand, X_cand, viol_cand = self._candidate(x0)
        if self.pending_iterate is not None:
            # previous solve was not feasible yet: continue its SQP iterations
            X_it, U_it = self.pending_iterate
            X_guess, U_guess = np.vstack([x0, X_it[2:], X_it[-1:]]), np.vstack([U_it[1:], U_it[-1:]])
            X_guess[1:, 0] -= self.track.length * np.round((X_guess[1, 0] - x0[0]) / self.track.length)
        elif X_cand[self.N // 2, IX_VX] < cfg.GUESS_STOPPED_SPEED:
            U_guess, X_guess = self._policy_rollout(x0)
        else:
            U_guess, X_guess = U_cand, X_cand
        self._set_guess(X_guess, U_guess)
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        status = self.solver.solve()
        solve_time = float(self.solver.get_stats("time_tot"))
        U = np.array([self.solver.get(i, "u") for i in range(self.N)])
        X_it = np.array([self.solver.get(i, "x") for i in range(self.N + 1)])

        violation = np.inf
        if np.all(np.isfinite(U)):
            X = self._rollout_model(x0, U)
            if np.all(np.isfinite(X)):
                violation = self._violation(X, U)
        if violation <= cfg.FEASIBILITY_TOL:
            source = "solver"
            self.pending_iterate = None
        else:
            finite = np.all(np.isfinite(X_it)) and np.all(np.isfinite(U))
            self.pending_iterate = (X_it, U) if finite else None
            if viol_cand <= violation:
                X, U, source, violation = X_cand, U_cand, "shifted", viol_cand
            else:
                source = "solver"
        if violation > cfg.FEASIBILITY_TOL:
            self.stats["infeasible"] += 1
        self.prev_U = U
        commands = plan_to_commands(U, self.veh)
        self.debug = {"x_pred": x_pred, "X": X, "U": U}

        command = commands[0]
        for _ in range(self.n_sub if self.delay_steps > 0 else 0):
            self.delay_buffer.append(command.copy())

        self.optimal_control_sequence = commands
        step_time = time.perf_counter() - t_step
        self.last_info = {
            "source": source, "acados_status": int(status), "violation": round(float(violation), 4),
            "solve_ms": round(1e3 * solve_time, 1), "step_ms": round(1e3 * step_time, 1),
        }
        self._update_stats(source, solve_time, step_time)
        if self.render_utils is not None:
            self.render_utils.update_mpc(rollout_trajectory=None,
                                         optimal_trajectory=self.plant.rollout(x_pred, commands)[None])

        self.angular_control, self.translational_control = float(command[0]), float(command[1])
        return self.angular_control, self.translational_control

    def _update_stats(self, source: str, solve_time: float, step_time: float) -> None:
        st = self.stats
        st[source] += 1
        st["steps"] += 1
        st["solve_time"] += solve_time
        st["step_time"] += step_time
        if st["steps"] % STATS_PRINT_EVERY == 0:
            n = st["steps"]
            print(f"[SafeMPC] {n} steps: solver output used {100 * st['solver'] / n:.1f}%, "
                  f"shifted previous solution {100 * st['shifted'] / n:.1f}%, "
                  f"no feasible plan {100 * st['infeasible'] / n:.1f}%, "
                  f"mean acados {1000 * st['solve_time'] / n:.1f} ms, mean step {1000 * st['step_time'] / n:.1f} ms"
                  f" | last: {self.last_info}")
