"""System identification drive for a small room.

Every maneuver is driven on a circle, because a circle is the only "infinite
straight" a 5 m room has: steady state is reached and held for as long as needed,
and the lateral acceleration v^2 / R reaches the grip limit at low speed.

The program (``SYSID_PROGRAM = "full"``), in order, first counter-clockwise:

1. warm-up lap on the main circle (largest circle with ``SYSID_WALL_MARGIN`` to the walls)
2. kinematic laps at 0.8 m/s on three radii and one lap at full steering lock
   -> steering gain (``steering_multiplier``) and lock (``s_max``)
3. speed staircases on the main and an inner circle up to the grip limit
   -> understeer gradient, tyre curves, ``mu``
4. random steering steps on top of the circle tracker at three speeds
   -> yaw inertia ``I_z``, cornering stiffness, servo speed
5. acceleration / braking / coasting pulses
   -> ``accel_multiplier``, ``brake_multiplier``, ``c_rr``
6. large, fast steering doublets at high lateral load (like an MPC's corrections in hard corners)
7. figure eight to change direction, then 2-4 and 6 clockwise (left/right asymmetry, steering offset)

``SYSID_PROGRAM = "transients"`` drives only 6, both directions (~1.5 min).

Every command is a closed-loop circle tracker (pure pursuit on the circle plus an
integral on the radius error, so understeer does not push the car wide) and a
speed PI. Excitation is added on top. Identification uses the commands that were
actually applied (they are recorded), so the feedback does not bias the fit.

Safety: if the body comes closer than ``SYSID_ABORT_CLEARANCE`` to a wall, or the
car is more than ``SYSID_ABORT_RADIUS_ERROR`` off its circle, the program brakes to
a stop and ends. The staircases stop by themselves once the car can no longer hold
its radius (that is the grip limit, which is what they measure). Optionally
``Settings.MPC_SAFETY_FILTER`` filters the commands on top (counter-clockwise only:
the filter works relative to the map's raceline, which is the ccw main circle).

When the program is done ``experiment_complete`` is set; the simulator stops, the
car stays at rest. Fit the recording with ``SysId/fit_parameters.py``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from Control_Toolkit_ASF.Controllers import template_planner
from Control_Toolkit_ASF.Controllers.SysId.arena import Arena, Circle, segment_clearance, tangent_segment
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.Settings import Settings
from utilities.state_utilities import (
    LINEAR_VEL_X_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)

STEER_CMD_LIMIT = 0.4  # [rad] command clip (state_utilities control limits)
STAIRCASE_RUNOFF = 0.25  # [m] staircases run this far inside their circle
TRANSIENT_MAX_RADIUS_ERROR = 0.25  # [m] large steering doublets stop at this distance from the circle
G = 9.81


class _Abort(Exception):
    pass


class SysIdPlanner(template_planner):

    def __init__(self):
        print("Loading SysIdPlanner")
        super().__init__()
        Settings.RECORDING_FOLDER = getattr(Settings, "SYSID_RECORDING_FOLDER", "./ExperimentRecordings/Sysid/")

        # Nominal model: only used for feedforward and for converting accelerations into commands
        vp = VehicleParameters(Settings.CONTROLLER_CAR_PARAMETER_FILE)
        self.l_wb = float(vp.lf + vp.lr)
        self.steer_mult = float(vp.steering_multiplier)
        self.accel_mult = float(vp.accel_multiplier)
        self.brake_mult = float(vp.brake_multiplier)
        self.c_rr = float(vp.c_rr)
        self.mu_nominal = float(vp.mu)
        # kinematic radius at the steering lock, on the side that locks first
        self.r_min = self.l_wb / math.tan(min(float(vp.s_max), -float(vp.s_min)))
        self.half_diagonal = 0.5 * math.hypot(vp.length, vp.width)
        self.dt = Settings.TIMESTEP_CONTROL

        self.wall_margin = float(getattr(Settings, "SYSID_WALL_MARGIN", 0.35))
        self.abort_clearance = float(getattr(Settings, "SYSID_ABORT_CLEARANCE", 0.12))
        self.abort_radius_error = float(getattr(Settings, "SYSID_ABORT_RADIUS_ERROR", 0.7))
        self.speed_max = float(getattr(Settings, "SYSID_SPEED_MAX", 4.5))
        self.directions = str(getattr(Settings, "SYSID_DIRECTIONS", "both"))
        self.program_name = str(getattr(Settings, "SYSID_PROGRAM", "full"))

        self.arena = Arena(Settings.MAP_PATH, Settings.MAP_NAME, Settings.MAP_SCALE)
        self.main = self.arena.main_circle(self.half_diagonal, self.wall_margin)
        if self.main.r < 1.15 * self.r_min:
            raise RuntimeError(
                f"SysId: the largest circle with {self.wall_margin} m wall margin has r={self.main.r:.2f} m, "
                f"but the car needs r >= {1.15 * self.r_min:.2f} m. Free more space or lower SYSID_WALL_MARGIN.")
        self.inner_r = max(1.3 * self.r_min, 0.7 * self.main.r)
        self.lobe_r = 1.2 * self.r_min
        self.figure_eight = None
        if self.directions == "both":
            self.figure_eight = self.arena.figure_eight(self.lobe_r, self.half_diagonal, 0.15)
            if self.figure_eight is None:
                print("SysId: no figure eight fits the map; driving counter-clockwise only. "
                      "Run again with SYSID_DIRECTIONS='cw' and the car placed clockwise.")
                self.directions = "ccw"
        print(f"SysId arena: main circle center ({self.main.cx:.2f}, {self.main.cy:.2f}) r={self.main.r:.2f} m, "
              f"inner r={self.inner_r:.2f} m, kinematic min radius {self.r_min:.2f} m")
        if self.figure_eight is not None:
            a, b, c = self.figure_eight
            print(f"SysId figure eight: lobes ({a.cx:.2f}, {a.cy:.2f}) and ({b.cx:.2f}, {b.cy:.2f}), r={a.r:.2f} m")

        self.experiment_complete = False
        self.a_lat_limit = None  # [m/s^2] lowest grip limit a staircase found
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self):
        super().reset()
        self.state = None
        self.t = 0.0
        self.phase = "start"
        self._phase_start = 0.0
        self.phase_log = []  # (time, phase name)
        self.v_ref_filtered = 0.0
        self.speed_integral = 0.0
        self.radius_integral = 0.0
        self.rng = np.random.default_rng(0)
        self._program = self._full_program()
        self._current_circle = None
        self._set_raceline_direction(-1 if self.directions == "cw" else +1)
        self.experiment_complete = False

    def process_observation(self, controller_observation=None):
        self.state = self.get_car_state(controller_observation)
        if self.experiment_complete:
            self.angular_control, self.translational_control = 0.0, 0.0
        else:
            try:
                self.angular_control, self.translational_control = next(self._program)
            except StopIteration:
                self._finish("program complete")
                self.angular_control, self.translational_control = 0.0, 0.0
        self.t += self.dt
        self.simulation_index += 1
        return float(self.angular_control), float(self.translational_control)

    # ------------------------------------------------------------- program
    def _full_program(self):
        try:
            directions = {"both": (+1, -1), "ccw": (+1,), "cw": (-1,)}[self.directions]
            for i, d in enumerate(directions):
                main = self.main.with_direction(d)
                if i == 0:
                    yield from self._drive(main, v=1.0, laps=0.5, name="warmup")
                else:
                    yield from self._change_direction()
                    yield from self._transition(main, v=1.0)
                yield from self._direction_block(main, quick=(self.program_name == "quick"))
            yield from self._stop(main)
        except _Abort as e:
            print(f"SysId ABORT: {e}")
            yield from self._stop(self._current_circle or self.main, brake=4.0, check=False)
        self._finish("program complete")

    def _direction_block(self, main: Circle, quick: bool):
        d = main.direction
        tag = "ccw" if d > 0 else "cw"
        if self.program_name == "transients":
            fast = main.with_radius(main.r - STAIRCASE_RUNOFF)
            yield from self._transition(fast, v=1.0)
            yield from self._steer_transients(fast, tag)
            yield from self._transition(main, v=1.0)
            return
        # Kinematic: low speed on three radii, then full lock
        yield from self._drive(main, v=0.8, laps=0.5, name=f"{tag}_kinematic_R{main.r:.2f}")
        for r in (self.inner_r, 1.12 * self.r_min):
            yield from self._transition(main.with_radius(r), v=0.8)
            yield from self._drive(main.with_radius(r), v=0.8, laps=0.75, name=f"{tag}_kinematic_R{r:.2f}")
        yield from self._full_lock(d, v=0.8, name=f"{tag}_full_lock")
        # Steady state up to the grip limit. The fast circle keeps STAIRCASE_RUNOFF to the main circle
        fast = main.with_radius(main.r - STAIRCASE_RUNOFF)
        yield from self._transition(fast, v=1.0)
        yield from self._staircase(fast, name=f"{tag}_staircase_R{fast.r:.2f}")
        if not quick:
            inner = main.with_radius(self.inner_r)
            yield from self._transition(inner, v=1.0)
            yield from self._staircase(inner, name=f"{tag}_staircase_R{inner.r:.2f}")
            yield from self._transition(fast, v=1.0)
        # Transient lateral, at fractions of the measured limit
        v_lim = self._limit_speed(fast.r)
        fractions = (0.45, 0.6, 0.75) if (d > 0 and not quick) else (0.6,)
        for f in fractions:
            v = round(f * v_lim, 2)
            yield from self._drive(fast, v=v, duration=2.0, name=f"{tag}_settle_{v}")
            yield from self._drive(fast, v=v, duration=6.0, steer_excitation=0.05, name=f"{tag}_steer_steps_{v}")
        if not quick:
            yield from self._steer_transients(fast, tag)
        # Longitudinal
        if d > 0:
            yield from self._pulses(fast, name=f"{tag}_long_pulses", quick=quick)
        yield from self._transition(main, v=1.0)

    def _steer_transients(self, circle: Circle, tag: str):
        """Large, fast steering doublets at high lateral load, like an MPC's corrections in hard corners.

        A doublet is +A for T, then -A for T (random sign, T 0.12-0.22 s), followed by 0.35-0.7 s
        for the circle tracker: sharp large-angle steering with zero net heading change, so the car
        stays near the circle. Levels A = 0.06 / 0.10 / 0.14 rad command at about 2.0 and 2.4 m/s.
        If the car gets more than TRANSIENT_MAX_RADIUS_ERROR off the circle, it brakes gently back
        onto it and the larger levels at that speed are skipped.
        """
        v_lim = self._limit_speed(circle.r)
        speeds = sorted({round(min(2.0, 0.75 * v_lim), 2), round(min(2.4, 0.87 * v_lim), 2)})
        for v in speeds:
            yield from self._drive(circle, v=v, duration=2.5, name=f"{tag}_transients_settle_{v}")
            for amp in (0.06, 0.10, 0.14):
                schedule, t = [], 0.3
                while t < 5.0:
                    sign, T = self.rng.choice((-1.0, 1.0)), self.rng.uniform(0.12, 0.22)
                    schedule.append((t, t + T, sign * amp))
                    schedule.append((t + T, t + 2 * T, -sign * amp))
                    t += 2 * T + self.rng.uniform(0.35, 0.7)

                def offset(tr, sch=schedule):
                    for t_a, t_b, val in sch:
                        if t_a <= tr < t_b:
                            return val
                    return 0.0

                wide = {"hit": False}

                def too_wide(s, c=circle, w=wide):
                    w["hit"] = abs(self._radius_error(c, s)) > TRANSIENT_MAX_RADIUS_ERROR
                    return w["hit"]

                yield from self._drive(circle, v=v, duration=5.0, steer_offset=offset, until=too_wide,
                                       name=f"{tag}_transients_v{v}_amp{amp:.2f}")
                if wide["hit"]:
                    print(f"SysId: {TRANSIENT_MAX_RADIUS_ERROR} m off the circle at amplitude {amp} and {v} m/s, "
                          f"skipping larger doublets at this speed")
                    yield from self._drive(circle, v=1.0, duration=6.0, name=f"{tag}_transients_recover",
                                           accel_command=lambda s: self._accel_to_command(-1.5)
                                           if s[LINEAR_VEL_X_IDX] > 1.3 else None,
                                           until=lambda s, c=circle: s[LINEAR_VEL_X_IDX] < 1.3
                                           and abs(self._radius_error(c, s)) < 0.1)
                    break
            self.radius_integral = 0.0

    def _limit_speed(self, r):
        """Speed of the grip limit on radius ``r``: measured by a staircase if one ran, else nominal."""
        a_lat = self.a_lat_limit if self.a_lat_limit is not None else self.mu_nominal * G
        return min(self.speed_max, math.sqrt(a_lat * r))

    # ----------------------------------------------------------- maneuvers
    def _drive(self, circle: Circle, v: float, laps: Optional[float] = None, duration: Optional[float] = None,
               steer_excitation: float = 0.0, name: str = "drive", accel_command=None, until=None,
               excitation_hold=(0.1, 0.5), steer_offset=None):
        """Track ``circle`` at speed ``v`` for ``laps`` laps or ``duration`` seconds.

        ``accel_command(state) -> command or None`` overrides the speed PI;
        ``until(state) -> bool`` ends the segment early.
        """
        self._set_phase(name)
        angle = 0.0
        last_phi = None
        t0 = self.t
        exc_value, exc_until = 0.0, -1.0
        while True:
            s = self.state
            phi = math.atan2(s[POSE_Y_IDX] - circle.cy, s[POSE_X_IDX] - circle.cx)
            if last_phi is not None:
                angle += circle.direction * _wrap(phi - last_phi)
            last_phi = phi
            if laps is not None and angle >= 2 * math.pi * laps:
                return
            if duration is not None and self.t - t0 >= duration:
                return
            if until is not None and until(s):
                return
            self._check_safety(circle)

            steer = self._circle_steer(circle)
            if steer_excitation > 0.0:
                if self.t >= exc_until:  # random steps, random hold 0.1-0.5 s: broadband up to ~5 Hz
                    exc_value = steer_excitation * self.rng.choice((-1.0, -0.5, 0.5, 1.0))
                    exc_until = self.t + self.rng.uniform(*excitation_hold)
                steer += exc_value
            if steer_offset is not None:  # callable(seconds since the segment start) -> steering added
                steer += steer_offset(self.t - t0)
            accel = accel_command(s) if accel_command is not None else None
            if accel is None:
                accel = self._speed_command(v)
            else:
                self.v_ref_filtered = float(s[LINEAR_VEL_X_IDX])
                self.speed_integral = 0.0
            yield float(np.clip(steer, -STEER_CMD_LIMIT, STEER_CMD_LIMIT)), float(accel)

    def _staircase(self, circle: Circle, name: str):
        """Hold speed steps of 0.25 m/s for 3 s each until the car cannot hold the radius."""
        v_cap = min(self.speed_max, math.sqrt(1.5 * self.mu_nominal * G * circle.r))
        v = 1.0
        wide_steps = 0

        def pushed_wide(s):
            nonlocal wide_steps
            wide_steps = wide_steps + 1 if self._radius_error(circle, s) > 0.12 else 0
            return wide_steps >= 4

        while v <= v_cap + 1e-6:
            yield from self._drive(circle, v=v, duration=3.0, name=f"{name}_v{v:.2f}", until=pushed_wide)
            if wide_steps >= 4:
                v_now = float(self.state[LINEAR_VEL_X_IDX])
                a_lat = v_now ** 2 / circle.r
                print(f"SysId: grip limit on R={circle.r:.2f} m at v={v_now:.2f} m/s (a_y ~ {a_lat:.1f} m/s^2)")
                if self.a_lat_limit is None or a_lat < self.a_lat_limit:
                    self.a_lat_limit = a_lat
                break
            v += 0.25
        # Gentle braking: hard braking would use up the grip the car needs to come back onto the circle
        yield from self._drive(circle, v=1.0, name=f"{name}_recover", duration=6.0,
                               accel_command=lambda s: self._accel_to_command(-1.5) if s[LINEAR_VEL_X_IDX] > 1.3 else None,
                               until=lambda s: s[LINEAR_VEL_X_IDX] < 1.3 and abs(self._radius_error(circle, s)) < 0.1)
        self.radius_integral = 0.0

    def _full_lock(self, direction: int, v: float, name: str):
        """One lap at full steering lock: the circle is the kinematic minimum radius."""
        self._set_phase(name)
        heading_turned, last = 0.0, float(self.state[POSE_THETA_IDX])
        while heading_turned < 2 * math.pi:
            th = float(self.state[POSE_THETA_IDX])
            heading_turned += direction * _wrap(th - last)
            last = th
            self._check_safety(None)
            yield direction * STEER_CMD_LIMIT, self._speed_command(v)

    def _pulses(self, circle: Circle, name: str, quick: bool):
        """Acceleration, braking and coasting between v_lo and v_hi on the circle.

        Acceleration pulses are offsets from the command that holds the speed, measured on the car
        right before (``u_hold``): the real drag (rolling, drivetrain, cornering) is unknown, and a
        fixed command smaller than the drag slows the car down instead of accelerating it. Braking
        pulses are absolute (brake + drag always decelerates); coasting is command 0.
        """
        v_lo, v_hi = 1.0, min(2.6, 0.75 * self._limit_speed(circle.r))
        v_mid = 0.5 * (v_lo + v_hi)
        held = []
        settle = self._drive(circle, v=v_mid, duration=4.0, name=f"{name}_hold")
        for cmd in settle:
            if self.t >= self._phase_start + 2.0:  # the speed PI has settled
                held.append(cmd[1])
            yield cmd
        u_hold = float(np.median(held)) if held else self._accel_to_command(self.c_rr * G)
        print(f"SysId: command to hold {v_mid:.2f} m/s on R={circle.r:.2f} m: {u_hold:.3f}")
        levels = ((1.0, -1.0), (4.0, -4.0)) if quick else ((1.0, -1.0), (2.5, -2.5), (4.0, -4.0), (2.0, 0.0))
        stall = {"n": 0}

        def reached_or_stalled(s):
            # accelerating but slower than v_lo - 0.3 for 1 s: the pulse is too weak, give up on it
            stall["n"] = stall["n"] + 1 if s[LINEAR_VEL_X_IDX] < v_lo - 0.3 else 0
            return s[LINEAR_VEL_X_IDX] >= v_hi or stall["n"] * self.dt >= 1.0

        for a_up, a_down in levels:
            for _ in range(2):
                stall["n"] = 0
                yield from self._drive(circle, v=v_lo, name=f"{name}_accel_{a_up}",
                                       accel_command=lambda s, a=a_up: u_hold + a / self.accel_mult,
                                       until=reached_or_stalled, duration=6.0)
                if self.state[LINEAR_VEL_X_IDX] < v_lo:  # stalled: get back to speed before the next pulse
                    yield from self._drive(circle, v=v_lo + 0.3, duration=3.0, name=f"{name}_recover")
                label = "coast" if a_down == 0.0 else f"brake_{a_down}"
                yield from self._drive(circle, v=v_lo, name=f"{name}_{label}",
                                       accel_command=lambda s, a=a_down: self._accel_to_command(a) if a else 0.0,
                                       until=lambda s: s[LINEAR_VEL_X_IDX] <= v_lo, duration=8.0)
        yield from self._drive(circle, v=1.0, duration=2.0, name=f"{name}_recover")

    def _transition(self, target: Circle, v: float):
        """Move from the current circle onto ``target``.

        Concentric circles: switch at once, the tracker converges radially.
        Otherwise: stay on the current circle until the common tangent point, drive the
        straight tangent segment, then join ``target``.
        """
        current = self._current_circle or target
        seg = tangent_segment(current, target)
        if seg is not None:
            p_a, p_b, t = seg
            clear = segment_clearance(self.arena, p_a, p_b)
            if clear < self.half_diagonal + 0.1:
                raise _Abort(f"no clear tangent from circle ({current.cx:.2f}, {current.cy:.2f}) "
                             f"to ({target.cx:.2f}, {target.cy:.2f}): clearance {clear:.2f} m")

            def at_tangent_point(s):
                d = np.array([s[POSE_X_IDX], s[POSE_Y_IDX]]) - p_a
                return np.hypot(*d) < 0.3 and d @ t > -0.05

            yield from self._drive(current, v=v, laps=1.5, until=at_tangent_point, name="transition_wait")
            yield from self._line(p_a, p_b, t, v=v, name="transition_line")
        yield from self._drive(target, v=v, duration=8.0, name="transition",
                               until=lambda s: abs(self._radius_error(target, s)) < 0.1
                               and math.cos(self._heading_error(target, s)) > 0.98)

    def _line(self, p_a, p_b, t, v: float, name: str):
        """Pure pursuit along the straight segment p_a -> p_b."""
        self._set_phase(name)
        length = float((p_b - p_a) @ t)
        while True:
            s = self.state
            p = np.array([s[POSE_X_IDX], s[POSE_Y_IDX]])
            along = float((p - p_a) @ t)
            if along >= length - 0.1:
                return
            self._check_safety(None)
            lookahead = float(np.clip(0.4 + 0.3 * abs(s[LINEAR_VEL_X_IDX]), 0.5, 1.2))
            q = p_a + t * min(along + lookahead, length + lookahead)
            dx, dy = q - p
            th = float(s[POSE_THETA_IDX])
            local_y = -math.sin(th) * dx + math.cos(th) * dy
            kappa = 2.0 * local_y / max(dx * dx + dy * dy, 1e-3)
            steer = math.atan(self.l_wb * kappa) / self.steer_mult
            yield float(np.clip(steer, -STEER_CMD_LIMIT, STEER_CMD_LIMIT)), self._speed_command(v)

    def _change_direction(self):
        """ccw main -> ccw lobe -> (cross at the touching point) -> cw lobe."""
        lobe_ccw, lobe_cw, _ = self.figure_eight
        yield from self._transition(lobe_ccw, v=1.0)
        yield from self._drive(lobe_ccw, v=1.0, laps=0.5, name="figure8_ccw_lobe")
        self.radius_integral = 0.0
        yield from self._transition(lobe_cw, v=1.0)
        self._set_raceline_direction(-1)
        yield from self._drive(lobe_cw, v=1.0, laps=0.5, name="figure8_cw_lobe")
        self.radius_integral = 0.0

    def _stop(self, circle: Circle, brake: float = 1.5, check: bool = True):
        self._set_phase("stop")
        while self.state[LINEAR_VEL_X_IDX] > 0.05:
            if check:
                self._check_safety(None)
            yield self._circle_steer(circle), self._accel_to_command(-brake)
        for _ in range(int(0.5 / self.dt)):
            yield 0.0, 0.0

    # ------------------------------------------------------------ control
    def _circle_steer(self, c: Circle) -> float:
        """Pure pursuit to a point on the circle plus an integral on the radius error."""
        self._current_circle = c
        s = self.state
        x, y, th, v = float(s[POSE_X_IDX]), float(s[POSE_Y_IDX]), float(s[POSE_THETA_IDX]), float(s[LINEAR_VEL_X_IDX])
        phi = math.atan2(y - c.cy, x - c.cx)
        e = math.hypot(x - c.cx, y - c.cy) - c.r
        lookahead = float(np.clip(0.4 + 0.3 * abs(v), 0.5, 0.9 * c.r))
        phi_t = phi + c.direction * lookahead / c.r
        dx = c.cx + c.r * math.cos(phi_t) - x
        dy = c.cy + c.r * math.sin(phi_t) - y
        local_y = -math.sin(th) * dx + math.cos(th) * dy
        kappa = 2.0 * local_y / max(dx * dx + dy * dy, 1e-3)
        if abs(e) < 0.4 and v > 0.5:
            self.radius_integral = float(np.clip(self.radius_integral + e * self.dt, -1.0, 1.0))
        kappa += c.direction * 0.6 * self.radius_integral
        return math.atan(self.l_wb * kappa) / self.steer_mult

    def _speed_command(self, v_ref: float) -> float:
        v = float(self.state[LINEAR_VEL_X_IDX])
        step = 1.5 * self.dt  # reference ramps at 1.5 m/s^2
        self.v_ref_filtered += float(np.clip(v_ref - self.v_ref_filtered, -step, step))
        err = self.v_ref_filtered - v
        self.speed_integral = float(np.clip(self.speed_integral + err * self.dt, -2.0, 2.0))
        a = 2.0 * err + 1.0 * self.speed_integral + self.c_rr * G
        return self._accel_to_command(float(np.clip(a, -3.0, 3.0)))

    def _accel_to_command(self, a: float) -> float:
        """Physical acceleration -> command, with the nominal multipliers."""
        return a / self.accel_mult if a >= 0.0 else a / self.brake_mult

    # ------------------------------------------------------------- safety
    def _check_safety(self, circle: Optional[Circle]):
        s = self.state
        x, y, th = float(s[POSE_X_IDX]), float(s[POSE_Y_IDX]), float(s[POSE_THETA_IDX])
        # Body corners
        hl, hw = self.half_diagonal * 0.83, self.half_diagonal * 0.55
        cx = x + math.cos(th) * np.array([hl, hl, -hl, -hl]) - math.sin(th) * np.array([hw, -hw, hw, -hw])
        cy = y + math.sin(th) * np.array([hl, hl, -hl, -hl]) + math.cos(th) * np.array([hw, -hw, hw, -hw])
        clearance = float(np.min(self.arena.clearance_at(cx, cy)))
        if clearance < self.abort_clearance:
            raise _Abort(f"wall clearance {clearance:.2f} m in phase {self.phase}")
        if circle is not None and self.phase not in ("transition",) and not self.phase.startswith("transition"):
            e = self._radius_error(circle, s)
            if abs(e) > self.abort_radius_error:
                raise _Abort(f"radius error {e:.2f} m in phase {self.phase}")

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _radius_error(c: Circle, s) -> float:
        return math.hypot(s[POSE_X_IDX] - c.cx, s[POSE_Y_IDX] - c.cy) - c.r

    @staticmethod
    def _heading_error(c: Circle, s) -> float:
        phi = math.atan2(s[POSE_Y_IDX] - c.cy, s[POSE_X_IDX] - c.cx)
        return _wrap(float(s[POSE_THETA_IDX]) - (phi + c.direction * math.pi / 2))

    def _set_raceline_direction(self, direction: int):
        """Load the map raceline for ``direction`` (<map>_wp.csv or <map>_wp_reverse.csv).

        Only the simulator's track bookkeeping and ``MPC_SAFETY_FILTER`` use the raceline; the
        filter works relative to it, so it has to point the way the car drives.
        """
        reverse = direction < 0
        if self.waypoint_utils is None or bool(Settings.REVERSE_DIRECTION) == reverse:
            return
        Settings.REVERSE_DIRECTION = reverse
        self.waypoint_utils.reset()
        self.waypoint_utils.reset_frenet_progress()
        if self.state is not None:
            self.waypoint_utils.update_next_waypoints(self.state)  # the driver reads the nearest index this step
        print(f"SysId: raceline switched to {'clockwise' if reverse else 'counter-clockwise'}")

    def _set_phase(self, name: str):
        self._phase_start = self.t
        if name != self.phase:
            self.phase = name
            self.phase_log.append((round(self.t, 2), name))
            print(f"SysId [{self.t:7.2f} s] {name}")
            if self.render_utils is not None:
                try:
                    self.render_utils.set_label_dict({"sysid": name})
                except Exception:
                    pass

    def _finish(self, reason: str):
        if not self.experiment_complete:
            print(f"SysId: {reason} after {self.t:.1f} s")
            self.experiment_complete = True


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
