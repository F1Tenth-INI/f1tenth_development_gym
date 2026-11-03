"""Car parameters from bench (non-driving) measurements.

Reads a measurement sheet with the *raw* readings (scale readings, string lengths,
stopwatch times, force-gauge readings, wheel angles, video samples; template:
``f1tenth_system/sysid/cars/TEMPLATE_bench.yml``, procedures:
``f1tenth_system/sysid/bench_measurements.md``), computes the model parameters and
writes a car parameter file. Keep the sheet in git: the parameters can then be
recomputed at any time.

    python -m Control_Toolkit_ASF.Controllers.SysId.bench_measurements \\
        --sheet ../../sysid/cars/yokomo1_bench.yml --out yokomo1_bench

Any field may be null (not measured): that parameter keeps its value from the base
file. A field may be a list of repeated readings: the mean is used and the spread printed.
At the end the script prints the matching ``fit_parameters`` command: parameters
measured reliably on the bench are left out of the driving fit.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time

import numpy as np
import yaml
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
CAR_FILES = os.path.normpath(os.path.join(HERE, "..", "..", "..", "utilities", "car_files"))
G = 9.81

# Driving-fit parameters, and which bench results replace them (fixed) or only start them (still fitted)
FIT_ALL = ["mu", "I_z", "B_f", "C_f", "B_r", "C_r", "servo_p", "s_max", "c_rr",
           "brake_multiplier", "accel_multiplier", "steering_multiplier"]
BENCH_FIXES = {"I_z", "C_f", "C_r", "servo_p", "s_max", "steering_multiplier"}


def val(x):
    """(mean, std, n) of a reading or list of readings; None if not measured."""
    if x is None:
        return None
    a = np.atleast_1d(np.asarray(x, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def mean(x):
    v = val(x)
    return None if v is None else v[0]


class Bench:
    def __init__(self, sheet: dict):
        self.s = sheet
        self.out = {}  # yaml key -> value
        self.lines = []  # report lines

    def set(self, key, value, how, spread=None):
        self.out[key] = float(value)
        sp = f" ±{spread:.3g}" if spread else ""
        self.lines.append(f"  {key:30s} {value:10.4g}{sp:10s} {how}")

    def note(self, text):
        self.lines.append(f"  {'':30s} {'':10s} {'':10s} {text}")

    def section(self, name):
        return self.s.get(name) or {}

    # ------------------------------------------------------------------ steps
    def run(self):
        geo = self.section("geometry")
        l_wb = mean(geo.get("wheelbase_m"))
        for key, field in (("width", "width_m"), ("length", "length_m"), ("imu_x", "imu_x_m"), ("imu_y", "imu_y_m")):
            v = val(geo.get(field))
            if v:
                self.set(key, v[0], "ruler", v[1])

        drive = self.section("drive")
        r_wheel = None
        n_rolls, d_rolls = mean(drive.get("wheel_rolls")), mean(drive.get("wheel_roll_distance_m"))
        if n_rolls and d_rolls:
            r_wheel = d_rolls / (2 * math.pi * n_rolls)
            self.set("wheel_radius", r_wheel, f"rolled {n_rolls:g} turns = {d_rolls:g} m")

        # Mass and CG
        mass = self.section("mass")
        m_v = val(mass.get("total_kg"))
        w_f, w_r = mean(mass.get("front_axle_kg")), mean(mass.get("rear_axle_kg"))
        m = m_v[0] if m_v else (w_f + w_r if w_f and w_r else None)
        if m:
            self.set("m", m, "scale" if m_v else "front + rear axle", m_v[1] if m_v else None)
        if w_f and w_r and l_wb:
            if m_v and abs((w_f + w_r) - m) > 0.02 * m:
                self.note(f"WARNING: axle loads sum to {w_f + w_r:.3f} kg, total is {m:.3f} kg (scale/blocks not level?)")
            share_f = w_f / (w_f + w_r)
            self.set("l_wb", l_wb, "ruler")
            self.set("lr", l_wb * share_f, f"axle loads: {100 * share_f:.1f} % on the front")
            self.set("lf", l_wb * (1 - share_f), "l_wb - lr")
        w_l, w_rt = mean(mass.get("left_side_kg")), mean(mass.get("right_side_kg"))
        track = mean(geo.get("track_m"))
        if w_l and w_rt and track:
            y = track * (w_l - w_rt) / (2 * (w_l + w_rt))
            self.note(f"CG {100 * y:+.1f} cm to the left of the centerline (model assumes 0)")

        # CG height
        cg = self.section("cg_height")
        lift, w_f_tilt = mean(cg.get("rear_lift_m")), mean(cg.get("front_axle_tilted_kg"))
        r_load = mean(cg.get("wheel_radius_loaded_m")) or r_wheel
        if lift and w_f_tilt and w_f and l_wb and m and r_load:
            theta = math.asin(lift / l_wb)
            h = r_load + l_wb * (w_f_tilt - w_f) / (m * math.tan(theta))
            self.set("h", h, f"rear lifted {lift:g} m ({math.degrees(theta):.0f} deg)")
        elif mean(cg.get("direct_m")):
            self.set("h", mean(cg.get("direct_m")), "measured / estimated directly")
        tip = mean(cg.get("tip_over_angle_deg"))
        if tip and track:
            self.note(f"tip-over check: h = {0.5 * track / math.tan(math.radians(tip)):.3f} m "
                      f"(upper bound: body roll lowers the tip angle)")

        # Yaw inertia (bifilar pendulum)
        yi = self.section("yaw_inertia")
        d, L, n = mean(yi.get("string_spacing_m")), mean(yi.get("string_length_m")), mean(yi.get("periods_counted"))
        t = val(yi.get("time_s"))
        T, T_rel = None, 0.0
        cum = yi.get("cumulative_times_s")  # stopwatch lap button at every full period: slope = period
        if cum and len(cum) >= 3:
            c = np.asarray(cum, float)
            k = np.arange(1, len(c) + 1)
            T, b = np.polyfit(k, c, 1)
            res = c - (T * k + b)
            T_rel = float(np.sqrt(np.sum(res ** 2) / (len(c) - 2) / np.sum((k - k.mean()) ** 2))) / T
        elif n and t:
            T = t[0] / n
            T_rel = t[1] / t[0] if t[1] else 0.0
        if d and L and T and m:
            m_fix = mean(yi.get("fixture_mass_kg")) or 0.0
            I_tot = (m + m_fix) * G * d * d * T * T / (16 * math.pi ** 2 * L)
            I_fix = 0.0
            t_fix, n_fix = mean(yi.get("fixture_time_s")), mean(yi.get("fixture_periods_counted"))
            if m_fix and t_fix and n_fix:
                I_fix = m_fix * G * d * d * (t_fix / n_fix) ** 2 / (16 * math.pi ** 2 * L)
            I_z = I_tot - I_fix
            rel = 2 * T_rel  # I ~ T^2 (stopwatch only; d and L errors come on top: 1 mm on d = 1.4 %)
            self.set("I_z", I_z, f"bifilar: T = {T:.4f} s, d = {d:g} m, L = {L:g} m" +
                     (f", fixture {I_fix:.4f} subtracted" if I_fix else ""), I_z * rel)

        # Steering gain and lock
        st = self.section("steering")
        if st.get("commands_rad") and (st.get("wheel_angle_left_deg") or st.get("wheel_angle_right_deg")):
            self.steering(st)
        if l_wb:
            for side in ("left", "right"):
                dia = mean(st.get(f"turning_circle_diameter_{side}_m"))
                if dia:
                    self.note(f"full lock {side}: rear axle circle R = {dia / 2:.3f} m -> "
                              f"{math.degrees(math.atan(l_wb / (dia / 2))):.1f} deg = {math.atan(l_wb / (dia / 2)):.4f} rad")
            dl, dr = mean(st.get("turning_circle_diameter_left_m")), mean(st.get("turning_circle_diameter_right_m"))
            if dl:  # more accurate than the plateau of the photos, so it overrides it
                self.set("s_max", math.atan(2 * l_wb / dl), "turning circle at full lock left")
            if dr:
                self.set("s_min", -math.atan(2 * l_wb / dr), "turning circle at full lock right")

        # Servo speed
        sv = self.section("servo")
        if sv.get("steps") or (sv.get("samples_t_s") and sv.get("samples_angle_deg")):
            self.servo(sv)

        # Friction
        fr = self.section("friction")
        peak, slide = val(fr.get("lateral_peak_N")), val(fr.get("lateral_slide_N"))
        if peak and m:
            mu = peak[0] / (m * G)
            self.set("mu", mu, "sideways pull, breakaway force / weight (start value, driving fit refines)", peak[1] / (m * G))
            if slide:
                ratio = min(slide[0] / peak[0], 1.0)
                c = 2.0 - (2.0 / math.pi) * math.asin(ratio)  # Pacejka E=0: F(inf)/F(peak) = sin(C pi/2), C in [1, 2]
                self.set("C_f", c, f"sliding / breakaway = {ratio:.3f}  ->  sin(C pi/2)")
                self.set("C_r", c, "same pull test (whole car)")
        spin = val(fr.get("longitudinal_spin_N"))
        if spin and m:
            self.note(f"longitudinal mu (wheelspin against the gauge) = {spin[0] / (m * G):.3f}")

        # Rolling resistance
        rr = self.section("rolling")
        push = val(rr.get("push_force_N"))
        incline = mean(rr.get("incline_angle_deg"))
        if push and m:
            self.set("c_rr", push[0] / (m * G), "slow push with the drivetrain connected, force / weight", push[1] / (m * G))
        elif incline:
            self.set("c_rr", math.tan(math.radians(incline)), f"keeps rolling on a {incline:g} deg incline")

        # Drivetrain
        D, t_run, erpm = mean(drive.get("erpm_distance_m")), mean(drive.get("erpm_time_s")), mean(drive.get("erpm_mean"))
        if D and t_run and erpm:
            self.set("speed_to_erpm_gain", erpm / (D / t_run), f"{erpm:g} ERPM at {D / t_run:.2f} m/s (also vesc.yaml)")
        I, F = drive.get("traction_current_A"), drive.get("traction_force_N")
        gain = mean(drive.get("acceleration_to_current_gain"))
        if I and F and m:
            I, F = np.asarray(I, float), np.asarray(F, float)
            k = float(np.sum(I * F) / np.sum(I * I))  # N per A through the origin
            self.note(f"traction: {k:.4f} N/A ({k / m:.4f} m/s^2 per A)")
            if gain:
                self.set("accel_multiplier", k * gain / m,
                         f"stall pull: {k:.3f} N/A x acceleration_to_current_gain {gain:g} / m (start value)")

    def steering(self, st):
        u = np.asarray(st["commands_rad"], float)
        sides = [np.asarray(st[k], float) for k in ("wheel_angle_left_deg", "wheel_angle_right_deg") if st.get(k)]
        delta = np.radians(np.mean(sides, axis=0))  # bicycle model: mean of both front wheels
        if len(sides) == 2:
            outer_inner = np.abs(np.radians(sides[0] - sides[1]))
            self.note(f"left/right wheel angle difference up to {math.degrees(outer_inner.max()):.1f} deg (Ackermann)")

        def model(p):
            k, off, lo, hi = p
            return np.clip(k * u + off, lo, hi)

        res = least_squares(lambda p: model(p) - delta,
                            [1.0, 0.0, min(delta.min(), -0.05) - 0.01, max(delta.max(), 0.05) + 0.01])
        k, off, lo, hi = res.x
        raw = k * u + off
        self.set("steering_multiplier", k, f"wheel angle / command, {len(u)} points")
        # Each side separately: the lock (mechanical, or the servo_min/max clip in vesc.yaml) is often asymmetric
        if np.any(raw > hi + 1e-3):
            self.set("s_max", hi, f"left plateau of the wheel angle ({math.degrees(hi):.1f} deg)")
        if np.any(raw < lo - 1e-3):
            self.set("s_min", lo, f"right plateau of the wheel angle ({math.degrees(lo):.1f} deg)")
        if not np.any(raw > hi + 1e-3) or not np.any(raw < lo - 1e-3):
            self.note("no plateau on one side: command further (e.g. to +-0.5) until the wheel stops moving")
        if abs(off) > math.radians(0.3):
            self.note(f"STEERING OFFSET {math.degrees(off):+.2f} deg at command 0: the model has none. Fix it in vesc.yaml: "
                      f"steering_angle_to_servo_offset -= {off / k:.4f} * steering_angle_to_servo_gain")

    def servo(self, sv):
        """Joint fit of all step videos: servo_p and sv_max shared, a start time per step.

        A large step mostly shows the rate limit (sv_max); a small one (no rate limit) shows servo_p.
        """
        steps = sv.get("steps") or [sv]
        data = []
        for st in steps:
            t = np.asarray(st["samples_t_s"], float)
            a = np.radians(np.asarray(st["samples_angle_deg"], float))
            target = st.get("target_angle_deg")
            data.append((t, a, a[0], math.radians(float(target)) if target is not None else a[-1], st.get("command_time_s")))

        def simulate(servo_p, sv_max, t0, t, a0, a1):
            dt = 1e-4  # the response starts exactly at t0, so the fit is smooth in t0
            tt = t0 + np.arange(0.0, max(t.max() - t0, 0.0) + 2 * dt, dt)
            d = np.empty_like(tt)
            x = a0
            for i in range(len(tt)):
                d[i] = x
                x += dt * float(np.clip(servo_p * (a1 - x), -sv_max, sv_max))
            return np.interp(t, tt, d, left=a0)

        def residual(p):
            return np.concatenate([simulate(p[0], p[1], p[2 + i], t, a0, a1) - a
                                   for i, (t, a, a0, a1, _) in enumerate(data)])

        lo = [1.0, 0.5] + [t.min() - 0.1 for t, *_ in data]
        hi = [200.0, 50.0] + [t.max() for t, *_ in data]
        best = None
        for frac in np.linspace(0.0, 0.5, 6):
            x0 = [20.0, 5.0] + [t.min() + frac * (t.max() - t.min()) for t, *_ in data]
            r = least_squares(residual, x0, bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
        servo_p, sv_max = best.x[:2]
        sizes = ", ".join(f"{math.degrees(a1 - a0):+.0f}" for _, _, a0, a1, _ in data)
        self.set("servo_p", servo_p, f"step videos ({sizes} deg), 1st order + rate limit")
        self.set("sv_max", sv_max, "max steering rate, same videos")
        self.set("sv_min", -sv_max, "= -sv_max")
        if all(abs(a1 - a0) * servo_p > sv_max for _, _, a0, a1, _ in data):
            self.note("all steps hit the rate limit: servo_p is weakly determined, add a small step (~3 deg)")
        for i, (*_, t_cmd) in enumerate(data):
            if t_cmd is not None:
                self.note(f"servo dead time after the command (step {i + 1}): {best.x[2 + i] - float(t_cmd):.3f} s")


def write_car_file(base_file, out_file, values: dict, source: str):
    lines = open(os.path.join(CAR_FILES, base_file)).read().splitlines()
    tyre = {"C_f": ("C_Pf", 1), "C_r": ("C_Pr", 1)}
    for key, value in values.items():
        if key in tyre:
            block, pos = tyre[key]
            start = next(i for i, l in enumerate(lines) if l.startswith(block + ":"))
            lines[start + 1 + pos] = f"  - {value:.4g}  # {key}, bench"
            continue
        for i, l in enumerate(lines):
            if re.match(rf"^{key}\s*:", l):
                lines[i] = f"{key}: {value:.5g}  # bench"
                break
        else:
            lines.append(f"{key}: {value:.5g}  # bench")
    header = f"# Bench measurements {time.strftime('%Y-%m-%d %H:%M')} from {source}, base {base_file}\n"
    with open(os.path.join(CAR_FILES, out_file), "w") as f:
        f.write(header + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True, help="measurement sheet (yaml)")
    ap.add_argument("--base", default=None, help="car file to start from (default: base_parameter_file in the sheet)")
    ap.add_argument("--out", default=None, help="writes utilities/car_files/<out>_car_parameters.yml")
    args = ap.parse_args()

    sheet = yaml.safe_load(open(args.sheet))
    base = args.base or sheet.get("base_parameter_file") or "gym_car_parameters.yml"
    b = Bench(sheet)
    b.run()
    print(f"car: {sheet.get('car')}   measured: {sheet.get('date')}   base: {base}\n")
    print("\n".join(b.lines))

    if args.out:
        out_file = f"{args.out}_car_parameters.yml"
        write_car_file(base, out_file, b.out, os.path.basename(args.sheet))
        print(f"\nwrote utilities/car_files/{out_file}")
        fixed = [p for p in FIT_ALL if p in BENCH_FIXES and p in b.out]
        fit = [p for p in FIT_ALL if p not in fixed]
        print("\ndriving fit, with the bench values fixed:" if fixed else "\ndriving fit:")
        print(f"  python3 -m Control_Toolkit_ASF.Controllers.SysId.fit_parameters --csv <recordings> "
              f"--car-params {out_file} --params {','.join(fit)} --out <name>")
        if fixed:
            print(f"  (fixed from the bench: {', '.join(fixed)}; mu, c_rr, accel_multiplier are start values and still fitted)")


if __name__ == "__main__":
    main()
