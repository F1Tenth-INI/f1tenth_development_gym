# Car dynamics model (`dynamic_model_pacejka_jax.py`)

This is the vehicle model that moves the simulated car. It is used by

- the simulator itself (`base_classes.py`, `Settings.SIM_ODE_IMPLEMENTATION = 'jax_pacejka'`), stepped at `Settings.TIMESTEP_SIM = 0.01 s`, and
- the JAX planners (RPGD / MPPI via `car_model_jax.py`) for their predictions.

The variant is chosen with `Settings.ODE_MODEL_OF_CAR_DYNAMICS`:

| `ode_model` | What it is | When to use |
|---|---|---|
| `ODE:ks_pacejka` (default) | Dynamic single-track model with Pacejka tyres, blended into a kinematic model at low speed | Normal use |
| `ODE:pacejka` | Dynamic single-track model only | Analysis at speeds above about 1 m/s. Not reliable near standstill (see Limitations) |
| `ODE:ks` | Kinematic bicycle only | Very slow driving. It has no grip limit |

The car parameters come from `utilities/car_files/<car>.yml` (`VehicleParameters.to_np_array()`). For the Yokomo car they were identified from real-car recordings (see [Parameter identification](#parameter-identification)).

---

## 1. State, inputs, coordinate frame

State vector (10 entries, same order as `utilities/state_utilities.py`):

| idx | name | meaning |
|---|---|---|
| 0 | `angular_vel_z` (ψ̇ or r) | yaw rate [rad/s] |
| 1 | `linear_vel_x` (v_x) | longitudinal velocity of the CoG, body frame [m/s] |
| 2 | `linear_vel_y` (v_y) | lateral velocity of the CoG, body frame [m/s] |
| 3 | `pose_theta` (ψ) | heading [rad], wrapped to [-π, π] |
| 4, 5 | cos ψ, sin ψ | redundant, for learning / planners |
| 6, 7 | `pose_x`, `pose_y` | CoG position, world frame [m] |
| 8 | `slip_angle` (β) | atan(v_y / v_x), output only |
| 9 | `steering_angle` (δ) | actual front wheel angle [rad] (a state, not the command) |

Control input `[desired_steering_angle, translational_control]`:
- desired steering angle: what the controller sends to the servo [rad].
- translational control: commanded longitudinal acceleration [m/s²] (this is the default, `Settings.MOTOR_PID_IN_CAR_MODEL = False`). If `MOTOR_PID_IN_CAR_MODEL = True` it is a desired speed, tracked by a PI loop.

Body frame: x points forward, y points left, and yaw is positive counter-clockwise.

The control delay (`Settings.CONTROL_DELAY`) is **not** part of this model. The simulator applies it before calling the model.

---

## 2. Processing pipeline per sub-step

```
desired steering ──► × steering_multiplier ──► P-servo (servo_p, rate limit sv_min/max) ──► δ̇ ──► δ clipped to [s_min, s_max]

accel command ──► brake/accel multiplier ──► power + speed limits (a_min, a_max·v_switch/v, v_min, v_max)
              ──► + rolling resistance ──► clip to ±μg ──► a_x

(δ, δ̇, a_x, state) ──► kinematic step ─┐
                    └─► dynamic step ──┴─► blend by speed ──► next state
```

### 2.1 Actuators

**Steering.** The servo is a first-order proportional loop with a rate limit:
δ̇ = clip(k_servo · (k_steer · δ_cmd − δ), sv_min, sv_max). The angle δ is limited to the steering lock [s_min, s_max].
- `steering_multiplier` (k_steer) calibrates the controller's command against the real bicycle-equivalent wheel angle. It covers the servo linkage, Ackermann geometry and servo trim.
- `s_max` is the effective lock: the largest bicycle-equivalent angle the real car reaches.

**Longitudinal.** The commanded acceleration is first scaled by `accel_multiplier` (for a > 0) or `brake_multiplier` (for a < 0). The switch between the two is a smooth tanh, so the model stays differentiable. These are ESC calibration factors: the real ESC converts the command into motor current, and the resulting acceleration is not equal to the command. Then come the classic CommonRoad limits:
- the power limit a ≤ a_max·v_switch/v above v_switch,
- the speed limits v_min / v_max,
- the a_min / a_max limits,
- the friction limit |a| ≤ μg.

**Rolling resistance:** a_rr = −c_rr · g · sign(v_x). It uses a smooth sign function, v/√(v² + v_dead²), so the car comes to rest instead of oscillating around v = 0.

### 2.2 Dynamic single-track model (`_pacejka_step`)

This is the standard single-track ("bicycle") model with nonlinear tyres [Rajamani 2012, ch. 2; Althoff et al., CommonRoad vehicle models, "ST" model]:

```
ṡ_x = v_x cos ψ − v_y sin ψ
ṡ_y = v_x sin ψ + v_y cos ψ
ψ̇   = r
v̇_x = a_x − F_yf sin δ / m + v_y r
v̇_y = (F_yr + F_yf cos δ) / m − v_x r
ṙ   = (l_f F_yf cos δ − l_r F_yr) / I_z
δ̇   = servo output (2.1)
```

- `a_x` is the longitudinal tyre acceleration from 2.1. It is applied along the body x-axis (4WD, so it is shared between the axles).
- `F_yf sin δ` is the part of the front lateral force that points backwards when the wheels are steered. This is the physical cornering drag. Because of it, `curve_resistance_factor` (an empirical extra drag term) is 0 for the fitted car.
- The terms `+v_y r` and `−v_x r` come from expressing Newton's law in a rotating body frame. They rotate the velocity vector and do no work.

**Slip angles** (small-angle form, valid in both driving directions):

```
α_f = sign(v_x) · δ − atan((v_y + l_f r) / |v_x|)
α_r =              − atan((v_y − l_r r) / |v_x|)
```

These angles have the sign that makes each tyre force oppose the sliding velocity of its wheel. Using |v_x| and sign(v_x)·δ keeps this true when reversing: a forward-only formula would produce forces that push the slide instead of resisting it. |v_x| is floored at 1 mm/s so the division is always defined.

**Vertical loads with longitudinal load transfer** (CoG height h):

```
F_zf = m (g l_r − a_x h) / (l_f + l_r)
F_zr = m (g l_f + a_x h) / (l_f + l_r)
```

**Lateral tyre force: Pacejka "Magic Formula"** [Pacejka 2012, ch. 4]:

```
F_y = μ · D · F_z · sin(C · atan(B α − E (B α − atan(B α))))
```

- D·μ is the peak friction coefficient (D = 1, so μ is the peak).
- C sets the shape after the peak. The force at very large slip is sin(C·π/2) of the peak.
- B·C·D·μ·F_z is the cornering stiffness C_α, the slope at α = 0.
- E = 0 for this car.

**Friction ellipse (combined slip).** A tyre can only transfer a total force of about μF_z. Whatever is used to accelerate or brake is not available for cornering [Milliken & Milliken 1995, ch. 2; Pacejka 2012, ch. 4.3]. The model uses the common "longitudinal priority" approximation of the friction ellipse:

```
F_y ← F_y · sqrt(1 − (a_x / (μ g))²)
```

On the real car, front and rear wheel speeds are identical under acceleration and braking (ratio 0.9995). It is a 4WD car with locked drive, so the drive force is shared roughly in proportion to axle load. Each axle then uses the same fraction a_x/(μg) of its grip, and the same derating factor applies to both. This is the same approach as the friction-ellipse models used in RC-car racing MPC [Liniger et al. 2015].

### 2.3 Kinematic bicycle (`_ks_step`)

At low speed the tyre slip angles become ill-defined (division by v_x), and the tyre equations become numerically stiff. The model then uses the kinematic bicycle [Rajamani 2012, ch. 2.2; Kong et al. 2015], referenced at the **CoG** so that it describes the same point as the dynamic model:

```
v   = sign(v_x) · sqrt(v_x² + v_y²)          (speed is the state)
β   = atan(l_r tan δ / L)                     (CoG slip angle, set by the steering)
ṡ_x = v cos(ψ + β),   ṡ_y = v sin(ψ + β)
ψ̇   = v cos β · tan δ / L
v̇   = a_x
v_x = v cos β,  v_y = v sin β,  r = v_x tan δ / L     (algebraic outputs)
```

- The speed v is the only velocity state. Steering only turns its direction, so steering never changes the kinetic energy.
- v_x, v_y and r are algebraic: they follow directly from v and δ and are not integrated. This means the kinematic model never keeps a stale yaw rate or lateral velocity.

### 2.4 Blending (`ODE:ks_pacejka`)

Each sub-step computes both models from the same state and mixes them. The kinematic model is valid when the car is **slow and its wheels are not sliding sideways** [Kong et al. 2015]. The weight therefore combines both conditions:

```
|v|      = sqrt(v_x² + v_y²)
slide    = max(|v_lat,f|, |v_lat,r|),  v_lat,f = (v_y + l_f r) cos δ − v_x sin δ,  v_lat,r = v_y − l_r r
w_speed  = 1 / (1 + exp(−4.8 (|v| − 1.75)))
w_slide  = 1 / (1 + exp(−10 (slide − 0.5)))
w        = 1 − (1 − w_speed)(1 − w_slide)
x_next   = (1 − w) · x_kinematic + w · x_dynamic      (heading blended on the unit circle)
```

- **Normal driving** (wheel sliding ≈ 0): the result is effectively kinematic below 1 m/s (w < 3%) and effectively dynamic above 2.5 m/s (w > 97%).
- **Sliding.** v_lat,f and v_lat,r are the sideways sliding speeds of the wheels, which the kinematic model assumes to be zero. If either exceeds about 0.5 m/s, the dynamic model takes over at any speed. A slide therefore decelerates at the physical rate (about μg) and is not stopped within one step. A blend on v_x alone used to stop sideways slides at 1–2.5 m/s at up to 360 m/s².
- **Total speed |v|** (not v_x) is used because tyre stiffness, and with it the integration limit of section 2.5, scales with 1/|v|. A car sliding sideways at 5 m/s is well within the valid range of the dynamic model. Fast reversing likewise uses the grip-limited dynamic model rather than the kinematic one.
- **No jumps.** Both models describe the CoG and agree in steady low-speed cornering, so the transition is smooth.

A lower speed threshold (kinematic only below about 0.3 m/s) was tested as well. The data at 1–2 m/s then pushed both tyres to their stiffness bounds (B·C ≈ 36), so in that range the kinematic model describes the real car better than a dynamic model with plausible tyres.

### 2.5 Integration

Explicit Euler with `dt = 0.01 s` (one call per simulator step, `intermediate_steps = 1`).

- **Stability limit.** The fastest tyre mode has an eigenvalue of roughly λ ≈ (l_f²C_αf + l_r²C_αr)/(I_z v) ≈ 49/v [1/s] with the fitted parameters, and Euler needs |λ·dt| < 2. The spectral radius of the one-step Jacobian of (r, v_y), computed numerically, confirms this: 0.94 at 0.25 m/s, 0.62 at 0.3 m/s, < 0.8 above. The dynamic model is therefore stable for |v| ≳ 0.25 m/s.
- **Saturated slides are not stiff.** When the tyres are saturated (large slip), the force hardly depends on the slip angle any more, and the spectral radius is ≈ 1.00 even at v_x → 0.
- **The blend keeps the step stable.** The kinematic part is algebraic, so the blended step is roughly w · (spectral radius of the dynamic step). With w_slide only active for slides of ≥ 0.5 m/s, that stays well below 1.

---

## 3. Why the model is physically sound

1. **It cannot create energy.** For zero drive force (a_x = 0) the kinetic energy E = ½m(v_x² + v_y²) + ½I_z r² changes as

   ```
   dE/dt = F_yf · v_lat,f + F_yr · v_lat,r ,
   v_lat,f = (v_y + l_f r) cos δ − v_x sin δ ,   v_lat,r = v_y − l_r r
   ```

   v_lat,f and v_lat,r are the sideways sliding velocities of the wheels. The rotation terms (v_y r, −v_x r) cancel. Each tyre force has the opposite sign to its wheel's sliding velocity (slip-angle sign, and C < 2 so the Magic Formula keeps its sign). Both terms are therefore ≤ 0: sliding can only dissipate energy. The previous version left out `+v_y r` and the `sin δ / cos δ` projections. That broke this property and let slides escalate without bound (see the table below).
2. **Accelerations are bounded by friction.** |a_x| ≤ μg, and the friction ellipse limits the combined horizontal tyre force to about μmg. So the total horizontal acceleration cannot exceed what the tyres can transmit.
3. **Correct low-speed limit.** For small slip and low speed the dynamic model reduces to the kinematic bicycle, so blending the two is consistent [Kong et al. 2015].
4. **Valid in reverse.** The slip-angle formulation of 2.2 keeps the tyre forces dissipative for v_x < 0.

**Stress test** (`AnalyseData/DynamicsModelFit/stress_test.py`). It simulates 300 random initial states: slides with |v_y| ≤ 3 m/s, |r| ≤ 8 rad/s, −5 ≤ v_x ≤ 8 m/s, random steering, each coasting or at ±5 m/s² for 3 s. "Before" is the previous model with the previous parameters. The horizontal acceleration is |dV/dt| of the CoG velocity in the world frame.

| | before | now |
|---|---|---|
| coasting runs gaining > 1% energy (no resistances) | 67 / 148 | 5 / 148 |
| worst coasting energy gain | 29 180 % | 2.7 % (Euler error at r ≈ 8 rad/s) |
| max \|v_y\| reached | 127 m/s | 8.1 m/s (initial speed, rotated into v_y) |
| horizontal acceleration, 95th pct / max over runs | 168 / 736 m/s² | 20 / 31 m/s² (see Limitations) |
| full throttle on full lock, max horizontal acceleration | 8.8 m/s² (= μg) | 5.3 m/s² (≤ μg = 5.9) |
| NaN | 0 | 0 |

---

## 4. Parameter identification

Parameters for `yokomo1_car_parameters.yml` were identified on 2026-10-28 from real-car recordings in `AnalyseData/Rosbags/2026-10-26/` (IPZ50 track, RPGD controller). The scripts and a detailed description of the method are in [`AnalyseData/DynamicsModelFit/`](../../../AnalyseData/DynamicsModelFit/README.md).

| recording | content | used for |
|---|---|---|
| `rosbag_20261026_091934_rpgd_sweep` | speed sweep, 0–4.7 m/s, up to 8 m/s² lateral | fit + validation |
| `rosbag_20261026_093432_rpgd_sweep_reverse` | same, other direction | fit + validation |
| `2026-10-26_09-31-34_.csv` (`reverse_0.5_slow`) | slow driving, about 1.5 m/s | validation only (never fitted) |

### 4.1 Which signals were trusted

| signal | check | used |
|---|---|---|
| `linear_vel_x` (wheel speed × r) | matches speed from the scan-matched pose (ratio 1.00) | yes |
| `angular_vel_z` (gyro) | matches pose yaw rate (ratio 1.00), noise ≈ 0.06 rad/s | yes |
| `pose_x/y/theta` (scan matcher) | | yes |
| `v_x_ekf` | about 20% too low, lagging | no |
| `linear_vel_y` | uncorrelated with the pose-derived v_y | no (initialised kinematically instead) |
| `*_control_executed` | the commands actually sent to the car | inputs |

### 4.2 Method

- **Windows.** About 4000 windows of 1 s (25 samples). Each starts from the measured state (r, v_x, pose). The servo state δ is pre-rolled from the previous 0.4 s of commands, and v_y is initialised kinematically. Each window is then simulated **open-loop** with the executed commands, through the same delay buffer and 0.01 s step as the simulator.
- **Loss.** Huber loss on yaw-rate, speed, heading and position errors over the whole horizon. Windows with wall contact (LiDAR < 0.15 m) or out-of-range servo commands were removed.
- **Optimiser.** Differential evolution (global) followed by Powell refinement, within physically plausible bounds (e.g. Pacejka C ∈ [1, 1.8]).
- **Train/validation split.** Interleaved 10 s blocks: 3 of every 4 blocks of the two sweeps for fitting, the 4th for validation, and the slow run as a fully separate test.

### 4.3 Model-independent checks

These come straight from the data, without any model, and support the fitted values:

- **Steering gain.** In quasi-steady cornering at low speed, the path curvature r/v ≈ 1.6 · tan(δ_cmd)/L for small commands. The gain falls for larger commands, so the real bicycle-equivalent wheel angle saturates at about 0.25–0.32 rad. This is why `s_max` = 0.25 (not 0.6).
- **Understeer.** Curvature drops with speed. A fit κ = c·δ/(L(1 + K v²)) gives K ≈ 0.067 s²/m², i.e. about 38% less curvature than kinematic at 3 m/s.
- **Grip.** Steady lateral acceleration v·r levels off at about 5–6 m/s² at high speed, so μ ≈ 0.6.
- **Drivetrain.** Rear/front wheel speed ratio is 0.9995 at all accelerations, so the car is 4WD with locked drive (basis of the friction ellipse in 2.2).
- **Longitudinal.** A regression gives a ≈ 1.29·u⁺ − 1.4 m/s² for u > 0, and much stronger deceleration than commanded for u < 0. This matches `accel_multiplier`, `c_rr` and `brake_multiplier`.

### 4.4 Identified values

| parameter | old | new | justification |
|---|---|---|---|
| `mu` | 0.9 | **0.6** | fit; steady lateral acceleration levels off at ≈ 5.5–6 m/s² |
| `s_min/s_max` | ±0.6 | **±0.25** | fit (clearly determined: 0.30 or 0.35 raise the loss by 11% / 36%); curvature saturation in 4.3 |
| `steering_multiplier` | 1.6 | **1.49** | fit; low-speed steering gain ≈ 1.5–1.6 in 4.3 |
| `servo_p` | 10 | **34** | fit; turn-in is limited mainly by the rate limit `sv_max` = 3.2 rad/s |
| `I_z` | 0.07 | **0.03** | fit (only loosely determined, 0.02–0.03 equally good); radius of gyration 0.10 m, plausible for a 2.9 kg car of 0.38 × 0.25 m |
| `C_Pf` (B, C, D, E) | 4.45, 1.3, 1, 0 | **4.7, 1.0, 1, 0** | fit; C = 1 means the front force has no peak and approaches μF_z asymptotically (front-limited, understeering car) |
| `C_Pr` (B, C, D, E) | 9.9, 1.1, 1, 0 | **4.85, 1.3, 1, 0** | fit gave C_r = 1.75, B_r = 3.6. C_r was set to 1.3 with B_r chosen to keep the same cornering stiffness: the data never reach the rear peak (α ≈ 20°), so the fall-off after the peak is not identified. C = 1.3 is the textbook value and keeps 89% of peak grip at large slip instead of 38%. Cost: +0.4° heading error at 0.5 s |
| `c_rr` | 0.12 | **0.10** | fit; consistent with the longitudinal regression |
| `curve_resistance_factor` | 0.4 | **0.0** | fit; the physical `F_yf sin δ` term covers cornering drag |
| `accel_multiplier` | 1.0 | **1.23** | fit; regression slope 1.29 |
| `brake_multiplier` | 2.0 | **6.9** | fit; the ESC brakes much harder than the command (deceleration is then limited to μg) |

Resulting cornering stiffnesses (static loads): C_αf ≈ 38 N/rad, C_αr ≈ 56 N/rad. The linear understeer gradient is K_us = (m/L)(l_r/C_αf − l_f/C_αr) ≈ 0.009 rad/(m/s²). This is about half the rough model-free estimate (0.017). In the model, additional understeer comes from the front tyre saturating and from the friction ellipse under throttle.

### 4.5 Validation (open-loop prediction from measured states, held-out data)

RMSE after 0.5 s / 1.0 s; "before" = previous model with the previous parameters.

| data | model | yaw rate [rad/s] | v_x [m/s] | heading [°] | position [m] |
|---|---|---|---|---|---|
| held-out sweep blocks | before | 0.59 / 0.50 | 0.53 / 0.88 | 14.9 / 24.5 | 0.24 / 0.70 |
| | **now** | **0.26 / 0.29** | **0.27 / 0.34** | **6.3 / 10.2** | **0.15 / 0.34** |
| slow run (never fitted) | before | 0.19 / 0.30 | 0.30 / 0.52 | 3.9 / 7.5 | 0.10 / 0.30 |
| | **now** | **0.17 / 0.17** | **0.16 / 0.18** | 4.2 / 8.5 | 0.11 / **0.27** |

- Fit and validation errors are equal, so there is no overfitting.
- Results with `CONTROL_DELAY` 0.04 s and 0.08 s are nearly identical. The table uses 0.04 s.
- For reference, assuming the yaw rate stays constant ("persistence") gives 1.1 rad/s error after 0.5 s.

---

## 5. Limitations and known gaps

- **Remaining error.** The yaw-rate error of about 0.25 rad/s comes mostly from fast transients in the aggressive sweeps. The servo and ESC are only modelled as a first-order lag plus a fixed delay.
- **Residual slides near standstill.** A residual slide below the 0.5 m/s threshold, at speeds of about 1 m/s or less, is removed by the kinematic model within 2–3 steps instead of about 0.07 s. The stress test shows this as accelerations of up to about 30 m/s² (5 μg) over a velocity change of only about 0.3 m/s. The effect on the pose is a few millimetres.
- **Pure `ODE:pacejka` near standstill.** Below about 0.3 m/s the explicit Euler step is not stable (see 2.5), and slip angles saturate at ±90°. Use `ODE:ks_pacejka`.
- **No lateral load transfer, roll or wheel dynamics.** Longitudinal tyre slip is not modelled: a_x is applied directly, limited by μg.
- **Not identified by the data.** The tyre behaviour beyond the peak, and very hard braking (only a few samples with u < −0.5).
- **Speed PI loop.** The PI loop (`MOTOR_PID_IN_CAR_MODEL = True`) resets its integral at every call, so in the simulator it acts as a P controller. It is inactive by default.
- **Other models not updated.** `SI_Toolkit_ASF/car_model.py` (`ODE_TF`) and `Control_Toolkit_ASF/Controllers/MPCAcados/model.py` were not updated to the equations above. `test_ode_tf_vs_jax_pacejka.py`, which compares against `car_model.py`, therefore fails.
- **Fitted for one car and surface.** The values are for the Yokomo car on the IPZ50 floor. Other surfaces need a new μ, either by refitting or with `Settings.SURFACE_FRICTION`, which overrides μ.

---

## References

- R. Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012. Single-track model and kinematic bicycle (ch. 2).
- H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., Butterworth-Heinemann, 2012. Magic Formula, combined slip.
- W. F. Milliken, D. L. Milliken, *Race Car Vehicle Dynamics*, SAE, 1995. Friction circle/ellipse, load transfer.
- M. Althoff, M. Koschi, S. Manzinger, "CommonRoad: Composable benchmarks for motion planning on roads", IEEE IV, 2017, and the *CommonRoad Vehicle Models* documentation. ST/KS models, acceleration and steering constraints.
- J. Kong, M. Pfeiffer, G. Schildbach, F. Borrelli, "Kinematic and dynamic vehicle models for autonomous driving control design", IEEE IV, 2015. Validity range of the kinematic vs. dynamic model.
- A. Liniger, A. Domahidi, M. Morari, "Optimization-based autonomous racing of 1:43 scale RC cars", *Optimal Control Applications and Methods*, 2015. Pacejka single-track model with friction ellipse for small RC cars.
- M. O'Kelly et al., "F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning", PMLR (NeurIPS 2019 Competition Track), 2020. Original F1TENTH gym single-track model.
