# Fitting the sim car model to real-car recordings

These scripts identify the parameters in `utilities/car_files/<car>.yml` that the simulator's dynamics model (`sim/f110_sim/envs/dynamic_model_pacejka_jax.py`) uses. They fit the model to recordings from the physical car. The model itself, and why it is physically sound, is documented in [`sim/f110_sim/envs/README_dynamics_model.md`](../../sim/f110_sim/envs/README_dynamics_model.md). That README also lists the current fitted values and the validation results.

| script | what it does | runtime |
|---|---|---|
| `fit.py` | fits the parameters (global + local optimisation), prints errors and the YAML lines | ~10 min on GPU |
| `evaluate.py` | prediction error of one or more parameter files, optionally with overrides | ~30 s |
| `steady_state_analysis.py` | checks that don't use the model: steering gain/lock, understeer, grip, drivetrain, ESC gains | ~10 s |
| `stress_test.py` | physical sanity of the model: no energy gain when sliding, acceleration within μg | ~1 min |
| `fitlib.py` | shared code: loading recordings, windowing, rollout, loss | – |

Run everything from the repo root (the `f1t` conda environment):

```bash
python AnalyseData/DynamicsModelFit/steady_state_analysis.py
python AnalyseData/DynamicsModelFit/fit.py --car yokomo1_car_parameters.yml --delay 0.04 --out fit.json
python AnalyseData/DynamicsModelFit/evaluate.py yokomo1_car_parameters.yml --delay 0.08
python AnalyseData/DynamicsModelFit/stress_test.py yokomo1_car_parameters.yml
```

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if you get a CUDA out-of-memory error. Don't run several fits in parallel on one GPU.

`fit.py` **does not modify the YAML**. Copy the printed values into it after checking them.

---

## 1. Data

The recordings are the CSVs that the car stack logs (same format as the simulator's `ExperimentRecordings`). By default, every CSV below `AnalyseData/Rosbags/2026-10-26/` that has the needed columns is used. Use `--recordings DIR` for other data, and `--test-file NAME` for the recording that is kept out of the fit entirely.

Columns used:

| column | meaning | why trusted |
|---|---|---|
| `angular_vel_z` | gyro yaw rate | matches the yaw rate from the scan-matched pose (ratio 1.00), noise ≈ 0.06 rad/s |
| `linear_vel_x` | wheel speed × r | matches speed from the scan-matched pose (ratio 1.00) |
| `pose_x`, `pose_y`, `pose_theta` | scan matcher pose | reference |
| `angular_control_executed`, `translational_control_executed` | commands actually sent to servo / ESC | model inputs |
| `LIDAR_*` | scan | to detect wall contact |

Not used: `v_x_ekf` (about 20% low and lagging) and `linear_vel_y` (does not correlate with the pose-derived lateral velocity). `steady_state_analysis.py` prints these checks, so redo them for new data.

---

## 2. Method

### 2.1 Multi-step open-loop prediction error

One-step errors are dominated by noise and say little about how the model behaves when the planner or simulator rolls it forward. The fit therefore uses **1 s open-loop predictions**, which is about the planning horizon:

1. **Windows.** A window starts at every 2nd sample, i.e. every 0.08 s (`fitlib.make_windows`).
2. **Initial state** from the measurement: r, v_x, pose.
   - The wheel angle δ is not measured. It comes from running the servo model over the previous 0.4 s of steering commands.
   - The lateral velocity v_y is initialised kinematically, v_y = v_x tan(δ) l_r / L, because the logged v_y is unreliable.
3. **Simulation.** The window is simulated with the **same model function, time step (0.01 s) and control delay as the simulator**. The executed commands are held for 0.04 s and delayed by `--delay`, just like `Settings.CONTROL_DELAY` in the sim.
4. **Comparison.** The predicted r, v_x, heading change and position are compared with the measurement at every one of the 25 samples.
5. **Loss.** Huber loss of the normalised errors, averaged over all windows and samples:

   ```
   huber(e_r / 0.3 rad/s) + huber(e_vx / 0.2 m/s) + huber(e_ψ / 0.05 rad) + huber(e_pos / 0.1 m)
   ```

   The scales put the four error types on a similar footing. The Huber loss limits the influence of the remaining outliers (e.g. scan-matcher jumps).

### 2.2 Data cleaning

A window is dropped if it contains

- **wall contact**: the minimum LiDAR range is < 0.15 m (about 4% of the sweep data), or
- **recovery manoeuvres**: |steering command| > 0.45, which only occurs during manual or recovery reversing.

### 2.3 Train / validation / test split

The speed sweeps get faster over time: the last quarter averages 3.0 m/s, against 1.5 m/s in the first. A chronological split would therefore test extrapolation rather than accuracy. Instead:

- **Train:** blocks 0, 1, 2 of every group of four 10 s blocks of the fit recordings.
- **Validation:** block 3 of every group.
- **Test:** one recording that the fit never sees (`--test-file`, by default the slow run `2026-10-26_09-31-34_.csv`).

If the training error is much lower than the validation error, the fit is overfitting.

### 2.4 Fitted parameters and bounds

| parameter | bounds | physical meaning |
|---|---|---|
| `mu` | 0.5 – 1.2 | peak tyre friction |
| `I_z` | 0.02 – 0.08 kg m² | yaw inertia |
| `s_max` (= −`s_min`) | 0.25 – 0.6 rad | effective steering lock |
| `steering_multiplier` | 0.9 – 1.8 | command → bicycle-equivalent wheel angle |
| `servo_p` | 5 – 40 1/s | servo bandwidth (the rate limit `sv_max` stays fixed) |
| `C_Pf` B, C | 2 – 20, 1.0 – 1.8 | front Magic Formula stiffness / shape |
| `C_Pr` B, C | 2 – 25, 1.0 – 1.8 | rear Magic Formula stiffness / shape |
| `c_rr` | 0 – 0.3 | rolling resistance |
| `curve_resistance_factor` | 0 – 1 | extra empirical cornering drag |
| `accel_multiplier`, `brake_multiplier` | 0.5 – 2.5, 0.5 – 10 | ESC: commanded → actual acceleration |

Fixed: geometry and mass (measured), D = 1 and E = 0 (μ already scales the peak, and D·μ cannot be separated), the actuator limits, and v_dead.

The bounds keep the solution physical. Pacejka C ≥ 1 and C ≤ 1.8 keeps at least 31% of peak grip at large slip; C > 2 would even reverse the force. The I_z range covers a radius of gyration of 0.08–0.17 m for a 2.9 kg car. **Always look at the "parameters at a bound" line.** A parameter sitting at a bound usually means that the model is compensating for something it doesn't model, not that the bound is too tight.

### 2.5 Optimiser

- **Global:** differential evolution (`scipy.optimize.differential_evolution`, population 8 × 13, 60 generations). The loss is not smooth in the parameters: steering lock and actuator clipping make it piecewise. The `stop_gradient` in the resistance terms also blocks gradients. A gradient-free global method therefore avoids the local optima a Powell-only search ran into.
- **Vectorised:** the whole population is evaluated in one JAX call (`fitlib.batch_loss_fn`, `jax.vmap` over parameter vectors). One generation takes about 8 s on a GPU for about 3000 windows.
- **Local:** Powell refinement from the best DE point.

### 2.6 Choosing the control delay

The delay is not fitted continuously. Run `fit.py` with `--delay 0.0 / 0.04 / 0.08` and compare the validation errors. For the 2026-10-26 data, 0.04 s and 0.08 s were equally good. This means the simulator's default `CONTROL_DELAY = 0.08` can be kept with the fitted parameters.

---

## 3. Checks that don't use the model (`steady_state_analysis.py`)

These come straight from the data and support or question the fitted values independently:

- **Quasi-steady cornering.** Path curvature r/v binned by steering command and speed, compared with the kinematic tan(δ)/L. This shows the low-speed steering gain (→ `steering_multiplier`), where curvature stops increasing (→ `s_max`), and how curvature drops with speed (understeer; a fit κ = cδ / (L(1 + K v²)) gives the gain c and the gradient K).
- **Grip.** The 98th percentile of the steady lateral acceleration v·r at high speed is a lower bound for μg.
- **Drivetrain.** The rear/front wheel speed ratio versus acceleration. A constant ratio means a locked 4WD, which justifies the friction-ellipse form in the model.
- **Longitudinal.** A regression a = k⁺u⁺ + k⁻u⁻ + a₀ + k_v v. k⁺ should match `accel_multiplier`, k⁻ should match `brake_multiplier`, and a₀ should match −`c_rr`·g plus drag.

---

## 4. Decisions taken for the 2026-10-26 fit

- **Rear tyre shape.** The fit gave C_r ≈ 1.75. This was replaced by the textbook C_r = 1.3, with B_r chosen to keep the same cornering stiffness B·C. The data never reach the rear tyre's peak, so the force after the peak is not determined by the fit. C = 1.75 would leave only 38% of the grip at large slip, which makes the simulator spin too easily. Cost: about +0.4° heading error at 0.5 s.
- **I_z.** Only loosely determined (0.02–0.03 equally good), so the more plausible 0.03 was used.
- **Checking a change.** Use `evaluate.py <yml> --set name=value` to see what a manual change costs before committing it.

---

## 5. Re-fitting for a new car or surface

1. Record a few minutes of driving that covers the speed range and some hard cornering. The RPGD speed sweep works well. Include a separate slow run as the test set.
2. Run `steady_state_analysis.py --recordings <dir>` and check that the signals pass the consistency checks.
3. Copy the car YAML. Set geometry and mass to the measured values.
4. Run `fit.py --car <your.yml> --recordings <dir> --test-file <slow run> --out fit.json` for 2–3 delays.
5. Check parameters at bounds, train vs. validation error, and the model-free checks. Then write the values into the YAML.
6. Run `stress_test.py <your.yml>`. It must report 0 non-finite runs, coasting energy gain of only a few percent, and full-throttle acceleration ≤ μg.
7. Update the tables in `README_dynamics_model.md`.
