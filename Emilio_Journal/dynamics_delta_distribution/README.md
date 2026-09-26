# Dynamics increment diagnostics

What the conditional \(p(s_{t+1}-s_t \mid s_t, a_t)\) looks like under Pure Pursuit, before choosing a unimodal Gaussian, a multimodal density, or an uncorrelated factorization.

The model sees only the 10-d car state and the 2-d action `(angular_control, translational_control)`. Lidar, waypoints, and walls are not inputs.

## Run

From the gym root, with the `f1t` environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python Emilio_Journal/dynamics_delta_distribution/run_experiment.py
```

That command recollects the dataset, refits both networks, and rewrites `outputs/`. Seed is fixed in `run_experiment.py` (`SEED = 0`).

The script does not edit any file outside this folder. Controller, delay, and recording flags are changed only in this process, then put back. A write guard rejects recordings, crash logs, and any other path under the gym that is not `outputs/` here.

## What is held out

Each episode is cut in time: the first 70% of its transitions train the network, the last 30% are the histogram sample. The networks are never fit on the rows that appear in the histograms.

- **1-step** input is `(s_t, a_t)`, target is \(\Delta s\).
- **3-step** input is the last three `(s, a)` pairs ending at \(t\). A window is used only when all three steps sit on the same side of the cut, so a test target does not see a training state.

`CONTROL_DELAY` is 0 for this run. The stored action is the command integrated over the control step (`TIMESTEP_CONTROL = 0.04 s`). Yaw increment is wrapped to \([-\pi, \pi]\). The sin/cos components are raw differences.

## Outputs

| File | Contents |
| --- | --- |
| `outputs/dataset.npz` | `state (N,10)`, `action (N,2)`, `next_state`, `delta`, `episode_id`, `train_mask` |
| `outputs/delta_histograms.pdf` | Held-out \(\Delta s\) histograms, \(\Delta s\) correlation, 1-step residual histograms, 3-step residual histograms |
| `outputs/metrics.json` | Counts, per-dimension mean/std, test RMSE |
| `outputs/mlp_1step.pt`, `outputs/mlp_3step.pt` | Tiny deterministic MLPs (2×32 tanh) plus the train-set standardization |
