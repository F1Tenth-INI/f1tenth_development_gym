# Async SAC Learner–Actor

This repo contains a **learner server** (trainer) and a **lightweight actor** (policy client) for an asynchronous SAC pipeline.  
Actors run inside the simulator, compute actions locally, and stream full-episode trajectories to the learner over TCP.  
The learner periodically trains on the replay buffer and **broadcasts updated actor weights** back to all connected actors.

---

## Features

- **SB3 SAC** off-policy training with periodic updates
- **TCP** transport (JSON framing + base64 blobs)
- **Manual observation normalization** (no VecNormalize)
- **Instant weight broadcast** on client connect
- **Warmup fallback** on actor before first weights arrive
- Multi-actor ready (`actor_id` included in each transition)

---

## Requirements

- Python ≥ 3.10
- `stable-baselines3`, `torch`
- `gymnasium`, `numpy`
- Your sim + utilities (e.g., `WaypointUtils`, `LidarHelper`, `state_utilities`)

> This setup intentionally **does not** use VecNormalize; observations are pre-scaled manually in the actor.

---

## Running the RL pipeline

### From an existing model zip

Loads `{model_dir}/{model_name}.zip`, infers spaces, and starts training/broadcast.

```bash
python TrainingLite/rl_racing/server.py   --model-name SAC_RCA1_wpts_lidar_50_async   --device cuda   --train-every-seconds 10   --replay-capacity 1000000
```

### From scratch

You must provide **obs**/**act** dimensions (e.g., obs=80, act=2).

```bash
python TrainingLite/rl_racing/server.py   --init-from-scratch   --obs-dim 80   --act-dim 2   --device cuda   --train-every-seconds 10   --replay-capacity 1000000
```

### Run the agent in simulation

Make sure in Settings.py that
CONTROLLER='sac_agent'
EXPERIMENT_LENGTH = 10000000 ( just very long).

Then just run the simulation.

```bash
python run.py   --init-from-scratch   --obs-dim 80   --act-dim 2   --device cuda   --train-every-seconds 10   --replay-capacity 1000000
```

## Architecture

```
[ Simulator ]
     │
     │  obs -> (actor builds feature vector)
     ▼
[ Actor ]
  - Local SB3 SAC policy for CPU inference
  - Receives actor weights from learner
  - Sends full episodes: (obs, action, reward, next_obs, done, info)
     │
     ▼ TCP
[ Learner Server ]
  - Ingests episodes into ReplayBuffer
  - Trains SAC periodically
  - Broadcasts updated actor weights to all clients
```

**Observation flow**

- Actor constructs `obs` via `_build_observation()` (car state + waypoints + lidar + last actions) and applies **manual scaling**.
- After `env.step()`, the env calls `driver.planner.compute_observation()` to build **next_obs** with the same translator.
- These pre-scaled `obs`/`next_obs` are stored and sent to the learner.

**Action flow**

- SAC outputs actions in `[-1, 1]`.
- Actor maps to sim units:
  - `steering = a0 * 0.4`
  - `accel = a1 * accel_scale` (tunable; default conservative to reduce early crashes)

---

**Defaults & knobs (server):**

- `learning_starts = 2000` (raise to 10000 later if desired)
- `utd_ratio ≈ 4.0` (gradient steps per newly added sample; capped by `min_grad_steps=16`, `max_grad_steps=4096`)
- Saves model to `models/{model_name}/{model_name}.zip` after each train round

**Actor behavior**

- On startup, builds a **fresh SAC** (shape & hyperparams only).
- **Waits for weights** from the learner; until they arrive, uses a **warmup fallback** action so the car moves.
- Once weights arrive: prints `✅ Actor weights updated.` and starts using the real policy.

---

## Observation schema (example)

```
[ state_features(4) ] +
[ waypoints(30) ] +           # 15 waypoints (x,y) downsampled
[ lidar(40) ] +               # processed/pooled lidar
[ last_actions(6) ]           # last 3 actions (2 each)
= 80 dims total
```

Manual scaling applied (example):

```python
normalization = [0.1, 1.0, 0.5, 1/0.4]               + [0.1]*30 + [0.1]*40 + [1.0]*6
obs *= normalization
```

> Keep the same scaling on both actor (collection) and learner (consumption). If you change the feature order or scaling, **bump the model name** to avoid mixing schemas.

---

## Troubleshooting

- **Actor doesn’t move**

  - Ensure the server is running and logs `Weights sent to ...`.
  - Actor should print `✅ Actor weights updated.` at least once.
  - Adjust `accel_scale` or warmup settings for initial motion.

- **Actor stuck in warmup**

  - Server started `--init-from-scratch` but never sent weights: confirm server logs show the initial broadcast on connect.
  - Ensure `handle_client` always sends `_weights_blob` if present.

- **SB3 environment error (DummyVecEnv)**

  - Make sure actor uses **SB3’s** `DummyVecEnv`:
    ```python
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: _SpacesOnlyEnv(obs_space, act_space)])
    ```

- **Crashes dominate early data**

  - Start with smaller `accel_scale` and/or “constant” warmup with modest forward.
  - Lower `learning_starts` temporarily (e.g., 2000).

- **Weights don’t load**
  - Actor logs will show key mismatches; ensure learner and actor use the **same policy architecture** and action/obs dims.

---

## Multi-actor

Run multiple sims/actors (different `actor_id`s) pointing to the same learner to speed up data collection. Diverse seeds and slightly different `accel_scale` improve exploration.

---

## License

MIT (or your preference).
