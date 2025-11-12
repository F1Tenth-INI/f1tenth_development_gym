# Async SAC Learner–Actor

This folder contains a **learner server** (SAC trainer) and a **lightweight actor** (policy client) for an asynchronous SAC pipeline.  
Actors run inside the simulator, compute actions locally, and stream full-episode trajectories to the learner over TCP.  
The learner periodically trains on the replay buffer and **broadcasts updated actor weights** back to all connected actors.

---

## How to run

### Option 1: Server Only (Manual Client Launch)

Run the learner server and let it create a new model by setting a model name that does not yet exist.
You can also continue training an existing model by providing a `--model-name` of a pretrained model.

```bash
python TrainingLite/rl_racing/run_training.py --SIMULATION_LENGTH 300000 --model-name OriginalReward1
```

The server will now wait for an agent to connect and provide observations.
In another terminal, run the simulation with the SAC agent:

```bash
python run.py
```

### Option 2: Server with Auto-Started Client (Recommended)

Run both server and client together with a single command:

```bash
python TrainingLite/rl_racing/run_training.py --auto-start-client --SIMULATION_LENGTH 300000 --model-name OriginalReward1
```

This will automatically:
- Start the learner server
- Launch the simulation client
- Begin training

As the simulation runs, the SAC agent controller will collect observations (states, waypoints, sensor data etc. and a reward for a state-action pair) and send them to the server.

The server saves the observations to the replay buffer and continuously trains on them. Check out the plots in the models folder (`TrainingLite/rl_racing/Models/YOUR_MODEL_NAME`) to track the progress.

**ATTENTION:** The server runs independently from the client. Depending on the hardware, the bottleneck can either be observation collection or training.
In the learner server, check UTD (Update-to-Data) in the prints. This number represents how many training steps are done per observation.

## Inference

After successful training, the model can be used in inference mode for evaluation or deployment.

### Evaluation with Custom Settings

Run the simulator with a trained model in inference mode:

```bash
python run.py --RENDER_MODE human_fast --SIMULATION_LENGTH 2000 --SAVE_RECORDINGS True --SAC_INFERENCE_MODEL_NAME OriginalReward1
```

**Available options:**

All settings from `Settings.py` are available as command-line arguments. Common examples include:
- `--RENDER_MODE`: Visualization mode (`None`, `human`, `human_fast`)
- `--SIMULATION_LENGTH`: Number of timesteps to run
- `--SAVE_RECORDINGS`: Save episode data to CSV (`True`/`False`)
- `--SAC_INFERENCE_MODEL_NAME`: Name of the trained model to load
- `--MAP_NAME`: Select map (e.g., `RCA1`, `RCA2`)
- `--CONTROLLER`: Controller type (e.g., `sac_agent`, `pure_pursuit`)
- And many more from `Settings.py`

When a model name is provided via `--SAC_INFERENCE_MODEL_NAME`, the SAC planner:
- Loads the model weights directly (no server needed)
- Runs in deterministic mode (no exploration)
- Does not send transitions to any server

**Note:** The model must be available on the computer where the planner is running.

## Features

- **SB3 SAC** off-policy training with periodic updates
- **TCP** transport (JSON framing + base64 blobs)
- **Manual observation normalization** (no VecNormalize)
- **Instant weight broadcast** on client connect
- **Warmup fallback** on actor before first weights arrive
- Multi-actor ready (`actor_id` included in each transition)

---

> This setup intentionally **does not** use VecNormalize; observations are pre-scaled manually in the actor.

---

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

**Command-line arguments for training:**

Server-specific arguments:
- `--model-name`: Name of the model (required)
- `--host`: Server host address (default: `0.0.0.0`)
- `--port`: Server port (default: `5555`)
- `--device`: Training device (`cpu` or `cuda`)
- `--train-every-seconds`: Training interval in seconds
- `--gradient-steps`: Number of gradient steps per training iteration
- `--replay-capacity`: Replay buffer capacity
- `--learning-starts`: Minimum samples before training starts
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate for SAC
- `--discount-factor`: Discount factor (gamma)
- `--train-frequency`: Training frequency
- `--auto-start-client`: Automatically start client simulation (flag)
- `--forward-client-output`: Forward client output to terminal (flag, enabled by default)

Simulation settings (all from `Settings.py`):
- `--SIMULATION_LENGTH`: Total simulation timesteps
- `--RENDER_MODE`: Visualization mode
- `--MAP_NAME`: Select map
- `--CONTROLLER`: Controller type
- And all other settings from `Settings.py`

**Command-line arguments for inference:**

All settings from `Settings.py` are available, including:
- `--SAC_INFERENCE_MODEL_NAME`: Name of trained model to load
- `--RENDER_MODE`: Visualization (`None`, `human`, `human_fast`)
- `--SIMULATION_LENGTH`: Number of timesteps to run
- `--SAVE_RECORDINGS`: Save episode recordings (`True`/`False`)
- `--MAP_NAME`: Select map for evaluation

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
