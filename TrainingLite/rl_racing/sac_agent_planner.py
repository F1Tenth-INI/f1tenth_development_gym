#!/usr/bin/env python3
"""
Agent client that replaces the PurePursuit planner and communicates with the
learner_server over TCP to:
  - receive policy weights
  - compute actions locally (CPU) using SB3 SAC policy
  - stream experience tuples (s, a, r, s', done, info) back to the learner

Drop-in replacement for your controller class (planner): implements
`process_observation(...)` and exposes `on_step_end(reward, done, info)` which
you should call from your env step once reward/done are known.

Integration (minimal diff):
1) Construct the planner:
     driver.planner = RLAgentPlanner(host="127.0.0.1", port=5555, actor_id=0)
2) In `RacingEnv.step` *after* you compute `reward`, `terminated`, `truncated`,
   and `info`, add:
     if hasattr(driver, "planner") and hasattr(driver.planner, "on_step_end"):
         driver.planner.on_step_end(float(reward), bool(terminated or truncated), info)

This mirrors the actor→learner design we discussed.
"""
from __future__ import annotations
import os
import sys

from typing import Optional, Dict, Any, Deque
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from Control_Toolkit_ASF.Controllers import template_planner
from stable_baselines3.common.vec_env import DummyVecEnv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
        
# Your project imports
from utilities.state_utilities import *  # indices like LINEAR_VEL_X_IDX, etc.
from utilities.waypoint_utils import WaypointUtils
from utilities.lidar_utils import LidarHelper

from TrainingLite.rl_racing.tcp_client import _TCPActorClient


# ------------------------
# Tiny dummy env to instantiate an SB3 policy for inference
# ------------------------

class _SpacesOnlyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_space: spaces.Box, act_space: spaces.Box):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}

    def step(self, action):  # never used
        raise RuntimeError("_SpacesOnlyEnv is not meant to be stepped")


# ------------------------
# Vectorized environment wrapper for VecNormalize compatibility
# ------------------------

class _DummyVecEnv:
    """Minimal vectorized environment wrapper for VecNormalize compatibility"""
    
    def __init__(self, env):
        self.env = env
        self.num_envs = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.render_mode = getattr(env, 'render_mode', None)
        self.metadata = getattr(env, 'metadata', {})
    
    def reset(self):
        obs, info = self.env.reset()
        return np.array([obs]), [info]
    
    def step(self, actions):
        action = actions[0]  # Take first action since we only have 1 env
        obs, reward, done, truncated, info = self.env.step(action)
        return np.array([obs]), np.array([reward]), np.array([done]), np.array([truncated]), [info]
    
    def close(self):
        self.env.close()


# ------------------------
# RL Agent Planner (drop-in replacement for PurePursuitPlanner)
# ------------------------
class RLAgentPlanner(template_planner):
    def __init__(self):
        super().__init__()
        print("Initializing RLAgentPlanner (actor client)")

        # --- networking ---
        self.client = _TCPActorClient(host="127.0.0.1", port=5555, actor_id=2)
        self.client.start()

        # --- define spaces (must match server) ---
        obs_low  = np.array([-1, -1, -1, -1] + [-1]*30 + [0]*40 + [-1]*6, dtype=np.float32)
        obs_high = np.array([ 1,  1,  1,  1] + [ 1]*30 + [1]*40 + [ 1]*6, dtype=np.float32)
        obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        act_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                            high=np.array([ 1,  1], dtype=np.float32), dtype=np.float32)

        # SB3’s real VecEnv (NOT the hand-rolled one)
        from stable_baselines3.common.vec_env import DummyVecEnv
        def make_env():
            return _SpacesOnlyEnv(obs_space, act_space)
        vec_env = DummyVecEnv([make_env])

        device = "cpu"
        # Fresh SAC just for shapes/settings; weights will be pushed from server
        self.model = SAC(
            "MlpPolicy",
            vec_env,
            device=device,
            verbose=0,
            gamma=0.99,
            learning_rate=1e-3,
            policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
            buffer_size=1,   # irrelevant on the actor
            batch_size=256,
            train_freq=1,
        )

        # weight + warmup state
        self._received_weights = False
        self._warned_no_weights = False
        # choose your warmup: "constant", "random", or "policy"
        self.warmup_strategy = "constant"
        # constant warmup action in policy space [-1, 1]: [steer, accel]
        self.warmup_action = np.array([0.0, 0.30], dtype=np.float32)  # slight forward
        # scale from policy space to sim units
        self.accel_scale = 3.0   # bump to e.g. 5.0 or 10.0 if you need more oomph

        # no VecNormalize in this setup
        self.vec_normalize = None

        # planner state
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.action_history_queue = deque([np.zeros(2) for _ in range(10)], maxlen=10)

        self.prev_obs_raw: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None

        # episode accumulation
        self._episode: list[dict] = []

        self.waypoint_utils: WaypointUtils = WaypointUtils()
        self.lidar_utilities: LidarHelper = LidarHelper()
        self.reset()


    def reset(self):
        self.action_history_queue.clear()
        self.action_history_queue.extend([np.zeros(2) for _ in range(10)])
        self.prev_obs_raw = None
        self.prev_action = None
        self.angular_control = 0.0
        self.translational_control = 0.0
        # do not clear self._episode here; do it on episode end

    def process_observation(self, ranges=None, ego_odom=None, observation=None):
        # pull latest weights if any
        sd = self.client.pop_latest_state_dict()
        if sd is not None:
            try:
                self.model.policy.actor.load_state_dict(sd, strict=True)
                self.model.policy.actor.eval()
                self._received_weights = True
                print("[RLAgentPlanner] ✅ Actor weights updated.")
            except Exception as e:
                print(f"[RLAgentPlanner] ❌ Failed to load actor weights: {repr(e)}")

        self.lidar_utilities.update_ranges(ranges)

        # --- build raw obs (manual normalization happens inside) ---
        raw_obs = self._build_observation()
        obs_for_policy = raw_obs
        if self.vec_normalize is not None:
            obs_for_policy = self.vec_normalize.normalize_obs(raw_obs[None, :])[0]

        # choose action
        if self._received_weights:
            action, _ = self.model.predict(obs_for_policy, deterministic=False)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            if not self._warned_no_weights:
                print("[RLAgentPlanner] ⚠️ No weights yet; using warmup strategy "
                    f"'{self.warmup_strategy}'. You can adjust accel_scale={self.accel_scale}.")
                self._warned_no_weights = True
            action = self._fallback_action(obs_for_policy)

        # scale to simulator units
        steering = float(np.clip(action[0], -1, 1) * 0.4)
        accel    = float(np.clip(action[1], -1, 1) * self.accel_scale)

        # remember pre-normalized obs & raw action for transition building
        self.prev_obs_raw = raw_obs
        self.prev_action  = action

        self.angular_control = steering
        self.translational_control = accel
        self.action_history_queue.append(action)

        return self.angular_control, self.translational_control


    def on_step_end(self, reward: float, done: bool, info: Optional[Dict[str, Any]] = None, next_obs=None) -> None:
        """Called by env AFTER stepping. Pass the obs returned by env.step after translating with compute_observation()."""
        if self.prev_obs_raw is None or self.prev_action is None:
            return  # first step guard

        if next_obs is None:
            # Prefer passing next_obs from env using the same builder; fallback remains
            next_obs = self._build_observation()

        transition = {
            "obs":      self.prev_obs_raw.astype(np.float32),
            "action":   self.prev_action.astype(np.float32),
            "next_obs": next_obs.astype(np.float32),
            "reward":   float(reward),
            "done":     bool(done),
            "info":     info or {},
        }
        self._episode.append(transition)

        if done:
            try:
                self.client.send_transition_batch(self._episode)
                total_reward = sum(t["reward"] for t in self._episode)
                print(f"[RLAgentPlanner] Sent episode with {len(self._episode)} transitions. Total reward: {total_reward}")
            except Exception as e:
                print(f"[RLAgentPlanner] Failed to send episode: {e}")
            finally:
                self._episode = []
                self.prev_obs_raw = None
                self.prev_action = None

    def close(self):
        pass
    
    def _fallback_action(self, obs_for_policy: np.ndarray) -> np.ndarray:
        """Action to use before first weights arrive."""
        if self.warmup_strategy == "constant":
            return self.warmup_action.copy()
        elif self.warmup_strategy == "random":
            a = np.random.uniform(low=-0.1, high=0.1, size=(2,)).astype(np.float32)
            a[1] = abs(a[1]) + 0.2  # bias forward
            return np.clip(a, -1.0, 1.0)
        elif self.warmup_strategy == "policy":
            # use the untrained policy (stochastic); mild but moving
            a, _ = self.model.predict(obs_for_policy, deterministic=False)
            return np.asarray(a, dtype=np.float32).reshape(-1)
        else:
            return np.array([0.0, 0.3], dtype=np.float32)
        
    def compute_observation(self) -> np.ndarray:
        """Public helper so the env can build the obs using the same translator."""
        return self._build_observation()

    # ---- helpers ----
    def _build_observation(self) -> np.ndarray:
        # car state
        car_state = self.car_state


        # waypoints: always 15 waypoints (30 values)
        xy = self.waypoint_utils.next_waypoint_positions_relative  # shape (K, 2)
        xy = xy[::2]
        wpts = xy.astype(np.float32).ravel()  # shape (30,)

        # lidar
        lidar = self.lidar_utilities.processed_ranges

        # last 3 actions (6 dims)
        last_actions = list(self.action_history_queue)[-3:]
        last_actions = np.array(last_actions).reshape(-1)

        state_features = np.array([
            car_state[LINEAR_VEL_X_IDX],
            car_state[LINEAR_VEL_Y_IDX],
            car_state[ANGULAR_VEL_Z_IDX],
            car_state[STEERING_ANGLE_IDX],
        ], dtype=np.float32)
    
        observation_array = np.concatenate([state_features, wpts, lidar, last_actions]).astype(np.float32)

        # match env normalization
        normalization_array =  [0.1, 1.0, 0.5, 1 / 0.4] + [0.1] * len(wpts)+ [0.1] * len(lidar) + [1.0] * len(last_actions) # Adjust normalization factors for each feature

        observation_array *= np.array(normalization_array, dtype=np.float32)
        return observation_array


