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
import threading
import asyncio
import queue
from typing import Optional, Dict, Any, Deque
from collections import deque
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from Control_Toolkit_ASF.Controllers import template_planner

# from .tcp_utilities import (
#     pack_frame,
#     read_frame,
#     np_to_blob,
#     bytes_to_state_dict,
# )

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
        
# Your project imports
from utilities.state_utilities import *  # indices like LINEAR_VEL_X_IDX, etc.
from utilities.waypoint_utils import WaypointUtils
from utilities.lidar_utils import LidarHelper


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

        model_name = "SAC_RCA1_wpts_lidar_46"
        model_dir = os.path.join(root_dir, "TrainingLite","rl_racing","models", model_name)
        model_path = os.path.join(model_dir, model_name)

        # --- local SAC policy (inference only) ---
        # Observation: 4 (state) + 30 (wpts) + 40 (lidar) + 6 (last actions) = 80
        obs_low = np.array([-1, -1, -1, -1] + [-1]*30 + [0]*40 + [-1]*6, dtype=np.float32)
        obs_high = np.array([ 1,  1,  1,  1] + [ 1]*30 + [1]*40 + [ 1]*6, dtype=np.float32)
        obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        act_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                               high=np.array([ 1,  1], dtype=np.float32), dtype=np.float32)
        dummy_env = _SpacesOnlyEnv(obs_space, act_space)
        vec_dummy_env = _DummyVecEnv(dummy_env)  # Wrap for VecNormalize compatibility

        # Load model with proper device mapping (CPU for deployment)
         # Dynamically set device and map model to CPU if CUDA is unavailable
        device = "cpu"
        self.model = SAC.load(model_path, map_location=device)  # Use map_location to handle CPU fallback

        # --- Load VecNormalize for consistent normalization ---
        try:
            norm_path = os.path.join(model_dir, "vecnormalize.pkl")
            self.vec_normalize = VecNormalize.load(norm_path, vec_dummy_env)
            self.vec_normalize.training = False  # Freeze running stats
            self.vec_normalize.norm_reward = False  # Don't normalize rewards
            print("Loaded VecNormalize stats successfully for consistent normalization.")
        except FileNotFoundError:
            print(f"WARNING: VecNormalize file {norm_path} not found. Using manual normalization only.")
            self.vec_normalize = None

        # --- planner/driver-facing state ---
        self.angular_control: float = 0.0
        self.translational_control: float = 0.0

        # Initialize action history queue (same as training environment)
        self.action_history_queue = deque([np.zeros(2) for _ in range(10)], maxlen=10)

        self.prev_obs: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None

        self.waypoint_utils: WaypointUtils = WaypointUtils()
        self.lidar_utilities: LidarHelper = LidarHelper()
        self.reset()

    def reset(self):
        """Reset the planner state to match training environment reset behavior"""
        # Reset action history to zeros (same as training environment initialization)
        self.action_history_queue.clear()
        self.action_history_queue.extend([np.zeros(2) for _ in range(10)])
        
        # Reset other state variables
        self.prev_obs = None
        self.prev_action = None
        self.angular_control = 0.0
        self.translational_control = 0.0
        
        print("RLAgentPlanner reset - action history initialized to zeros")

    # ---- public API expected by your driver ----
    def process_observation(self, ranges=None, ego_odom=None, observation = None):

        self.lidar_utilities.update_ranges(ranges)
        # Build current observation from driver state
        obs = self._build_observation()
        # Apply VecNormalize normalization if available
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)
        # 4) Query policy for action (stochastic)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # scale to simulator control units (match your env scaling)
        steering = float(np.clip(action[0], -1, 1) * 0.4)
        accel = float(np.clip(action[1], -1, 1) * 10.0)
        # 5) remember for next transition
        self.prev_obs = obs
        self.prev_action = action
        # 6) export to driver
        self.angular_control = steering
        self.translational_control = accel
        # 7) update action history (for observation building of next step)
        self.action_history_queue.append(action)

        return self.angular_control, self.translational_control

    def on_step_end(self, reward: float, done: bool, info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def close(self):
        pass

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


