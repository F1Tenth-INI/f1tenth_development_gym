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
import csv
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
from Control_Toolkit_ASF.Controllers.PurePursuit.pp_planner import PurePursuitPlanner
from stable_baselines3.common.vec_env import DummyVecEnv

from utilities.waypoint_utils import *

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
        
# Your project imports
from utilities.state_utilities import *  # indices like LINEAR_VEL_X_IDX, etc.


from TrainingLite.rl_racing.tcp_client import _TCPActorClient
from TrainingLite.rl_racing.sac_utilities import SacUtilities, TransitionLogger

import torch
torch.set_num_threads(1)          # intra-op parallelism
torch.set_num_interop_threads(1)  # inter-op parallelism

# ------------------------
# RL Agent Planner (drop-in replacement for PurePursuitPlanner)
# ------------------------
class RLAgentPlanner(template_planner):
    def __init__(self):
        super().__init__()
        print("Initializing RLAgentPlanner (actor client)")

        # Training vs Inference   
        self.inference_model_name = Settings.SAC_INFERENCE_MODEL_NAME  # Model name thats loaded: if none: use weights from server
        self.training_mode = (self.inference_model_name is None)  # Training mode if no inference model specified
        
        if self.training_mode:
            print(f"[RLAgentPlanner] Mode: TRAINING (receiving weights from server)")
        else:
            print(f"[RLAgentPlanner] Mode: INFERENCE (using model: {self.inference_model_name})")


        self.clear_buffer_on_reset = True
        self.terminate_server_after_simulation = True

        # --- networking ---

        if self.training_mode or self.inference_model_name is None:
            # self.client = _TCPActorClient(host="192.168.194.226", port=5555, actor_id=2)
            self.client = _TCPActorClient(host="127.0.0.1", port=5555, actor_id=1)
            self.client.start()
            
            # Send clear buffer message on initialization
            if self.clear_buffer_on_reset:
                self.client.send_clear_buffer()
                print("[RLAgentPlanner] Sent clear buffer message to server")

        # Initialization of SAC utilities
        dummy_env = SacUtilities.create_vec_env()

        if self.training_mode or self.inference_model_name is None:
            self.model = SacUtilities.create_model(dummy_env, device="cpu")
            self._received_weights = False
            self._warned_no_weights = False
        else:
            model_path, model_dir = SacUtilities.resolve_model_paths(self.inference_model_name)
            self.model = SAC.load(model_path, env=dummy_env, device="cpu")
            print(f"[Agent] Success: Loaded SAC model: {self.inference_model_name} from {model_path}")

            self._received_weights = True  # no waiting for weights in inference
            self._warned_no_weights = False

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
        self.state_history = deque([np.zeros(10) for _ in range(10)], maxlen=10)

        # Lowpass filter state for control outputs
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        self.lowpass_alpha = 1.0  # Filter coefficient: 0 = no passing, 1 = no smoothing

        self.prev_obs_raw: Optional[np.ndarray] = None
        self.prev_action: Optional[np.ndarray] = None

        self.action_denormalization_array = np.array([0.4, 3.0])

        # episode accumulation
        self._episode: list[dict] = []

        self.fallback_planner: PurePursuitPlanner =  PurePursuitPlanner()
        self.fallback_planner.waypoint_utils = self.waypoint_utils
        self.fallback_planner.lidar_utils = self.lidar_utils
        self.transition_logger = TransitionLogger()
        self.save_transitions = False

        self.reset()


    def reset(self):
        self.action_history_queue.clear()
        self.action_history_queue.extend([np.zeros(2) for _ in range(10)])
        self.state_history.clear()
        self.state_history.extend([np.zeros(10) for _ in range(10)])
        
        self.transition_logger.clear()

        self.prev_obs_raw = None
        self.prev_action = None
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        # do not clear self._episode here; do it on episode end

    def process_observation(self, ranges=None, ego_odom=None, observation=None):

        # pull latest weights if any
        if self.training_mode or self.inference_model_name is None:
            sd = self.client.pop_latest_state_dict()
            if sd is not None:
                try:
                    self.model.policy.actor.load_state_dict(sd, strict=True)
                    self.model.policy.actor.eval()
                    self._received_weights = True
                    print("[RLAgentPlanner] ✅ Actor weights updated.")
                except Exception as e:
                    print(f"[RLAgentPlanner] ❌ Failed to load actor weights: {repr(e)}")


        # --- build raw obs (manual normalization happens inside) ---
        raw_obs = self._build_observation()
        obs_for_policy = raw_obs
        if self.vec_normalize is not None:
            obs_for_policy = self.vec_normalize.normalize_obs(raw_obs[None, :])[0]

        # choose action
        if self._received_weights:
            action, _ = self.model.predict(obs_for_policy, deterministic=(not self.training_mode))
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            if not self._warned_no_weights:
                print("[RLAgentPlanner] ⚠️ No weights yet; using warmup strategy ")
                self._warned_no_weights = True
            action = self._fallback_action()

        # action = self._fallback_action()
        # scale to simulator units
        action = np.clip(action, -1, 1)
        steering, accel = action * self.action_denormalization_array

        # remember pre-normalized obs & raw action for transition building
        self.prev_obs_raw = raw_obs
        self.prev_action  = action

        # Apply lowpass filter to control outputs
        # filtered = alpha * new_value + (1 - alpha) * previous_value
        self.angular_control = self.lowpass_alpha * steering + (1 - self.lowpass_alpha) * self.prev_angular_control
        self.translational_control = self.lowpass_alpha * accel + (1 - self.lowpass_alpha) * self.prev_translational_control
        
        # Update previous values for next iteration
        self.prev_angular_control = self.angular_control
        self.prev_translational_control = self.translational_control
        
        self.action_history_queue.append(action)
        self.state_history.append(self.car_state)
        

        return self.angular_control, self.translational_control


    def on_step_end(self, driver_obs:Dict[str, Any]) -> None:

        reward = driver_obs['reward']
        done = driver_obs['done']
        info = driver_obs['info']

        """Called by env AFTER stepping. Pass the obs returned by env.step"""
        if self.prev_obs_raw is None or self.prev_action is None:
            return  # first step guard


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
        if self.save_transitions:
            self.transition_logger.log(transition)

        if done:
            if self.save_transitions:
                self.transition_logger.save_csv()
            if self.training_mode:
                try:
                    self.client.send_transition_batch(self._episode)
                    total_reward = sum(t["reward"] for t in self._episode)
                    # print(f"[RLAgentPlanner] Sent episode with {len(self._episode)} transitions. Total reward: {total_reward}")
                except Exception as e:
                    print(f"[RLAgentPlanner] Failed to send episode: {e}")
                finally:
                    self._episode = []
                    self.prev_obs_raw = None
                    self.prev_action = None

    def on_simulation_end(self, collision=False):
        """Called when the simulation ends. Sends a terminate message to the server."""
        if self.terminate_server_after_simulation:
            if self.training_mode and hasattr(self, 'client') and self.client is not None:
                try:
                    self.client.send_terminate()
                    print("[RLAgentPlanner] Sent terminate message to server")
                except Exception as e:
                    print(f"[RLAgentPlanner] Failed to send terminate message: {e}")
       
    def close(self):
        pass
    
    def _fallback_action(self) -> np.ndarray:

        self.fallback_planner.lidar_utils = self.lidar_utils
        self.fallback_planner.car_state = self.car_state
        self.fallback_planner.waypoint_utils = self.waypoint_utils
        self.fallback_planner.car_state=(self.car_state)

        fallback_control = self.fallback_planner.process_observation()
        fallback_action = fallback_control / self.action_denormalization_array
        # return [0., 0.]
        return fallback_action

    # ---- helpers ----
    def _build_observation(self) -> np.ndarray:
        # car state
        car_state = self.car_state

        # border_points_left, border_points_right = self.waypoint_utils.get_track_border_positions_relative(self.waypoint_utils.next_waypoints, car_state)
        curvatures = self.waypoint_utils.next_waypoints[:, WP_KAPPA_IDX]

        border_distances_right = self.waypoint_utils.next_waypoints[:, WP_D_RIGHT_IDX]
        border_distances_left = self.waypoint_utils.next_waypoints[:, WP_D_LEFT_IDX]
        border_distances = np.concatenate([border_distances_right, border_distances_left])
        
        [border_points_left, border_points_right] = self.waypoint_utils.get_track_border_positions_relative(self.waypoint_utils.next_waypoints, car_state)
        border_points = np.concatenate([border_points_right.flatten(), border_points_left.flatten()])

        # Get frenet coordinates
        s, d, e, k = self.waypoint_utils.frenet_coordinates
        # print(f"distance: {s}, lateral offset: {d}, heading error: {e}, curvature: {k}")
        # lidar
        lidar = self.lidar_utils.processed_ranges

        # last 3 actions (6 dims)
        last_actions = list(self.action_history_queue)[-3:]
        last_actions = np.array(last_actions).reshape(-1)


        state_features = np.array([
            car_state[LINEAR_VEL_X_IDX],
            car_state[LINEAR_VEL_Y_IDX],
            car_state[ANGULAR_VEL_Z_IDX],
            car_state[STEERING_ANGLE_IDX],
        ], dtype=np.float32)


        observation_array = np.concatenate([
            state_features, 
            curvatures, 
            # lidar, 
            # border_distances,
            border_points,
            last_actions, 
            [d, e], 
            [Settings.GLOBAL_WAYPOINT_VEL_FACTOR]
        ]).astype(np.float32)

        # match env normalization
        normalization_array =  np.concatenate((
            [0.1, 1.0, 0.5, 1 / 0.4], 
            [1.0] * len(curvatures), 
            # [0.1] * len(lidar), 
            # [1.0] * len(border_distances),
            [0.2] * len(border_points),
            [1.0] * len(last_actions), 
            [0.5, 0.5],
            [Settings.GLOBAL_WAYPOINT_VEL_FACTOR]
            )) # Adjust normalization factors for each feature
        
        # SAC Training loop

        observation_array *= np.array(normalization_array, dtype=np.float32)
        return observation_array