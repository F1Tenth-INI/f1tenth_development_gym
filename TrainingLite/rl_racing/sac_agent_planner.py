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
import time

# Configure unbuffered output for immediate print statements (important for ROS nodes)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    # Fallback for older Python versions
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

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
            print(f"\n[RLAgentPlanner] Mode: TRAINING (receiving weights from server)")
        else:
            print(f"\n[RLAgentPlanner] Mode: INFERENCE (using model: {self.inference_model_name})")

        self.pre_fill_with_pp = Settings.SAC_PREFILL_BUFFER_WITH_PP
        self.pre_fill_amount = Settings.SAC_PREFILL_BUFFER_WITH_PP_AMOUNT
        # server_training_ready: starts True if no prefill, False if prefill is active.
        # Flipped to True when learner broadcasts training_status(training_ready=True).
        self.server_training_ready = not self.pre_fill_with_pp

        if self.pre_fill_with_pp and self.training_mode:
            print(f"\n[RLAgentPlanner] Pre-filling replay buffer with {self.pre_fill_amount} PurePursuit transitions")

        self.clear_buffer_on_reset = True
        self.terminate_server_after_simulation = True

        # --- networking ---

        if self.training_mode or self.inference_model_name is None:
            learner_host = getattr(Settings, "SAC_LEARNER_HOST", "127.0.0.1")
            learner_port = int(getattr(Settings, "SAC_LEARNER_PORT", 5555))
            print(f"[RLAgentPlanner] Connecting to learner at {learner_host}:{learner_port}")
            self.client = _TCPActorClient(host=learner_host, port=learner_port, actor_id=1)
            self.client.start()
            
            # Send clear buffer message on initialization
            if self.clear_buffer_on_reset:
                self.client.send_clear_buffer()
                print("\n[RLAgentPlanner] Sent clear buffer message to server")

        # Initialization of SAC utilities
        dummy_env = SacUtilities.create_vec_env()

        if self.training_mode or self.inference_model_name is None:
            self.model = SacUtilities.create_model(dummy_env, device="cpu")
            self._received_weights = False
            self._warned_no_weights = False
            self._warned_fallback_unavailable = False
        else:
            model_path, model_dir = SacUtilities.resolve_model_paths(self.inference_model_name)
            self.model = SAC.load(model_path, env=dummy_env, device="cpu")
            print(f"[Agent] Success: Loaded SAC model: {self.inference_model_name} from {model_path}")

            self._received_weights = True  # no waiting for weights in inference
            self._warned_no_weights = False
            self._warned_fallback_unavailable = False

        # constant warmup action in policy space [-1, 1]: [steer, accel]
        self.warmup_action = np.array([0.0, 0.30], dtype=np.float32)  # slight forward

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

        self.autonomous_driving = True
        self.control_index = 0
        self.total_sent = 0

        self.reset()


    def reset(self):
        self.action_history_queue.clear()
        self.action_history_queue.extend([np.zeros(2) for _ in range(10)])
        self.state_history.clear()
        self.state_history.extend([np.zeros(10) for _ in range(10)])
        
        self.transition_logger.clear()
        self.control_index = 0

        self.prev_obs_raw = None
        self.prev_action = None
        self.angular_control = 0.0
        self.translational_control = 0.0
        self.prev_angular_control = 0.0
        self.prev_translational_control = 0.0
        # do not clear self._episode here; do it on episode end

    def process_observation(self, ranges=None, ego_odom=None, observation=None):
        nikita_t_start = time.time()
        
        if not self.autonomous_driving:
            return self._fallback_action()
        
        # pull latest weights if any
        nikita_t1 = time.time()
        if self.training_mode or self.inference_model_name is None:
            training_ready = self.client.get_training_ready()
            if training_ready is not None and training_ready != self.server_training_ready:
                self.server_training_ready = training_ready
                if self.server_training_ready:
                    print("\n[RLAgentPlanner] Learner reports training-ready; enabling policy actions")

            sd = self.client.pop_latest_state_dict()
            if sd is not None:
                try:
                    # with torch.no_grad():
                    self.model.policy.actor.load_state_dict(sd, strict=True)
                    self.model.policy.actor.eval()
                        
                        # # NIKITA: Warmup forward pass to rebuild PyTorch optimizations
                        # dummy_obs = torch.zeros((1, self.model.observation_space.shape[0]), dtype=torch.float32)
                        # _ = self.model.policy.actor(dummy_obs)
                        
                    self._received_weights = True
                    nikita_t2 = time.time()
                    print(f"\n[RLAgentPlanner] ✅ Actor weights updated. (took {(nikita_t2-nikita_t1)*1000:.2f}ms)")
                except Exception as e:
                    print(f"\n[RLAgentPlanner] ❌ Failed to load actor weights: {repr(e)}")
        nikita_t_weights = time.time()

        # --- build raw obs (manual normalization happens inside) ---
        nikita_t3 = time.time()
        raw_obs = self._build_observation()
        nikita_t4 = time.time()
        obs_for_policy = raw_obs
        if self.vec_normalize is not None:
            obs_for_policy = self.vec_normalize.normalize_obs(raw_obs[None, :])[0]

        # choose action
        nikita_t5 = time.time()
        if self._received_weights and self.server_training_ready:
            with torch.no_grad():
                # NIKITA: for slowdown testing
                action, _ = self.model.predict(obs_for_policy, deterministic=(not self.training_mode))
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        else:
            if not self._warned_no_weights:
                print("\n[RLAgentPlanner] ⚠️ No weights yet; using warmup strategy ")
                self._warned_no_weights = True
            # if self.pre_fill_with_pp and not self.server_training_ready:
                # print(f"\n[RLAgentPlanner] ⚠️ Pre-filling replay buffer with PurePursuit transitions; using fallback action")
            action = self._fallback_action()
        nikita_t6 = time.time()

        # action = self._fallback_action()
        # scale to simulator units
        action = np.clip(action, -1, 1)
        steering, accel = action * self.action_denormalization_array

        #TODO: nikita: add clipping here
        if Settings.SAC_CURRICULUM_SPEED:
            # accel = np.clip(accel, -Settings.SAC_ACCEL_CAP, Settings.SAC_ACCEL_CAP)
            # speed = np.sqrt(np.power(raw_obs[LINEAR_VEL_X_IDX], 2) + np.power(raw_obs[LINEAR_VEL_Y_IDX], 2))
            vx = self.car_state[LINEAR_VEL_X_IDX]
            vy = self.car_state[LINEAR_VEL_Y_IDX]
            speed = np.sqrt(vx**2 + vy**2)
            # print(speed)

            ### Speed cap, with a soft gradual cap
            slowdown_margin = 0.5
            diff = speed - (Settings.SAC_CURRICULUM_SPEED_LIMIT - slowdown_margin)
            if diff > 0 and accel > 0:
                throttle_factor = max(0.0, 1.0 - diff / slowdown_margin)
                accel = accel * throttle_factor

                # print("speed limit reached, setting accel to 0")
                # accel = 0.0

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
        
        self.control_index += 1
        
        # NIKITA TIMING
        nikita_t_end = time.time()
        nikita_t_total = (nikita_t_end - nikita_t_start) * 1000
        if nikita_t_total > 10.0:  # Only print if > 10ms
            print(f"[TIMING] process_observation: total={nikita_t_total:.2f}ms | "
                  f"weights={((nikita_t_weights-nikita_t1)*1000):.2f}ms | "
                  f"obs_build={((nikita_t4-nikita_t3)*1000):.2f}ms | "
                  f"predict={((nikita_t6-nikita_t5)*1000):.2f}ms")
        
        return self.angular_control, self.translational_control


    def on_step_end(self, driver_obs:Dict[str, Any]) -> None:
        nikita_t_step_start = time.time()
        
        reward = driver_obs['reward']
        done = driver_obs['done']
        info = driver_obs['info']

        # if(done and self.autonomous_driving):
        #     print(f"DONE CALLED.")

        """Called by env AFTER stepping. Pass the obs returned by env.step"""
        if self.prev_obs_raw is None or self.prev_action is None:
            return  # first step guard

        nikita_t7 = time.time()
        next_obs = self._build_observation()
        nikita_t8 = time.time()

        transition = {
            "obs":      self.prev_obs_raw.astype(np.float32),
            "action":   self.prev_action.astype(np.float32),
            "next_obs": next_obs.astype(np.float32),
            "reward":   float(reward),
            "done":     bool(done),
            "info":     info or {},
        }
        nikita_t9 = time.time()
        self._episode.append(transition)
        if self.save_transitions:
            self.transition_logger.log(transition)
        nikita_t10 = time.time()

        # print(self.control_index)
        if done or self.control_index >= Settings.MAX_EPISODE_LENGTH:
            if self.save_transitions:
                self.transition_logger.save_csv()
            if self.training_mode and self.autonomous_driving:
                try:
                    if len(self._episode) > 1:
                        nikita_t11 = time.time()
                        self.client.send_transition_batch(self._episode)
                        nikita_t12 = time.time()
                        self.total_sent += len(self._episode)
                        total_reward = sum(t["reward"] for t in self._episode)
                        print(f"\n[RLAgentPlanner] Sending episode with {len(self._episode)} transitions with total reward {total_reward}. (send took {(nikita_t12-nikita_t11)*1000:.2f}ms)")
                    
                except Exception as e:
                    print(f"\n[RLAgentPlanner] Failed to send episode: {e}")
                finally:
                    self._episode = []
                    self.control_index = 0
                    self.prev_obs_raw = None
                    self.prev_action = None
            else:
                print(f"\n[RLAgentPlanner] Not sending episode because autonomous_driving is False.")
        
        # NIKITA TIMING
        nikita_t_step_end = time.time()
        nikita_t_step_total = (nikita_t_step_end - nikita_t_step_start) * 1000
        if nikita_t_step_total > 5.0:  # Only print if > 5ms
            print(f"[TIMING] on_step_end: total={nikita_t_step_total:.2f}ms | "
                  f"obs_build={((nikita_t8-nikita_t7)*1000):.2f}ms | "
                  f"transition_create={((nikita_t9-nikita_t8)*1000):.2f}ms | "
                  f"append={((nikita_t10-nikita_t9)*1000):.2f}ms")

    def on_simulation_end(self, collision=False):
        """Called when the simulation ends. Sends a terminate message to the server."""
        if self.terminate_server_after_simulation:
            if self.training_mode and hasattr(self, 'client') and self.client is not None:
                try:
                    self.client.send_terminate()
                    print("\n[RLAgentPlanner] Sent terminate message to server")
                except Exception as e:
                    print(f"\n[RLAgentPlanner] Failed to send terminate message: {e}")
        
        
    def get_total_progress(self):
        return min(1.0, self.total_sent / Settings.SIMULATION_LENGTH)
    
    def close(self):
        pass
    
    def _fallback_action(self) -> np.ndarray:
        # Keep the PP fallback planner synchronized with the current runtime context.
        self.fallback_planner.waypoint_utils = self.waypoint_utils
        self.fallback_planner.lidar_utils = self.lidar_utils
        self.fallback_planner.render_utils = self.render_utils

        if self.car_state is None:
            if not self._warned_fallback_unavailable:
                print("\n[RLAgentPlanner] Fallback PP unavailable (car_state not initialized); using warmup action")
                self._warned_fallback_unavailable = True
            return self.warmup_action.copy()

        self.fallback_planner.set_car_state(self.car_state)

        try:
            fallback_control = self.fallback_planner.process_observation()
            self._warned_fallback_unavailable = False
        except Exception as e:
            if not self._warned_fallback_unavailable:
                print(f"\n[RLAgentPlanner] Fallback PP failed ({e}); using warmup action")
                self._warned_fallback_unavailable = True
            return self.warmup_action.copy()

        fallback_action = fallback_control / self.action_denormalization_array
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
        border_points = np.concatenate([border_points_right[::3].flatten(), border_points_left[::3].flatten()])


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
            [0.5, 0.5]
            , [1]
            )) # Adjust normalization factors for each feature
        
        # SAC Training loop

        observation_array *= np.array(normalization_array, dtype=np.float32)
        return observation_array