#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import os
import subprocess
from typing import List, Optional, NamedTuple
import threading
from concurrent.futures import ThreadPoolExecutor

import argparse

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium import spaces
from stable_baselines3.common.logger import configure
import time
import csv

from tcp_utilities import pack_frame, read_frame, blob_to_np  # shared utils (JSON + base64 framing)
from sac_utilities import _SpacesOnlyEnv, SacUtilities, EpisodeReplayBuffer, TrainingLogHelper

from utilities.Settings import Settings

from utilities.StatTracker import StatTracker

from stable_baselines3.common.type_aliases import ReplayBufferSamples

from dataclasses import dataclass

@dataclass
class PriorityWeights:
    w_d: float
    w_e: float
    reward_weight: float
    velocity_weight: float
    alpha: float
    beta: float


class CustomReplayBuffer(ReplayBuffer):
    """Class to extend SB3 replaybuffer, to enable custom weighted sampling, and other additional helping functions"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_d = Settings.SAC_WP_OFFSET_WEIGHT #should i have all my weights in a dict or is this bad?
        self.w_e = Settings.SAC_WP_HEADING_ERROR_WEIGHT
        self.reward_weight = Settings.SAC_REWARD_WEIGHT
        self.velocity_weight = Settings.SAC_VELOCITY_WEIGHT
        self.alpha = Settings.SAC_PRIORITY_FACTOR
        self.custom_sampling = Settings.USE_CUSTOM_SAC_SAMPLING
        self.beta = Settings.SAC_IMPORANCE_SAMPLING_CORRECTOR
        self.initial_beta = self.beta
        self.beta_end = 1.0
        self.beta_annealing_horizon = Settings.SAC_BETA_ANNEALING_RATIO * Settings.SIMULATION_LENGTH
        self.state_weights = np.ones(self.buffer_size, dtype=np.float64) * 1e-6 # Initialize with small values to avoid zero div
        self.importance_sampling_correctors = np.ones(self.buffer_size, dtype=np.float64)
        self.batch_is_correctors = None

        self.state_to_TD_ratio = Settings.SAC_STATE_TO_TD_RATIO #if 0, only TD error based priorities

        self.TD_weights = np.zeros(self.buffer_size, dtype=np.float64)
        self.new_weight_priority = 1.0
        self.dynamic_importance_sampling_corrector = Settings.SAC_DYNAMIC_IS_CORRECTOR

        self.current_sampled_inds = None #indexes of the data which was just sampled in last sample() call

        self.clip_weights = Settings.SAC_CLIP_WEIGHTS

        self.rank_based_sampling = Settings.SAC_RANK_BASED_SAMPLING

        self.steps_taken = np.ones(self.buffer_size, dtype=np.int32)

        # if 

        ###FOR DEBUGGING
        self.counter = 0     
        if self.custom_sampling:
            print("Using custom SAC replay buffer sampling with weights:")
            print(f"Offset weight: {self.w_d}, Heading error weight: {self.w_e}, Reward weight: {self.reward_weight}, Speed weight: {self.velocity_weight}, Priority factor: {self.alpha}")

        # if Settings.SAC_STAT_TRACKER:
        #     self.stat_tracker = StatTracker()
        # else:
        #     self.stat_tracker = None

        #TODO: Nikita: implement higher weighting on recently added experiences


    def add(self, obs, next_obs, action, reward, done, infos, steps_taken: int):
        """Override SB3 add so that the weight can be computed once per transition, and then stored in the buffer"""
        # --> actual weighting computation is only within this function
        super().add(obs, next_obs, action, reward, done, infos)

        #handle special pos index for SB3
        try:
            idx = (self.pos - 1) % self.buffer_size
        except Exception:
            return
        
        # Extract newest unnormalized observation added by super().add()
        obs = self.observations[idx, 0, :]

        if self.stat_tracker is not None:
            self.stat_tracker.register_transition(obs, idx, reward, done) #gets the unnormalized obs directly
            # self.stat_tracker.print_stats()

        if not self.custom_sampling:
            return

        #store steps taken for each transition over whole buffer
        self.steps_taken[idx] = steps_taken

        #set to max samplepriority for newest transitions
        #TODO: option to have this be able to be set in settings
        self.TD_weights[idx] = self.new_weight_priority

        #TODO: the observations are normalized, is that bad or not?
        #compute weight
        d = obs[-2]
        e = obs[-1]
        # rew = self.rewards[idx, 0] + 1
        rew = abs(self.rewards[idx, 0]) # both positive and negative rewards contain lots of information

        # Scale the reward contribution to the priority by the number of steps in the n-step transition.
        # Without this, longer n-step transitions are over-prioritised.
        # Use per-step average reward for priority computation.
        try:
            reward_per_step = float(rew) / max(1, int(steps_taken))
        except Exception:
            reward_per_step = float(rew)

        #velx = obs[0] and vely = obs[1]
        vel = np.sqrt(obs[0]**2 + obs[1]**2)
        # vel = (1/(1-e)) * vel #we only care about speed into the right direction
        vel = (1 - min(e, 0.9)) * vel #we only care about speed into the right direction
        # print(rew)

        w = self.w_d * abs(d) + self.w_e * abs(e) + self.reward_weight * abs(reward_per_step) + self.velocity_weight * abs(vel)
        w = np.clip(w, 1e-6, 1e3)

        if self.clip_weights:
            w = np.clip(w, 0, 5)

        self.state_weights[idx] = w
        
        return
    
    def update_TD_priorities(self, TD_update_inds: np.ndarray, TD_update_priorities: np.ndarray):
        self.TD_weights[TD_update_inds] = TD_update_priorities

        #get newest max prio values for new transitions
        self.new_weight_priority = max(self.new_weight_priority, TD_update_priorities.max())

        return

    def sample(self, safe_batch_size: int, env=None):
        """custom sample function, if none then use default SB3"""
        #weights already calculated in add(), here we just normalize and sample

        if not self.custom_sampling:
            return super().sample(batch_size=safe_batch_size, env=env)
        
        #IDEA -> prioritize bad and difficult states -> far from line, large curve, large heading error 
        #the index self.pos cannot be sampled, this is the index which will be overwritten next, and as such it contains invalid data
        #sac needs current obs and next obs, and next obs is not given yet at index self.pos
        if self.full:
            possible_inds = np.arange(self.buffer_size)
            mask = possible_inds != self.pos #mask which would set the self.pos index as false
            possible_inds = possible_inds[mask]
        else:
            possible_inds = np.arange(self.pos)

        length = len(possible_inds)

        # Use stored per-transition weights for sampling instead of recomputing from obs
        w_vec = self.state_weights[possible_inds].astype(np.float64)

        """TD error based priorities"""
        TD_vec = self.TD_weights[possible_inds].astype(np.float64)

        combined_weight = TD_vec * (1 - self.state_to_TD_ratio) + w_vec * self.state_to_TD_ratio

        # Squish priorities logarithmically to dampen outliers while preserving rank
        combined_weight = np.log1p(combined_weight)

        if self.rank_based_sampling:
            sorted_inds = np.argsort(-combined_weight) #highest to lowest
            ranked_buffer_inds = possible_inds[sorted_inds]
            ranks = np.arange(1, length + 1)
            p = ranks ** -self.alpha
        else:
            p = combined_weight ** self.alpha
            
        p_tot = p.sum()
        if p_tot <= 0 or not np.isfinite(p_tot):
            p = np.ones_like(p) / length
        else:
            p /= p_tot
        """"""""""""""""""""""""""""""""""""""""""""""""
        """DEBUG DEBUG DEBUG"""

        # if self.counter % 1000 == 0:
        #     # test_idx = np.random.choice(length, size=1)[0]
        #     print("--- Debug Info ---")
        #     # print("idx: " + str(test_idx))
        #     # print("w: " + str(w_vec[test_idx]))
        #     # uni_p = 1.0 / max(1, self.size())
        #     # print("TD_vec: " + str(TD_vec[test_idx]))
        #     # print("combined weight: " + str(combined_weight[test_idx]))
        #     # print("uniform sampling prob: " + str(uni_p))
        #     print("max weight: " + str(combined_weight.max()))
        #     print("min weight: " + str(combined_weight.min()))

        #     print("max prob: " + str(p.max()))
        #     print("min prob: " + str(p.min()))

        #     # # Find who is holding this static max value
        #     # max_weight_idx_in_possible = np.argmax(combined_weight)
        #     # max_weight_val = combined_weight[max_weight_idx_in_possible]
        #     # actual_buffer_idx = possible_inds[max_weight_idx_in_possible]

        #     # print(f"--- Max Weight Debug ---")
        #     # print(f"Max Weight: {max_weight_val}")
        #     # print(f"Held by Buffer Index: {actual_buffer_idx}")

        """"""""""""""""""""""""""""""""""""""""""""""""
        """DEBUG DEBUG DEBUG"""

        #TODO: do i want replace or not? :/
        sampled_p_index = np.random.choice(length, size=safe_batch_size, p=p)
        
        if self.rank_based_sampling:
            batch_inds = ranked_buffer_inds[sampled_p_index]
        else:
            #map the sampled indices back to the possible_inds
            batch_inds = possible_inds[sampled_p_index]


        """"""""""""""""""""""""""""""""""""""""""""""""
        """DEBUG DEBUG DEBUG"""

        # if self.counter % 1000 == 0:
        #     # Check if this index was just sampled
        #     if actual_buffer_idx in batch_inds:
        #         print(f"✅ ALERT: The Max Weight Index {actual_buffer_idx} WAS SAMPLED this batch!")
        #     else:
        #         print(f"❌ The Max Weight Index {actual_buffer_idx} was NOT sampled.")

            
        self.counter += 1

        """"""""""""""""""""""""""""""""""""""""""""""""
        """DEBUG DEBUG DEBUG"""

        #set all the is_weights
        sample_probs = p[sampled_p_index]
        is_weights = (1 / (length * sample_probs)) ** self.beta
        is_weights = is_weights / is_weights.max()

        #print("the current beta is:" + str(self.beta))


        self.batch_is_correctors = is_weights.reshape(-1, 1).astype(np.float32)

        self.current_sampled_inds = batch_inds

        # self.importance_sampling_correctors[batch_inds] = is_weights

        # self.current_sampled_inds = batch_inds
        
        # ###TODO: this is slow and inefficient
        # for idx in batch_inds:
        #     if idx > self.pos:
        #         self.importance_sampling_correctors[idx] = (1 / (self.buffer_size * p[idx-1])) ** self.beta
        #     else:
        #         self.importance_sampling_correctors[idx] = (1 / (self.buffer_size * p[idx])) ** self.beta
        
        # self.batch_is_correctors = self.importance_sampling_correctors[batch_inds]
        
        # self.batch_is_correctors = self.batch_is_correctors.reshape(-1, 1).astype(np.float32)
        # assert np.all(np.isin(batch_inds, possible_inds))

        return self._get_samples(batch_inds, env=env)
    

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> WeightedReplayBufferSamples:

        if self.stat_tracker is not None:
            self.stat_tracker.batch_update_sample_count(batch_inds)

        if not self.custom_sampling:
            samples = super()._get_samples(batch_inds, env=env)
            steps_taken = self.to_torch(self.steps_taken[batch_inds].reshape(-1, 1))
            is_weights = torch.ones((len(batch_inds), 1), device=samples.observations.device)   
            return WeightedReplayBufferSamples(
                observations=samples.observations,
                actions=samples.actions,
                next_observations=samples.next_observations,
                dones=samples.dones,
                rewards=samples.rewards,
                is_weights=is_weights,
                steps_taken = steps_taken)  
        

        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        steps_taken = self.steps_taken[batch_inds].reshape(-1, 1)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )

        
        return WeightedReplayBufferSamples(*tuple(map(self.to_torch, data)), 
                                           is_weights = self.to_torch(self.batch_is_correctors),
                                           steps_taken = self.to_torch(steps_taken),
                                            )

    
class WeightedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    is_weights: torch.Tensor
    # For n-step replay buffer
    steps_taken: torch.Tensor


# class WeightedReplayBufferSamples(ReplayBufferSamples):
#     def __init__(self, *args, is_weights=None):
#         super().__init__(*args)
#         self.is_weights = is_weights


class LearnerServer:
    def __init__(
        self,
        host: str,
        port: int,
        # load_model_name: if None -> start from scratch; otherwise attempt to load this model
        load_model_name: Optional[str] = None,
        # save_model_name: where to save training progress (fallback if not provided)
        save_model_name: Optional[str] = None,
        device: str = "cpu",
        train_every_seconds: float = 10.0,
        grad_steps: int = 1024,
        replay_capacity: int = 100_000,
        learning_starts: int = 2000,
        batch_size: int = 1024,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        train_frequency: int = 1,
    ):
        self.host = host
        self.port = port
        self.load_model_name = load_model_name
        self.save_model_name = save_model_name
        self.device = device
        self.train_every_seconds = train_every_seconds
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.train_frequency = train_frequency
        self.max_utd = 4.0  # max updates per data point

        # Settings
        self.learning_starts = learning_starts
        self.grad_steps = grad_steps

        self.init_from_scratch = None
        # Store reference to event loop for worker threads
        self._loop_ref: Optional[asyncio.AbstractEventLoop] = None
        # Create dedicated executor with minimal workers to reduce CPU contention
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="learner_worker")
        # networking
        self._clients: set[asyncio.StreamWriter] = set()
        self._client_lock = asyncio.Lock()
        self._should_terminate = False
        self._terminate_lock = asyncio.Lock()
        self._training_in_progress = False  # Flag to prevent concurrent training
        self._training_lock = asyncio.Lock()
        self._last_weight_broadcast_time = 0.0  # Track last broadcast to rate-limit them
        self._min_broadcast_interval = 5.0  # Only broadcast at most every 5 seconds

        # data
        self.episode_buffer = EpisodeReplayBuffer(capacity_episodes=2000)
        self._weights_blob: Optional[bytes] = None

        # RL bits (lazy init fallback; but we’ll eager-init in run())
        self.model: Optional[SAC] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.vecnorm: Optional[VecNormalize] = None
        self.total_actor_timesteps = 0
        self.total_weight_updates = 0

        self.last_episode_time = None 
        self.episode_timeout = 100.0

        #used for n-step buffer, for standard buffer set = 1
        self.n_step = getattr(Settings, "SAC_N_STEP", 1)
        # self.n_step_discount_factor = self.discount_factor ** self.n_step
        self.custom_sampling = Settings.USE_CUSTOM_SAC_SAMPLING


        # file paths: default save name fallback
        if self.save_model_name is None:
            # If save name not provided, fall back to load name if present, else use a generic default
            if self.load_model_name is not None:
                self.save_model_name = self.load_model_name
            else:
                self.save_model_name = "SAC_RCA1_0"

        # resolve save paths (where we will write checkpoints/logs)
        self.save_model_path, self.model_dir = SacUtilities.resolve_model_paths(self.save_model_name)

        # resolve load path if different
        self.load_model_path = None
        if self.load_model_name is not None:
            self.load_model_path, _ = SacUtilities.resolve_model_paths(self.load_model_name)

        self.trainingLogHelper = TrainingLogHelper(self.save_model_name, self.model_dir)

        SacUtilities.zip_relevant_files(self.model_dir)

        self._initialize_model()

    # ---------- setup / init ----------
    def _load_base_model(self):
        """
        Initialize the model + replay buffer.

        - If `load_model_name` is set, attempt to load from that path.
        - Else: create a new model (train from scratch).
        """
        dummy_env = SacUtilities.create_vec_env()

        # If no load model specified -> start from scratch
        if self.load_model_name is None:
            print("[server] No load model specified: creating new model from scratch.")
            self.init_from_scratch = True
            self.model = SacUtilities.create_model(
                env=dummy_env,
                buffer_size=self.replay_capacity,
                device=self.device,
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor,
                train_freq=self.train_frequency,
                batch_size=self.batch_size,
            )
            print(f"[server] Success: Created new SAC model (scratch).")
        else:
            # try loading provided model, otherwise fall back to creating new
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                self.model = SAC.load(self.load_model_path, env=dummy_env, device=self.device)
                self.init_from_scratch = False
                print(f"[server] Success: Loaded SAC model: {self.load_model_name}")
            except Exception as e:
                print(f"[server] Failed to load SAC model at {self.load_model_path}: {e}")
                print("Creating new model from scratch.")
                self.init_from_scratch = True
                self.model = SacUtilities.create_model(
                    env=dummy_env,
                    buffer_size=self.replay_capacity,
                    device=self.device,
                    learning_rate=self.learning_rate,
                    discount_factor=self.discount_factor,
                    train_freq=self.train_frequency,
                    batch_size=self.batch_size,
                )
                print(f"[server] Success: Created new SAC model (fallback).")

        # Minimal logger so .train() works outside .learn()

        self.model.batch_size = self.batch_size
        self.model.gradient_steps = self.grad_steps
        self.model._logger = configure(folder=None, format_strings=[])

        # Fresh replay buffer
        # self.replay_buffer = RewardBiasedReplayBuffer(

        """Nikita: this is where i changed it to CustomReplayBuffer"""
        self.replay_buffer = CustomReplayBuffer(
            buffer_size=self.replay_capacity,
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            device=self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        if Settings.SAC_STAT_TRACKER:
            stat_save_dir = os.path.join(self.model_dir, "stat_logs")
            self.replay_buffer.stat_tracker = StatTracker(save_dir=stat_save_dir, save_name="stats_log.csv")
            self._last_stat_save_ts = 0.0
        else:
            self.replay_buffer.stat_tracker = None

        self.model.replay_buffer = self.replay_buffer

        # Device diagnostics: verify model and replay buffer are on the correct device
        print(f"[server] Device config: requested={self.device}")
        print(f"[server] Model policy device: {next(self.model.policy.parameters()).device}")
        if torch.cuda.is_available():
            print(f"[server] CUDA available: True, cuda_device_count={torch.cuda.device_count()}, current_device={torch.cuda.current_device()}, device_name={torch.cuda.get_device_name()}")
        else:
            print(f"[server] CUDA available: False")
        print(f"[server] Replay buffer device: {self.replay_buffer.device}")

        # Save model info
        info = {
            "grad_steps": self.grad_steps,
            "batch_size": self.batch_size,
        }
        self.trainingLogHelper.save_meta_info(self.model, info)
        # Cache weights for initial broadcast (random if scratch)
        self._weights_blob = SacUtilities.state_dict_to_bytes(self.model.policy.actor.state_dict())

        print(
            "[server] Base model + replay buffer initialized.",
            f"Mode={'scratch' if self.init_from_scratch else f'from-{self.load_model_name}'}",
        )


    def _initialize_model(self):
        if self.model is not None: return
        self._load_base_model()
        print(f"[server] Initialized model")

    def _save_model(self):
        """Save the current model to disk (uses save_model_name)."""
        if self.model is not None:
            try:
                target = os.path.join(self.model_dir, self.save_model_name)
                self.model.save(target)
                print(f"[server] Model saved to {target}")
            except Exception as e:
                print(f"[server] Error saving model: {e}")

    # ---------- ingestion + training ----------
    # No normalization for now: obs arrive normalized already
    def _normalize_obs(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)
      
    
    def _dummy_vec_from_spaces(self, obs_space: spaces.Box, act_space: spaces.Box):
        return DummyVecEnv([lambda: _SpacesOnlyEnv(obs_space, act_space)])
    
    def _ingest_episodes_into_replay(self, episodes: List[List[dict]]) -> int:
        if self.model is None or self.replay_buffer is None:
            return 0
        n_added = 0
        for ep in episodes:
            if self.n_step == 1:
                transitions = ep
            else:
                transitions = self._compute_n_step_transitions(ep)
                
            for t in transitions:
                obs = t["obs"].astype(np.float32)
                next_obs = t["next_obs"].astype(np.float32)
                action = t["action"].astype(np.float32)
                reward = float(t["reward"])
                done = bool(t["done"])
                steps_taken = t.get("steps_taken", 1)  # Default to 1 for non-n-step transitions
                # normalize obs like SB3 would during collection
                obs = self._normalize_obs(obs)
                next_obs = self._normalize_obs(next_obs)
                # sampling_weight = self.single_weight(obs, action, next_obs, reward)
                self.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos={}, # could pass TimeLimit info here if you track it
                    steps_taken=steps_taken  
                    # sampling_weight = sampling_weight
                )
                n_added += 1
        return n_added

    def _ingest_episodes_into_replay_old(self, episodes: List[List[dict]]) -> int:
        if self.model is None or self.replay_buffer is None:
            return 0
        n_added = 0
        for ep in episodes:
            for t in ep:
                obs = t["obs"].astype(np.float32)
                next_obs = t["next_obs"].astype(np.float32)
                action = t["action"].astype(np.float32)
                reward = float(t["reward"])
                done = bool(t["done"])
                # normalize obs like SB3 would during collection
                obs = self._normalize_obs(obs)
                next_obs = self._normalize_obs(next_obs)
                # sampling_weight = self.single_weight(obs, action, next_obs, reward)
                self.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos={},  # could pass TimeLimit info here if you track it
                    # sampling_weight = sampling_weight
                )
                n_added += 1
        return n_added
    
    def _compute_n_step_transitions(self, episode: List[dict]) -> List[dict]:
        """
        Takes list of episode transitions, converts into n-step reward structure.
        
        According to the n-step SAC formulation (eq. 8 from the paper):
        R_t^n = sum_{i=0}^{n-1} gamma^i * r_{t+i} 
              + alpha * sum_{i=1}^{n} gamma^i * H(pi(s_{t+i}))  [entropy bonus - handled in training]
              + gamma^n * Q(s_{t+n})                            [bootstrapping - handled in training]
        """
        n = self.n_step
        n_step_transitions = []
        episode_length = len(episode)
        for i in range(episode_length):
            n_step_reward = 0.0
            cur_discount = 1.0 #no discount on first step
            for j in range(n):
                if i + j < episode_length:
                    n_step_reward += cur_discount * episode[i + j]["reward"]
                    cur_discount *= self.discount_factor
                    steps_taken = j + 1
                    if episode[i + j].get("done", False): # stop if theres an episode end in the sequence -> not enough steps
                        break
                else:
                    break

            final_index = i + (steps_taken - 1)

            # assert steps_taken >= 1
            # assert steps_taken <= self.n_step


            #TODO: figure out wth is going on here


            # If we were able to go full N steps without ending the episode:
            # The next state is the observation N steps later.
            # But wait: if i+n is typically the start of the n-th step. 
            # We want the state AFTER n steps.
            
            # Case A: We ran out of episode (Terminal within window)
            if episode[final_index]["done"]:
                target_transition = episode[final_index]
                next_obs = target_transition["next_obs"] # Use the terminal state
                # CHECK FOR TIMEOUT HERE
                is_timeout = target_transition.get("info", {}).get("TimeLimit.truncated", False)
                
                if is_timeout:
                    done = False # <--- Force False so Bellman bootstraps
                else:
                    done = True # <--- Real crash
    
            
            # Case B: We successfully looked ahead N steps (Non-terminal)
            # The "next state" for the update is the 'obs' of the transition at i + n
            # OR the 'next_obs' of the transition at i + n - 1. They are the same.
            elif i + n < episode_length:
                target_transition = episode[i + n]
                next_obs = target_transition["obs"] # The state at start of step t+n
                done = False
                
            # Case C: Edge case where we hit end of list but 'done' wasn't True (e.g. timeout)
            else:
                target_transition = episode[-1]
                next_obs = target_transition["next_obs"]
                is_timeout = target_transition.get("info", {}).get("TimeLimit.truncated", False)
                if is_timeout:
                    done = False
                else:
                    done = bool(target_transition["done"])

            # --- 3. Build the Transition ---
            # We preserve the ORIGINAL observation and action from step 'i'
            current_transition = episode[i]
            
            new_transition = {
                "obs": current_transition["obs"],
                "action": current_transition["action"],
                "next_obs": next_obs,
                "reward": n_step_reward,
                "done": done,
                "steps_taken": steps_taken
            }

            n_step_transitions.append(new_transition)
            
        return n_step_transitions
    
    
    def _apply_crash_ramp(self, episode, ramp_steps: int = 50, max_ramp_value: float = 1000.0):
        """
        Modify rewards IN PLACE so that, if an episode ends in a crash,
        the last `ramp_steps` transitions before the crash get extra negative reward.

        extra_penalty: total additional penalty distributed over the ramp.
        """
        # find crash index (last transition where done and info["crash"] == True)
        crash_idx = None
        for i in reversed(range(len(episode))):
            t = episode[i]
            info = t.get("info", {})
            if t.get("done", False) and info.get("truncated", False):
                crash_idx = i
                break

        if crash_idx is None:
            # no crash in this episode -> leave rewards unchanged
            return episode

        # distribute extra_penalty backwards over ramp_steps
        # closer to crash -> larger share
        for k in range(ramp_steps):
            idx = crash_idx - k
            if idx < 0:
                break
            t = episode[idx]
            # weight from 0..1, increasing towards crash
            w = (ramp_steps - k) / ramp_steps
            shaped_penalty = w * (max_ramp_value )
            t["reward"] -= shaped_penalty
        return episode


    async def _train_loop(self):
        """Periodic training loop: drain new episodes -> add to replay -> train -> broadcast new weights."""
        loop_iteration_count = 0
        last_log_time = time.time()
        
        while True:
            loop_iteration_count += 1
            loop_start = time.time()
            
            # Log loop rate every 5 seconds
            now = time.time()
            if now - last_log_time >= 5.0:
                print(f"[server] Loop rate: {loop_iteration_count / (now - last_log_time):.1f} iterations/sec")
                loop_iteration_count = 0
                last_log_time = now
            
            # Custom stopping logic for bash, doesnt always stop normally...
            if Settings.EXTENDED_AUTO_STOP and self.total_actor_timesteps > Settings.SIMULATION_LENGTH and len(self._clients) == 0:
                print("[server] All clients disconnected. Assuming simulation finished. Shutting down.")
                async with self._terminate_lock:
                    self._should_terminate = True
                return
            
            # Check termination flag at the start of each iteration
            async with self._terminate_lock:
                if self._should_terminate:
                    print("[server] Training loop detected termination flag, exiting...")
                    return
            
            # Drain whatever the actor sent since last round
            episodes = self.episode_buffer.drain_all()

            n_added = self._ingest_episodes_into_replay(episodes)
            if n_added > 0:
                print(f"\n[server] Ingested {len(episodes)} episodes / {n_added} transitions into replay "
                    f"(size={self.replay_buffer.size() if self.replay_buffer else 0}).")
                
            # =================================================================
            # Auto-Shutdown Check
            # =================================================================
            if self.total_actor_timesteps >= Settings.SIMULATION_LENGTH:
                print(f"[server] Training Goal Reached! ({self.total_actor_timesteps}/{Settings.SIMULATION_LENGTH} steps). Initiating Shutdown.")
                
                # Signal the monitor loop to close the server
                async with self._terminate_lock:
                    self._should_terminate = True
                
                # Break out of the training loop immediately
                return
            # =================================================================

            # Calculate trainign steps per sample
            current_udt = self.total_weight_updates / max(1, self.total_actor_timesteps)
            if current_udt > self.max_utd:
                print(f"[server] UDT too high ({current_udt:.4f}), skipping training this round. total_weight_updates={self.total_weight_updates}, total_actor_timesteps={self.total_actor_timesteps}")
                await asyncio.sleep(1)
                continue  # skip training if UDT is already high

            if( self.replay_buffer.size() < self.learning_starts ):
                needed = self.learning_starts - self.replay_buffer.size()
                print(f"[server] Not training yet. Need {max(0, needed)} more samples.")
                await asyncio.sleep(self.train_every_seconds)
                continue

            # Train if we have enough samples
            if self.model is not None and self.replay_buffer is not None and self.replay_buffer.size() >= self.learning_starts:
                # Check if training is already in progress (prevent concurrent training)
                async with self._training_lock:
                    if self._training_in_progress:
                        print(f"[server] ⚠️ CONCURRENT TRAINING DETECTED! Training already running. Skipping this round.")
                        await asyncio.sleep(self.train_every_seconds)
                        continue
                    self._training_in_progress = True
                    print(f"[server] [LOOP_CONTROL] Starting training from event loop (loop iteration {loop_iteration_count})")
                
                # Run training in a separate thread to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                try:
                    await loop.run_in_executor(self._executor, self._train_step_blocking)
                except Exception as e:
                    print(f"[server] Error during training: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Mark training as complete
                    async with self._training_lock:
                        self._training_in_progress = False
                    elapsed = time.time() - loop_start
                    print(f"[server] [LOOP_CONTROL] Training and sync completed, total loop time: {elapsed:.2f}s")
            else:
                needed = self.learning_starts - (self.replay_buffer.size() if self.replay_buffer else 0)
                print(f"[server] Not training yet. Need {max(0, needed)} more samples.")
            
            # Wait before next training iteration
            await asyncio.sleep(self.train_every_seconds)


    def _train_step_blocking(self):
        """
        The actual training loop - runs in a worker thread to avoid blocking the asyncio event loop.
        This is the blocking version of training that was previously in _train_loop.
        """
        # Pin this thread to specific CPU cores to avoid contention with event loop
        import os
        import threading
        try:
            cpu_count = os.cpu_count() or 4
            # Use cores 8-15 for training (leaves cores 1-7 for client simulator)
            # Core 0 is reserved for event loop
            training_cores = set(range(8, min(16, cpu_count)))  # Cores 8-15 (or up to cpu_count)
            
            if os.name == 'nt':
                # Windows: use SetThreadAffinityMask
                import ctypes
                mask = 0
                for core in training_cores:
                    mask |= (1 << core)
                ctypes.windll.kernel32.SetThreadAffinityMask(ctypes.windll.kernel32.GetCurrentThread(), mask)
                print(f"[server] Training thread pinned to CPU cores: {training_cores}")
            else:
                # Linux/macOS: use sched_setaffinity
                os.sched_setaffinity(0, training_cores)
                print(f"[server] Training thread pinned to CPU cores: {training_cores}")
        except Exception as e:
            print(f"[server] Warning: Could not set CPU affinity: {e}")
        
        # Also reduce thread priority
        try:
            if os.name == 'nt':
                import ctypes
                ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
            else:
                os.nice(5)
        except Exception as e:
            print(f"[server] Warning: Could not reduce thread priority: {e}")
        
        t_total_start = time.time()
        
        # Check termination before starting training
        async_task = asyncio.run_coroutine_threadsafe(self._check_should_terminate(), self._loop_ref)
        if async_task.result():
            return

        grad_steps = self.grad_steps
        current_udt = self.total_weight_updates / max(1, self.total_actor_timesteps)
        print(f"\n[server] Training SAC... steps={grad_steps} | bs={self.model.batch_size} | buffer size={self.replay_buffer.size()} | UDT={current_udt:.4f}")
        time_start_training = time.time()

        # Manual training loop for SAC using torch
        actor = self.model.policy.actor
        critic = self.model.policy.critic
        critic_target = self.model.policy.critic_target
        actor_optimizer = self.model.policy.actor.optimizer
        critic_optimizer = self.model.policy.critic.optimizer
        batch_size = self.model.batch_size
        replay_buffer = self.model.replay_buffer
        gamma = self.model.gamma
        n_step_gamma = gamma ** self.n_step
        tau = self.model.tau
        ent_coef = self.model.ent_coef
        target_entropy = self.model.target_entropy
        log_ent_coef = getattr(self.model, "log_ent_coef", None)
        ent_coef_optimizer = getattr(self.model, "ent_coef_optimizer", None)

        if self.custom_sampling and self.replay_buffer.dynamic_importance_sampling_corrector:
            progress = min(1, self.total_actor_timesteps / self.replay_buffer.beta_annealing_horizon)
            self.replay_buffer.beta = self.replay_buffer.initial_beta + progress *(self.replay_buffer.beta_end - self.replay_buffer.initial_beta)
            print("Beta has been updated to: " + str(self.replay_buffer.beta))

        actor_loss = None
        critic_loss = None
        ent_coef_loss = None
                            
        for step in range(grad_steps):
            # Check termination periodically (less frequently to avoid overhead)
            if step % 100 == 0:
                async_task = asyncio.run_coroutine_threadsafe(self._check_should_terminate(), self._loop_ref)
                if async_task.result():
                    print(f"[server] Training interrupted at step {step}/{grad_steps} due to termination request")
                    return
            
            then = time.time()
            try:
                # Reduce batch size to avoid OOM
                safe_batch_size = min(batch_size, 4096)
                data = replay_buffer.sample(safe_batch_size)

                obs      = data.observations
                actions  = data.actions
                next_obs = data.next_observations
                rewards  = data.rewards
                dones    = data.dones
                steps_taken = data.steps_taken
                is_weights = data.is_weights

                # Critic update
                with torch.no_grad():
                    next_actions, next_log_prob = self.model.policy.actor.action_log_prob(next_obs)
                    target_q1, target_q2 = critic_target(next_obs, next_actions)
                    target_q = torch.min(target_q1, target_q2)
                    discounts = torch.pow(gamma, steps_taken)

                    if log_ent_coef is not None:
                        ent_coef = torch.exp(log_ent_coef.detach())
                    # Ensure next_log_prob shape is [batch_size, 1]
                    if next_log_prob.dim() == 2 and next_log_prob.shape[1] != 1:
                        next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
                    else:
                        next_log_prob = next_log_prob.view(-1, 1)

                    target_q = rewards + discounts * (1 - dones) * (target_q - ent_coef * next_log_prob)

                    # Debug: print sampled-batch statistics occasionally
                    try:
                        if Settings.SAC_DEBUG_LOGGING and (step % 50 == 0):
                            rt = rewards.detach().cpu().numpy().reshape(-1)
                            st = steps_taken.detach().cpu().numpy().reshape(-1)
                            disc = discounts.detach().cpu().numpy().reshape(-1)
                            nlp = next_log_prob.detach().cpu().numpy().reshape(-1)
                            tq = target_q.detach().cpu().numpy().reshape(-1)
                            iw = is_weights.detach().cpu().numpy().reshape(-1)
                            import numpy as _np
                            print(f"[SAC DEBUG][step={step}] batch_size={rt.shape[0]} | steps mean={st.mean():.3f}, min={st.min()}, max={st.max()} | reward mean={rt.mean():.4f}, std={rt.std():.4f}")
                            print(f"[SAC DEBUG][step={step}] discounts mean={disc.mean():.4f} | next_logprob mean={nlp.mean():.4f}, ent_coef={float(ent_coef) if 'ent_coef' in locals() else None}")
                            print(f"[SAC DEBUG][step={step}] target_q mean={tq.mean():.4f}, min={tq.min():.4f}, max={tq.max():.4f} | is_weights mean={iw.mean():.4f}")
                    except Exception:
                        pass

                current_q1, current_q2 = critic(obs, actions)

                critic_loss = ((is_weights * ((current_q1 - target_q).pow(2))).mean() 
                               + (is_weights * ((current_q2 - target_q).pow(2))).mean())
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Get TD-error
                if self.replay_buffer.custom_sampling:
                    with torch.no_grad():
                        current_q1_new, current_q2_new = critic(obs, actions)
                        TD_error_1 = torch.abs(target_q - current_q1_new)
                        TD_error_2 = torch.abs(target_q - current_q2_new)

                        TD_update_priorities = ((TD_error_1 + TD_error_2) / 2.0)
                        TD_update_priorities = TD_update_priorities.cpu().numpy().flatten()
                        TD_update_priorities += 1e-6

                        if self.replay_buffer.stat_tracker:
                            self.replay_buffer.stat_tracker.batch_update_TD_errors(replay_buffer.current_sampled_inds, TD_update_priorities)

                        # optional clipping
                        if replay_buffer.clip_weights:
                            TD_update_priorities = np.clip(TD_update_priorities, 1e-6, 1.0)

                        replay_buffer.update_TD_priorities(TD_update_inds = replay_buffer.current_sampled_inds, TD_update_priorities = TD_update_priorities)

                # Actor update
                new_actions, log_prob = actor.action_log_prob(obs)
                q1_new, q2_new = critic(obs, new_actions)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (ent_coef * log_prob - q_new).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Entropy coefficient update
                if log_ent_coef is not None and ent_coef_optimizer is not None:
                    ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
                    ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    ent_coef_optimizer.step()

                # Soft update of target network
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                self.total_weight_updates += 1
                
                # Yield CPU time periodically to let client/event loop breathe
                if step % 8 == 0:  # Every 8 steps (~3% of 256 total steps = 32 yields)
                    time.sleep(0.002)  # 2ms sleep to yield CPU

            except torch.cuda.OutOfMemoryError as e:
                print(f"[server] CUDA OOM during training step {step}: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                continue

        # Stat tracker save
        if self.replay_buffer.stat_tracker is not None:
            now = time.time()
            if now - self._last_stat_save_ts > 60.0:
                self.replay_buffer.stat_tracker.print_stats()
                self._last_stat_save_ts = now
        
        # Check termination before broadcasting
        async_task = asyncio.run_coroutine_threadsafe(self._check_should_terminate(), self._loop_ref)
        if async_task.result():
            print("[server] Termination requested, skipping post-training save/broadcast")
            return
        
        # Send updates back via the event loop
        try:
            episodes = self.episode_buffer.drain_all()  # Get episodes from buffer
            
            log_dict = {
                "actor_loss": actor_loss.item() if actor_loss is not None else 0,
                "critic_loss": critic_loss.item() if critic_loss is not None else 0,
                "ent_coef_loss": ent_coef_loss.item() if ent_coef_loss is not None else 0,
                "ent_coef": ent_coef if isinstance(ent_coef, float) else (ent_coef.item() if ent_coef is not None else 0),
                "total_weight_updates": self.total_weight_updates,
                "training_duration": time.time() - time_start_training,
                "total_timesteps": self.total_actor_timesteps,
                "UDT": self.total_weight_updates / max(1, self.total_actor_timesteps),
            }
            print(f"\n[server] Training completed in {(time.time() - time_start_training):.2f} seconds.")
            if(len(episodes) > 0):
                print(log_dict)
                
                # Timing: CSV logging
                t_csv_start = time.time()
                self.trainingLogHelper.log_to_csv(self.model, episodes, log_dict)
                t_csv_elapsed = time.time() - t_csv_start
                print(f"[TIMING] CSV logging took {t_csv_elapsed:.3f}s")
                
                # Schedule plotting asynchronously if needed (don't block worker thread)
                if self.trainingLogHelper.should_plot():
                    asyncio.run_coroutine_threadsafe(
                        self._schedule_plot(),
                        self._loop_ref
                    )

            # Timing: Weight serialization
            t_weights_start = time.time()
            new_blob = SacUtilities.state_dict_to_bytes(self.model.policy.actor.state_dict())
            self._weights_blob = new_blob
            t_weights_elapsed = time.time() - t_weights_start
            print(f"[TIMING] Weight serialization took {t_weights_elapsed:.3f}s")
            
            # Broadcast weights via the async loop (rate-limited to avoid flooding client)
            now = time.time()
            time_since_last_broadcast = now - self._last_weight_broadcast_time
            if time_since_last_broadcast >= self._min_broadcast_interval:
                asyncio.run_coroutine_threadsafe(self._broadcast_weights(new_blob), self._loop_ref)
                self._last_weight_broadcast_time = now
                print("\n[server] Trained SAC and broadcast updated actor weights.")
            else:
                print(f"\n[server] Trained SAC (weights broadcast rate-limited, {self._min_broadcast_interval - time_since_last_broadcast:.1f}s until next broadcast).")

            # Schedule model saving asynchronously (don't block worker thread)
            model_save_path = os.path.join(self.model_dir, str(self.save_model_name))
            asyncio.run_coroutine_threadsafe(
                self._schedule_model_save(model_save_path),
                self._loop_ref
            )
            
            # Overall timing
            t_total_elapsed = time.time() - t_total_start
            # if t_total_elapsed > 5.0:
            print(f"[TIMING] TOTAL train_step_blocking took {t_total_elapsed:.2f}s")
        except Exception as e:
            print(f"[server] Error in post-training operations: {e}")
            import traceback
            traceback.print_exc()

    async def _check_should_terminate(self) -> bool:
        """Helper to check termination flag from worker thread"""
        async with self._terminate_lock:
            return self._should_terminate

    async def _schedule_plot(self):
        """Schedule plotting in a background task (non-blocking)."""
        try:
            # Run plotting in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.trainingLogHelper.plot_training_metrics)
        except Exception as e:
            print(f"[server] Error scheduling plot: {e}")

    async def _schedule_model_save(self, save_path: str):
        """Schedule model saving in a background task (non-blocking)."""
        try:
            # Run model saving in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self.model.save, save_path)
        except Exception as e:
            print(f"[server] Error auto-saving model: {e}")

    # ---------- networking ----------
    async def _broadcast_weights(self, blob: bytes):
        """Send weights to all currently connected clients."""
        t_bc_start = time.time()
        frame = {"type": "weights", "data": {"blob": blob, "format": "torch_state_dict", "algo": "SAC", "module": "actor"}}
        async with self._client_lock:
            dead: List[asyncio.StreamWriter] = []
            for w in self._clients:
                try:
                    w.write(pack_frame(frame))
                    await w.drain()
                except Exception:
                    dead.append(w)
            for w in dead:
                self._clients.discard(w)
        t_bc_elapsed = time.time() - t_bc_start
        if t_bc_elapsed > 0.1:
            print(f"[TIMING] Broadcast weights took {t_bc_elapsed:.3f}s (blob size: {len(blob)} bytes)")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"[server] Client connected: {addr}")
        async with self._client_lock:
            self._clients.add(writer)

        # ack
        try:
            writer.write(pack_frame({"type": "ack", "data": {"msg": "connected"}}))
            await writer.drain()
        except Exception:
            pass

        # ALWAYS send current weights if we have them (even when init_from_scratch=True)
        if self._weights_blob is not None:
            try:
                await self._broadcast_weights(self._weights_blob)
                print(f"[server] Weights sent to {addr} (bytes={len(self._weights_blob)})")
            except Exception as e:
                print(f"[server] Failed to send initial weights to {addr}: {e}")

        # read loop
        try:
            while True:
                msg = await read_frame(reader)
                if msg.get("type") == "transition_batch":
                    d = msg.get("data", {})
                    obs_list      = [blob_to_np(x) for x in d.get("obs", [])]
                    act_list      = [blob_to_np(x) for x in d.get("action", [])]
                    next_obs_list = [blob_to_np(x) for x in d.get("next_obs", [])]
                    reward_list   = d.get("reward", [])
                    done_list     = d.get("done", [])
                    info_list     = d.get("info", [])

                    # rebuild transitions for one episode
                    episode = []
                    for i in range(len(reward_list)):
                        episode.append({
                            "obs":      obs_list[i],
                            "action":   act_list[i],
                            "next_obs": next_obs_list[i],
                            "reward":   float(reward_list[i]),
                            "done":     bool(done_list[i]),
                            "info":     info_list[i] if i < len(info_list) else {},
                            "actor_id": int(d.get("actor_id", -1)),
                        })

                    if Settings.RL_CRASH_REWQARD_RAMPING:
                        episode = self._apply_crash_ramp(episode, ramp_steps=20, max_ramp_value=5.0)

                    # initialize shapes if this is the very first episode
                    if self.model is None:
                        self._initialize_model()

                    self.last_episode_time = time.time()

                    self.episode_buffer.add_episode(episode)
                    # print(f"[server] Stored episode: {len(episode)} transitions "
                    #     f"(total episodes pending train: {len(self.episode_buffer.episodes)})")
                    self.total_actor_timesteps += len(episode)
                    # optional ack
                    try:
                        writer.write(pack_frame({"type": "episode_ack", "data": {"n": len(episode)}}))
                        await writer.drain()
                    except Exception:
                        pass
                elif msg.get("type") == "clear_buffer":
                    d = msg.get("data", {})
                    actor_id = int(d.get("actor_id", -1))
                    
                    # Clear the episode buffer
                    episodes_cleared = len(self.episode_buffer.episodes)
                    self.episode_buffer.episodes.clear()
                    
                    # Clear the replay buffer if it exists
                    if self.replay_buffer is not None:
                        replay_size_before = self.replay_buffer.size()
                        self.replay_buffer.reset()
                        print(f"[server] Cleared replay buffer (had {replay_size_before} transitions)")
                    
                    print(f"[server] Cleared {episodes_cleared} episodes from buffer (requested by actor {actor_id})")
                    
                    # Send acknowledgment
                    try:
                        writer.write(pack_frame({"type": "clear_buffer_ack", "data": {"episodes_cleared": episodes_cleared}}))
                        await writer.drain()
                    except Exception:
                        pass
                elif msg.get("type") == "terminate":
                    d = msg.get("data", {})
                    actor_id = int(d.get("actor_id", -1))
                    
                    print(f"[server] Received terminate message from actor {actor_id}")
                    
                    # Save the model
                    self._save_model()
                    
                    # Set termination flag
                    async with self._terminate_lock:
                        self._should_terminate = True
                    
                    # Send acknowledgment
                    try:
                        writer.write(pack_frame({"type": "terminate_ack", "data": {"msg": "terminating"}}))
                        await writer.drain()
                    except Exception:
                        pass
                    
                    # Break out of the read loop
                    break
                else:
                    # ignore other messages
                    pass
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[server] Client loop error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            async with self._client_lock:
                self._clients.discard(writer)
            print(f"[server] Client disconnected: {addr}")

    async def run(self):
        # Pin event loop thread to single core for isolation (leave rest for training)
        import os
        try:
            cpu_count = os.cpu_count() or 4
            # Use only core 0 for event loop (it's truly single-threaded for asyncio)
            event_loop_cores = {0}
            
            if os.name == 'nt':
                import ctypes
                mask = 1  # Mask for core 0
                ctypes.windll.kernel32.SetThreadAffinityMask(ctypes.windll.kernel32.GetCurrentThread(), mask)
                print(f"[server] Event loop thread pinned to CPU core: 0")
            else:
                os.sched_setaffinity(0, event_loop_cores)
                print(f"[server] Event loop thread pinned to CPU core: 0")
        except Exception as e:
            print(f"[server] Warning: Could not set event loop CPU affinity: {e}")
        
        # Store reference to the event loop for worker threads
        self._loop_ref = asyncio.get_event_loop()
        
        # Eagerly initialize so we have weights ready for the first client:
        if self.model is None:
            self._load_base_model() 

        # start trainer task and keep reference for cleanup
        train_task = asyncio.create_task(self._train_loop())

        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(
            f"[server] Listening on {addrs} | save_model='{self.save_model_name}' | load_model='{self.load_model_name}' | device='{self.device}'"
        )
        
        # Create a task to monitor termination flag
        async def monitor_termination():
            while True:

                if self.last_episode_time is not None:
                    time_since_last_ep = time.time() - self.last_episode_time
                    
                    if time_since_last_ep > self.episode_timeout:
                        print(f"[server] Timeout: No episodes received for {time_since_last_ep:.1f}s. Initiating shutdown.")
                        async with self._terminate_lock:
                            self._should_terminate = True

                async with self._terminate_lock:
                    if self._should_terminate:
                        print("[server] Termination flag set, shutting down server...")
                        server.close()
                        await server.wait_closed()
                        print("[server] Server shut down gracefully")
                        return
                await asyncio.sleep(0.5)  # Check every 0.5 seconds
        
        monitor_task = asyncio.create_task(monitor_termination())
        
        try:
            async with server:
                # Wait for monitor task to complete (which happens when termination is requested)
                await monitor_task
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n[server] Received interrupt signal (Ctrl+C), shutting down...")
        finally:
            # Set termination flag
            async with self._terminate_lock:
                self._should_terminate = True
            
            # Cancel training task
            if not train_task.done():
                print("[server] Cancelling training task...")
                train_task.cancel()
                try:
                    await train_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel monitor task if still running
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Close all client connections
            async with self._client_lock:
                print(f"[server] Closing {len(self._clients)} client connection(s)...")
                for writer in list(self._clients):
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
                self._clients.clear()
            
            # Close server
            server.close()
            await server.wait_closed()

            # Nikita: final stat tracker save
            if self.replay_buffer is not None and self.replay_buffer.stat_tracker is not None:
                self.replay_buffer.stat_tracker.save_csv(append=False)
            
            # Save model one last time (with timeout)
            print("[server] Saving model before exit...")
            try:
                save_task = asyncio.create_task(self._schedule_model_save(
                    os.path.join(self.model_dir, self.save_model_name)
                ))
                await asyncio.wait_for(save_task, timeout=5.0)
            except asyncio.TimeoutError:
                print("[server] Model save timed out, skipping...")
            except Exception as e:
                print(f"[server] Error saving model: {e}")
            
            # Shutdown executor
            print("[server] Shutting down executor...")
            self._executor.shutdown(wait=False, cancel_futures=True)
            
            print("[server] Shutdown complete")

