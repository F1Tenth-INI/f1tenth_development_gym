#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import ast
import os
import subprocess
import hashlib
import shutil
from typing import List, Optional, NamedTuple

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
        self.use_is_weights_for_actor = Settings.SAC_USE_IS_WEIGHTS_FOR_ACTOR

        self.current_sampled_inds = None #indexes of the data which was just sampled in last sample() call

        self.clip_weights = Settings.SAC_CLIP_WEIGHTS

        self.rank_based_sampling = Settings.SAC_RANK_BASED_SAMPLING

        self.custom_sampling_replace = Settings.SAC_CUSTOM_SAMPLING_REPLACE

        self.steps_taken = np.ones(self.buffer_size, dtype=np.int32)

        self.recalc_every = 20  # choose N
        self._sample_calls = 0
        self._cached_p = None
        self._cached_possible_inds = None
        self._cached_ranked_inds = None
        self._cached_length = 0

        self.log_squish = Settings.SAC_LOG_SQUISH

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
            self.stat_tracker.register_transition(obs, action, idx, reward, done, info=infos) #gets the unnormalized obs directly
            # self.stat_tracker.print_stats()
            # if self.stat_tracker_full_obs_action_save:
            #     self.stat_tracker.save_full_obs_action(obs, action)

        if not self.custom_sampling:
            return

        #store steps taken for each transition over whole buffer
        self.steps_taken[idx] = steps_taken

        #set to max samplepriority for newest transitions
        #TODO: option to have this be able to be set in settings
        self.TD_weights[idx] = self.new_weight_priority

        #TODO: the observations are normalized, is that bad or not?
        #compute weight
        d = obs[80] #-> these were off by one the whole time......
        e = obs[81]
        # rew = self.rewards[idx, 0] + 1


        rew = float(self.rewards[idx, 0])
        rew_clipped = np.clip(rew, -15.0, 1.0)
        rew_norm = (rew_clipped + 15.0) / 16.0
        rew_final = (1e-3 + rew_norm) ** 2.0

        # Scale the reward contribution to the priority by the number of steps in the n-step transition.
        # Without this, longer n-step transitions are over-prioritised.
        # Use per-step average reward for priority computation.
        try:
            reward_per_step = float(rew_final) / max(1, int(steps_taken))
        except Exception:
            reward_per_step = float(rew_final)

        #velx = obs[0] and vely = obs[1]
        vel = np.sqrt(obs[0]**2 + obs[1]**2)
        # vel = (1/(1-e)) * vel #we only care about speed into the right direction
        vel = (1 - min(e, 0.9)) * vel #we only care about speed into the right direction
        # print(rew)

        state_w = self.w_d * abs(d) + self.w_e * abs(e) + self.velocity_weight * abs(vel)
        reward_w = self.reward_weight * abs(reward_per_step)
        w = state_w + reward_w

        w = np.clip(w, 1e-6, 1e3)

        if self.clip_weights:
            w = np.clip(w, 0, 5)

        self.state_weights[idx] = w
        
        return
    
    def update_TD_priorities(self, TD_update_inds: np.ndarray, TD_update_priorities: np.ndarray):
        self.TD_weights[TD_update_inds] = TD_update_priorities

        """NIKITA: testing out more rigorous implementation, however makes more computation: TODO: see if this is worth, maybe remove"""
        # Calculate true maximum of combined weights across entire buffer
        # Account for the buffer not being fully filled initially
        valid_length = self.pos if not self.full else self.buffer_size
        combined_all = (self.TD_weights[:valid_length] * (1 - self.state_to_TD_ratio) + 
                        self.state_weights[:valid_length] * self.state_to_TD_ratio)
        
        self.new_weight_priority = combined_all.max() if len(combined_all) > 0 else 1.0

        """ OLD """
        # #get newest max prio values for new transitions
        # self.new_weight_priority = max(self.new_weight_priority, TD_update_priorities.max())

        return
    
    # def _recompute_probs(self, possible_inds):
    #     w_vec = self.state_weights[possible_inds].astype(np.float64)
    #     TD_vec = self.TD_weights[possible_inds].astype(np.float64)
    #     combined = TD_vec * (1 - self.state_to_TD_ratio) + w_vec * self.state_to_TD_ratio
    #     combined = np.log1p(combined)

    #     if self.rank_based_sampling:
    #         sorted_inds = np.argsort(-combined)
    #         ranked_buffer_inds = possible_inds[sorted_inds]
    #         ranks = np.arange(1, len(possible_inds) + 1)
    #         p = ranks ** -self.alpha
    #     else:
    #         p = combined ** self.alpha
    #         ranked_buffer_inds = None

    #     p_tot = p.sum()
    #     if p_tot <= 0 or not np.isfinite(p_tot):
    #         p = np.ones_like(p) / len(possible_inds)
    #     else:
    #         p /= p_tot

    #     self._cached_p = p
    #     self._cached_possible_inds = possible_inds
    #     self._cached_ranked_inds = ranked_buffer_inds
    #     self._cached_length = len(possible_inds)

    # def sample_recompute(self, safe_batch_size, env=None):
    #     if not self.custom_sampling:
    #         return super().sample(batch_size=safe_batch_size, env=env)

    #     # compute possible_inds (same as you already do)
    #     if self.full:
    #         possible_inds = np.arange(self.buffer_size)
    #         mask = possible_inds != self.pos #mask which would set the self.pos index as false
    #         possible_inds = possible_inds[mask]
    #     else:
    #         possible_inds = np.arange(self.pos)

    #     self._sample_calls += 1
    #     need_recalc = (
    #         self._cached_p is None
    #         or self._sample_calls % self.recalc_every == 0
    #         or self._cached_length != len(possible_inds)
    #     )

    #     if need_recalc:
    #         print("RECOMPUTING PROBABILITIES FOR SAMPLING...")
    #         self._recompute_probs(possible_inds)
    #     # else:
    #         # print("not recomputing prob :()")

    #     p = self._cached_p
    #     sampled_p_index = np.random.choice(self._cached_length, size=safe_batch_size, p=p)

    #     if self.rank_based_sampling:
    #         batch_inds = self._cached_ranked_inds[sampled_p_index]
    #     else:
    #         batch_inds = self._cached_possible_inds[sampled_p_index]

    #     # IS weights based on cached p
    #     sample_probs = p[sampled_p_index]
    #     is_weights = (1 / (self._cached_length * sample_probs)) ** self.beta
    #     is_weights = is_weights / is_weights.max()

    #     self.batch_is_correctors = is_weights.reshape(-1, 1).astype(np.float32)
    #     self.current_sampled_inds = batch_inds

    #     return self._get_samples(batch_inds, env=env)

    def sample(self, safe_batch_size: int, invert_TD = False, env=None):
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

        if invert_TD:
            TD_vec = np.clip(TD_vec, 1e-6, None) #avoid exploding
            TD_vec = 1 / TD_vec
            TD_vec = TD_vec / TD_vec.max() #rescale so that max is 1 again

        combined_weight = TD_vec * (1 - self.state_to_TD_ratio) + w_vec * self.state_to_TD_ratio

        # Squish priorities logarithmically to dampen outliers while preserving rank
        if self.log_squish:
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

        #TODO: do i want replace or not? :/
        sampled_p_index = np.random.choice(length, size=safe_batch_size, p=p, replace=self.custom_sampling_replace)

        #DEBUG
        # print(p)
        
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

        return self._get_samples(batch_inds, env=env)
    

    def sample_uniform(self, safe_batch_size: int, env=None):
        """default SB3 uniform sampling"""
        return super().sample(batch_size=safe_batch_size, env=env)
    

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> WeightedReplayBufferSamples:

        if self.stat_tracker is not None:
            self.stat_tracker.batch_update_sample_count(batch_inds) #this is where stat_tracker gets +1 on sample count

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
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        train_frequency: int = 1,
        save_replay_buffer: bool = False,
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
        self.save_replay_buffer_enabled = bool(save_replay_buffer)
        self._latest_training_info_payload: Optional[dict] = None

        # Settings
        self.learning_starts = learning_starts
        self.grad_steps = grad_steps

        self.init_from_scratch = None
        # networking
        self._clients: set[asyncio.StreamWriter] = set()
        self._client_lock = asyncio.Lock()
        self._should_terminate = False
        self._terminate_lock = asyncio.Lock()

        # data
        self.episode_buffer = EpisodeReplayBuffer(capacity_episodes=2000)
        self._weights_blob: Optional[bytes] = None

        # RL bits (lazy init fallback; but we’ll eager-init in run())
        self.model: Optional[SAC] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.vecnorm: Optional[VecNormalize] = None
        self.total_actor_timesteps = 0
        self.total_weight_updates = 0
        self._training_paused_for_udt = False

        self.last_episode_time = None 
        self.episode_timeout = 100.0

        #used for n-step buffer, for standard buffer set = 1
        self.n_step = getattr(Settings, "SAC_N_STEP", 1)
        # self.n_step_discount_factor = self.discount_factor ** self.n_step
        self.custom_sampling = Settings.USE_CUSTOM_SAC_SAMPLING
        self.critic_invert_TD = Settings.SAC_CUSTOM_CRITIC_INVERT_TD
        self.actor_invert_TD = Settings.SAC_CUSTOM_ACTOR_INVERT_TD

        self.save_model_checkpoints = Settings.SAC_SAVE_MODEL_CHECKPOINTS
        self.checkpoint_frequency = Settings.SAC_CHECKPOINT_FREQUENCY
        self.last_checkpoint_timestep = 0
        self._last_broadcast_training_ready: Optional[bool] = None

        self.pre_fill_with_pp = Settings.SAC_PREFILL_BUFFER_WITH_PP
        self.pre_fill_amount = Settings.SAC_PREFILL_BUFFER_WITH_PP_AMOUNT
        self.pre_fill_sac_epochs = Settings.SAC_PREFILL_BEHAVIOR_CLONING_EPOCHS

        self._bc_prefill_done = False
        self._bc_in_progress = False
        self.total_prefill_timesteps = 0
        self._bc_actor_fingerprint: Optional[str] = None
        self._printed_first_sac_fingerprint = False

        #if we want to prefill with PP, change learning starts to be equal to prefill amount
        if self.pre_fill_with_pp:
            self.learning_starts = self.pre_fill_amount
            print(f"[LearnerServer] Pre-filling enabled: setting learning_starts to {self.learning_starts}.")


        # file paths: default save name fallback
        if self.save_model_name is None:
            # If save name not provided, fall back to load name if present, else use a generic default
            if self.load_model_name is not None:
                self.save_model_name = self.load_model_name
            else:
                self.save_model_name = "SAC_RCA1_0"

        # resolve model root and client subfolder
        self.save_model_path, self.model_dir = SacUtilities.resolve_model_paths(self.save_model_name)
        self.client_model_dir = os.path.join(self.model_dir, "client")
        self.replay_buffer_csv_path = os.path.join(self.model_dir, "replay_buffer.csv")
        os.makedirs(self.client_model_dir, exist_ok=True)
        self.save_model_path = os.path.join(self.model_dir, self.save_model_name)

        # resolve load path if different
        self.load_model_path = None
        self.load_model_dir = None
        if self.load_model_name is not None:
            load_model_path_root, load_model_dir_root = SacUtilities.resolve_model_paths(self.load_model_name)
            load_model_path_server = os.path.join(load_model_dir_root, "server", self.load_model_name)
            self.load_model_dir = load_model_dir_root
            # Prefer the model root, but keep legacy support for older ".../server/" layouts.
            self.load_model_path = load_model_path_root if os.path.exists(load_model_path_root) else load_model_path_server

        self.trainingLogHelper = TrainingLogHelper(self.save_model_name, self.model_dir)

        self._ensure_client_observation_builder()
        SacUtilities.zip_relevant_files(self.model_dir)
        # Model + replay buffer are initialized lazily after we see the first observation.
        # This allows per-model custom observation builders to define different obs_dim.

    # ---------- setup / init ----------
    def _sync_save_model_from_load_model(self) -> None:
        """
        If load/save model names differ, copy all artifacts from load model dir
        into save model dir so training continues from a full clone (weights + replay CSV + metadata).
        """
        if self.load_model_name is None:
            return
        if self.save_model_name == self.load_model_name:
            return
        if self.load_model_dir is None or not os.path.isdir(self.load_model_dir):
            print(f"[server] Load model directory missing; cannot sync: {self.load_model_dir}")
            return

        try:
            os.makedirs(self.model_dir, exist_ok=True)
            for entry in os.listdir(self.load_model_dir):
                src_path = os.path.join(self.load_model_dir, entry)
                dst_path = os.path.join(self.model_dir, entry)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)

            # Ensure the save-model zip name exists so downstream load uses the new model name.
            src_zip = os.path.join(self.model_dir, f"{self.load_model_name}.zip")
            dst_zip = os.path.join(self.model_dir, f"{self.save_model_name}.zip")
            if os.path.isfile(src_zip) and src_zip != dst_zip:
                shutil.copy2(src_zip, dst_zip)

            # After sync, load from the save-model path so replay/model are coupled to the cloned folder.
            self.load_model_path = os.path.join(self.model_dir, self.save_model_name)
            print(
                f"[server] Synced model artifacts '{self.load_model_name}' -> "
                f"'{self.save_model_name}' in {self.model_dir}"
            )
        except Exception as e:
            print(f"[server] Failed to sync model artifacts from '{self.load_model_name}' to '{self.save_model_name}': {e}")

    def _ensure_client_observation_builder(self) -> None:
        target_builder_path = os.path.join(self.client_model_dir, "observation_builder.py")

        source_builder_path: Optional[str] = None
        if self.load_model_name is not None:
            _, load_model_dir_root = SacUtilities.resolve_model_paths(self.load_model_name)
            load_client_builder = os.path.join(load_model_dir_root, "client", "observation_builder.py")
            if os.path.isfile(load_client_builder):
                source_builder_path = load_client_builder

        if source_builder_path is None:
            default_builder = os.path.join(os.path.dirname(__file__), "observation_builder_template.py")
            if os.path.isfile(default_builder):
                source_builder_path = default_builder

        if source_builder_path is None:
            print("[server] No observation_builder_template.py source found.")
            return

        try:
            os.makedirs(self.client_model_dir, exist_ok=True)
            if os.path.abspath(source_builder_path) != os.path.abspath(target_builder_path):
                shutil.copyfile(source_builder_path, target_builder_path)
            print(f"[server] Using client observation builder: {target_builder_path}")
        except Exception as e:
            print(f"[server] Failed to prepare client observation builder: {e}")

    def _load_base_model(self, obs_dim: int):
        """
        Initialize the model + replay buffer.

        - If `load_model_name` is set, attempt to load from that path.
        - Else: create a new model (train from scratch).
        """
        dummy_env = SacUtilities.create_vec_env_from_obs_dim(obs_dim)

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
            self.replay_buffer.stat_tracker = StatTracker(
                save_dir=stat_save_dir,
                save_name="stats_log.csv",
                max_buffer_size=self.replay_capacity,
                extended_obs_action_save=Settings.SAC_STAT_TRACKER_FULL_OBS_ACTION_SAVE,
                csv_float_decimals=Settings.SAC_STAT_TRACKER_CSV_FLOAT_DECIMALS,
            )
            self._last_stat_save_ts = 0.0
        else:
            self.replay_buffer.stat_tracker = None

    
        self.replay_buffer.stat_tracker_full_obs_action_save = Settings.SAC_STAT_TRACKER_FULL_OBS_ACTION_SAVE


        self.model.replay_buffer = self.replay_buffer
        self._load_replay_buffer()
        # Restore historical counters from metrics, then ensure loaded replay
        # samples are reflected in the UDT denominator.
        self._restore_training_counters_from_metrics()
        self.total_actor_timesteps = max(
            self.total_actor_timesteps,
            int(self.replay_buffer.size()),
        )

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


    def _initialize_model(self, obs_dim: int):
        if self.model is not None:
            return
        self._load_base_model(obs_dim=obs_dim)
        print(f"[server] Initialized model (obs_dim={obs_dim})")

    def _expected_obs_dim(self) -> Optional[int]:
        """Return the current model observation dimension, if initialized."""
        if self.model is None:
            return None
        try:
            return int(self.model.observation_space.shape[0])
        except Exception:
            return None

    def _save_model(self):
        """Save the current model to disk (uses save_model_name)."""
        if self.model is not None:
            try:
                self.model.save(self.save_model_path)
                print(f"[server] Model saved to {self.save_model_path}")
            except Exception as e:
                print(f"[server] Error saving model: {e}")
        self.save_replay_buffer()

    def _buffer_obs_at(self, idx: int) -> np.ndarray:
        if self.replay_buffer is None:
            return np.array([], dtype=np.float32)
        return np.asarray(self.replay_buffer.observations[idx, 0], dtype=np.float32).reshape(-1)

    def _buffer_next_obs_at(self, idx: int) -> np.ndarray:
        if self.replay_buffer is None:
            return np.array([], dtype=np.float32)
        next_observations = getattr(self.replay_buffer, "next_observations", None)
        if next_observations is not None:
            return np.asarray(next_observations[idx, 0], dtype=np.float32).reshape(-1)
        next_idx = (idx + 1) % self.replay_buffer.buffer_size
        return np.asarray(self.replay_buffer.observations[next_idx, 0], dtype=np.float32).reshape(-1)

    def save_replay_buffer(self, target_path: Optional[str] = None) -> None:
        """Persist the current replay buffer into CSV."""
        if not self.save_replay_buffer_enabled:
            return
        if self.replay_buffer is None:
            print("[server] Replay buffer not initialized; skipping replay save.")
            return

        size = int(self.replay_buffer.size())
        target_csv_path = target_path or self.replay_buffer_csv_path
        try:
            target_dir = os.path.dirname(target_csv_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            with open(target_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["obs", "action", "next_obs", "reward", "done"])
                if size > 0:
                    start_idx = int(self.replay_buffer.pos) if self.replay_buffer.full else 0
                    for i in range(size):
                        idx = (start_idx + i) % self.replay_buffer.buffer_size
                        obs = self._buffer_obs_at(idx).tolist()
                        next_obs = self._buffer_next_obs_at(idx).tolist()
                        action = np.asarray(self.replay_buffer.actions[idx, 0], dtype=np.float32).reshape(-1).tolist()
                        reward = float(np.asarray(self.replay_buffer.rewards[idx, 0]).item())
                        done = bool(np.asarray(self.replay_buffer.dones[idx, 0]).item())
                        writer.writerow([repr(obs), repr(action), repr(next_obs), reward, int(done)])
            print(f"[server] Saved replay buffer with {size} transition(s) to {target_csv_path}")
        except Exception as e:
            print(f"[server] Failed to save replay buffer: {e}")

    def _load_replay_buffer(self) -> None:
        """Load replay transitions from model_dir/replay_buffer.csv when available."""
        if self.replay_buffer is None:
            return

        self.replay_buffer.reset()
        if not os.path.isfile(self.replay_buffer_csv_path):
            print(f"[server] No replay CSV at {self.replay_buffer_csv_path}; starting with empty replay buffer.")
            return

        loaded = 0
        try:
            with open(self.replay_buffer_csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if loaded >= self.replay_capacity:
                        break
                    obs = np.asarray(ast.literal_eval(row["obs"]), dtype=np.float32)
                    action = np.asarray(ast.literal_eval(row["action"]), dtype=np.float32)
                    next_obs = np.asarray(ast.literal_eval(row["next_obs"]), dtype=np.float32)
                    reward = float(row["reward"])
                    done = bool(int(float(row["done"])))
                    self.replay_buffer.add(
                        obs=obs,
                        next_obs=next_obs,
                        action=action,
                        reward=reward,
                        done=done,
                        infos={},
                    )
                    loaded += 1
            print(f"[server] Loaded {loaded} transition(s) from {self.replay_buffer_csv_path}")
        except Exception as e:
            print(f"[server] Failed to load replay CSV ({e}); resetting replay buffer to empty.")
            self.replay_buffer.reset()

    def _restore_training_counters_from_metrics(self) -> None:
        """Restore cumulative counters from learning_metrics.csv (last valid row)."""
        metrics_path = self.trainingLogHelper.csv_path
        if not os.path.isfile(metrics_path):
            return

        try:
            with open(metrics_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if len(rows) < 2:
                return

            header = rows[0]
            expected_len = len(header)
            if expected_len == 0:
                return

            idx_updates = header.index("total_weight_updates") if "total_weight_updates" in header else None
            idx_timesteps = header.index("total_timesteps") if "total_timesteps" in header else None
            if idx_updates is None and idx_timesteps is None:
                return

            restored_updates: Optional[int] = None
            restored_timesteps: Optional[int] = None
            for row in reversed(rows[1:]):
                # Skip malformed rows (e.g. schema drift or partial writes).
                if len(row) != expected_len:
                    continue
                try:
                    if idx_updates is not None:
                        restored_updates = int(float(row[idx_updates]))
                    if idx_timesteps is not None:
                        restored_timesteps = int(float(row[idx_timesteps]))
                except Exception:
                    restored_updates = None
                    restored_timesteps = None
                    continue
                break

            if restored_updates is not None:
                self.total_weight_updates = max(self.total_weight_updates, restored_updates)
            if restored_timesteps is not None:
                self.total_actor_timesteps = max(self.total_actor_timesteps, restored_timesteps)

            if restored_updates is not None or restored_timesteps is not None:
                print(
                    "[server] Restored training counters from learning_metrics.csv: "
                    f"total_weight_updates={self.total_weight_updates}, "
                    f"total_actor_timesteps={self.total_actor_timesteps}"
                )
        except Exception as e:
            print(f"[server] Failed to restore counters from {metrics_path}: {e}")

    def _actor_fingerprint(self) -> str:
        """Create a stable fingerprint of the current actor weights for debugging handoff integrity."""
        if self.model is None:
            return "none"

        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, tensor in sorted(self.model.policy.actor.state_dict().items(), key=lambda x: x[0]):
                hasher.update(name.encode("utf-8"))
                hasher.update(tensor.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]


    # ---------- ingestion + training ----------
    # No normalization for now: obs arrive normalized already
    def _normalize_obs(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)
      
    
    def _dummy_vec_from_spaces(self, obs_space: spaces.Box, act_space: spaces.Box):
        return DummyVecEnv([lambda: _SpacesOnlyEnv(obs_space, act_space)])
    
    def _ingest_episodes_into_replay(self, episodes: List[List[dict]]) -> int:
        if self.model is None or self.replay_buffer is None:
            return 0
        expected_obs_dim = self._expected_obs_dim()
        dropped_for_dim_mismatch = 0
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
                    infos=t["info"], # could pass TimeLimit info here if you track it
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
                if expected_obs_dim is not None:
                    obs_dim = int(np.asarray(obs).reshape(-1).shape[0])
                    next_obs_dim = int(np.asarray(next_obs).reshape(-1).shape[0])
                    if obs_dim != expected_obs_dim or next_obs_dim != expected_obs_dim:
                        dropped_for_dim_mismatch += 1
                        continue
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
        if dropped_for_dim_mismatch > 0:
            print(
                f"[server] Dropped {dropped_for_dim_mismatch} transition(s) due to "
                f"observation dim mismatch (expected {expected_obs_dim})."
            )
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
    
    async def _behavior_clone_on_prefill(self, num_epochs: int = 5, batch_size: int = 256):
        """
        Pre train actor only on PP data before real training, to combat the policy mismatch seen in early training of critic
        """

        print(f"\n[LearnerServer] Starting Behavior Cloning warmup ({num_epochs} epochs, {batch_size} batch_size)...")

        actor_optimizer = torch.optim.Adam(
            self.model.policy.actor.parameters(), 
            lr=self.model.learning_rate
        )

        for epoch in range(num_epochs):
            epoch_losses = []

            for step in range(100):
                if step % 10 == 0:
                    await asyncio.sleep(0)
                try:
                    batch = self.replay_buffer.sample(safe_batch_size=batch_size, env=None)

                    obs_tensor = torch.as_tensor(
                        batch.observations, 
                        dtype=torch.float32, 
                        device=self.device
                    )

                    pp_actions = torch.as_tensor(
                        batch.actions, 
                        dtype=torch.float32, 
                        device=self.device
                    )

                    # Forward pass through actor (deterministic BC target)
                    with torch.enable_grad():
                        # SB3 SAC actor returns action tensor of shape [batch, action_dim]
                        predicted_actions = self.model.policy.actor(obs_tensor, deterministic=True)

                    # MSE loss: match PP actions
                    bc_loss = torch.nn.functional.mse_loss(predicted_actions, pp_actions)
                    
                    # Backward pass
                    actor_optimizer.zero_grad()
                    bc_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.policy.actor.parameters(), 
                        max_norm=0.5
                    )
                    actor_optimizer.step()
                    
                    epoch_losses.append(bc_loss.item())
                    
                except Exception as e:
                    print(f"[LearnerServer] BC step error: {e}")
                    break
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                print(f"  [BC Epoch {epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.6f}")
        
        print(f"[LearnerServer] ✅ Behavior Cloning warmup complete!\n")
    
    
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

    def _save_model_checkpoint(self, checkpoint_name: str):
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        try:
            self.model.save(checkpoint_path)
            # Keep replay buffer in the model root and overwrite on each save.
            self.save_replay_buffer()
            print(f"[server] Checkpoint saved to {checkpoint_path} as {checkpoint_name}")
            self.last_checkpoint_timestep = self.total_actor_timesteps
        except Exception as e:
            print(f"[server] Error saving checkpoint: {e}")

    # def redistribute_crash_penalty(self, episode, ramp_steps: int = 20, crash_penalty: float = 15.0):
    #     crash_idx = None
    #     for i in reversed(range(len(episode))):
    #         t = episode[i]
    #         info = t.get("info", {})
    #         if t.get("done", False) and info.get("truncated", False):
    #             crash_idx = i
    #             break
    #     if crash_idx is None:
    #         # no crash in this episode -> leave rewards unchanged
    #         return episode

    #     #remove original penalty
    #     episode[crash_idx]["reward"] += crash_penalty

    #     # closer to crash -> larger share
    #     for k in range(ramp_steps):
    #         idx = crash_idx - k
    #         if idx < 0:
    #             break
    #         t = episode[idx]
    #         # weight from 0..1, increasing towards crash
    #         w = (ramp_steps - k) / ramp_steps
    #         shaped_penalty = w * (crash_penalty/2) #divide by 2 so total penalty is not too high
    #         t["reward"] -= shaped_penalty
    #     return episode

    def _save_model_checkpoint(self, checkpoint_name: str):
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        try:
            self.model.save(checkpoint_path)
            print(f"[server] Checkpoint saved to {checkpoint_path} as {checkpoint_name}")
            self.last_checkpoint_timestep = self.total_actor_timesteps
        except Exception as e:
            print(f"[server] Error saving checkpoint: {e}")

    # def redistribute_crash_penalty(self, episode, ramp_steps: int = 20, crash_penalty: float = 15.0):
    #     crash_idx = None
    #     for i in reversed(range(len(episode))):
    #         t = episode[i]
    #         info = t.get("info", {})
    #         if t.get("done", False) and info.get("truncated", False):
    #             crash_idx = i
    #             break
    #     if crash_idx is None:
    #         # no crash in this episode -> leave rewards unchanged
    #         return episode

    #     #remove original penalty
    #     episode[crash_idx]["reward"] += crash_penalty

    #     # closer to crash -> larger share
    #     for k in range(ramp_steps):
    #         idx = crash_idx - k
    #         if idx < 0:
    #             break
    #         t = episode[idx]
    #         # weight from 0..1, increasing towards crash
    #         w = (ramp_steps - k) / ramp_steps
    #         shaped_penalty = w * (crash_penalty/2) #divide by 2 so total penalty is not too high
    #         t["reward"] -= shaped_penalty
    #     return episode


    async def _train_loop(self):
        """Periodic training loop: drain new episodes -> add to replay -> train -> broadcast new weights."""
        while True:
            #TODO: DEBUG:::::
            # print(f"Current waypoint vel factor: {Settings.GLOBAL_WAYPOINT_VEL_FACTOR}")

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

            if self.save_model_checkpoints and self.model is not None:
                if self.total_actor_timesteps >= Settings.SIMULATION_LENGTH * 0.85:
                    if self.total_actor_timesteps  - self.last_checkpoint_timestep >= self.checkpoint_frequency:
                        checkpoint_name = f"{self.save_model_name}_ckpt_{self.total_actor_timesteps}"
                        self._save_model_checkpoint(checkpoint_name)
                        print(f"[server] Checkpoint saved at timestep: {self.total_actor_timesteps}")

            
            
            await asyncio.sleep(self.train_every_seconds)
            # Wait for the first observation to initialize model/replay buffer.
            if self.model is None or self.replay_buffer is None:
                continue

            # Drain whatever the actor sent since last round
            episodes = self.episode_buffer.drain_all()

            n_added = self._ingest_episodes_into_replay(episodes)
            if( n_added > 0 ):
                if Settings.LEARNER_SERVER_DEBUG:
                    print(f"[server] Ingested {len(episodes)} episodes / {n_added} transitions into replay "
                    f"(size={self.replay_buffer.size() if self.replay_buffer else 0}).")
                await self._broadcast_training_status()
                
            # debug_time_2 = time.time()
            # =================================================================
            # NEW: Auto-Shutdown Check
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


            if( self.replay_buffer.size() < self.learning_starts ):
                needed = self.learning_starts - self.replay_buffer.size()
                print(f"[server] Not training yet. Need {max(0, needed)} more samples.")
                await asyncio.sleep(self.train_every_seconds)
                continue

            max_udt = getattr(Settings, "SAC_MAX_UTD", None)
            if max_udt is not None:
                max_udt = float(max_udt)
                if max_udt > 0.0 and current_udt > max_udt:
                    print(
                        f"[server] UDT {current_udt:.4f} above max "
                        f"{max_udt:.4f}; waiting."
                    )
                    continue
            # Train if we have enough samples
            if self.model is not None and self.replay_buffer is not None and self.replay_buffer.size() >= self.learning_starts:
                # gradient steps proportional to newly ingested data (UTD ≈ 4)
                grad_steps = self.grad_steps


                print(f"\n[server] Training SAC... steps={grad_steps} | bs={self.model.batch_size} | buffer size={self.replay_buffer.size()} | UDT={current_udt:.4f}")

                if self.pre_fill_with_pp and self._bc_prefill_done and not self._printed_first_sac_fingerprint:
                    cur_fp = self._actor_fingerprint()
                    print(f"[server] Actor fingerprint at first SAC start: {cur_fp}")
                    if self._bc_actor_fingerprint is not None and cur_fp == self._bc_actor_fingerprint:
                        print("[server] ✅ First SAC step starts from BC actor weights.")
                    elif self._bc_actor_fingerprint is not None:
                        print(f"[server] ⚠️ Actor fingerprint mismatch before first SAC step. BC={self._bc_actor_fingerprint} current={cur_fp}")
                    self._printed_first_sac_fingerprint = True

                time_start_training = time.time()


                # Manual training loop for SAC using torch
                actor = self.model.policy.actor
                critic = self.model.policy.critic
                critic_target = self.model.policy.critic_target
                actor_optimizer = self.model.policy.actor.optimizer
                critic_optimizer = self.model.policy.critic.optimizer
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

                # debug_time_4 = time.time()

                                
                training_steps_done = 0
                for step in range(grad_steps):
                    
                    if step % 10 == 0:
                        await asyncio.sleep(0.01)
                        async with self._terminate_lock:
                            # print("CHECKING TERMINATION AT STEP " + str(step))
                            if self._should_terminate:
                                # print("SHOULD TERMINATE FLAG DETECTED DURING TRAINING")
                                print(f"[server] Training interrupted at step {step}/{grad_steps} due to termination request")
                                return
                    
                    then = time.time()
                    try:
                        # Reduce batch size to avoid OOM
                        safe_batch_size = min(self.batch_size, 4096)

                        #NIKITA: testing testing testing
                        # old_alpha = self.replay_buffer.alpha
                        # self.replay_buffer.alpha = 0.0
                        if Settings.SAC_CUSTOM_UNIFORM_CRITIC:
                            old_alpha = self.replay_buffer.alpha
                            self.replay_buffer.alpha = 0.0
                        
                        if Settings.SAC_CRITIC_PURE_TD:
                            old_ratio = self.replay_buffer.state_to_TD_ratio
                            self.replay_buffer.state_to_TD_ratio = 0.0


                        data = self.replay_buffer.sample(safe_batch_size, invert_TD=self.critic_invert_TD)

                        if Settings.SAC_CRITIC_PURE_TD:
                            self.replay_buffer.state_to_TD_ratio = old_ratio

                        if Settings.SAC_CUSTOM_UNIFORM_CRITIC:
                            self.replay_buffer.alpha = old_alpha
                        # self.replay_buffer.alpha = old_alpha

                        obs      = data.observations
                        actions  = data.actions
                        next_obs = data.next_observations
                        rewards  = data.rewards  # shape handling below
                        dones    = data.dones
                        steps_taken = data.steps_taken

                        #TODO: Have the IS sampling weights be returned here
                        is_weights = data.is_weights if hasattr(data, "is_weights") else torch.ones((safe_batch_size, 1), device=obs.device)

                        # print("IS WEIGHTS WORKED?????")
                        # print(is_weights)
                        # print(f"[server] Sampled batch in {(time.time() - then):.5f} seconds.")

                        # Critic update
                        with torch.no_grad():
                            next_actions, next_log_prob = self.model.policy.actor.action_log_prob(next_obs)
                            # print(next_log_prob)
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

                            # print(next_log_prob)
                            """
                            R_t^n = sum_{i=0}^{n-1} gamma^i * r_{t+i} -> rewards handled when saving to replay buffer
                                  + alpha * sum_{i=1}^{n} gamma^i * H(pi(s_{t+i}))      -> here
                                  + gamma^n * Q(s_{t+n}) -> here
                            """
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
                            # target_q = rewards + gamma * (1 - dones) * (target_q - ent_coef * next_log_prob)
                        # print(f"[server] Critic lost time: {(time.time() - then):.5f} seconds.")

                        current_q1, current_q2 = critic(obs, actions)


                        ###Nikita: this is my critic loss!
                        critic_loss = ((is_weights * ((current_q1 - target_q).pow(2))).mean() 
                                       + (is_weights * ((current_q2 - target_q).pow(2))).mean())
                        # critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(current_q2, target_q)
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()

                        #get TD-error
                        if self.replay_buffer.custom_sampling:
                            with torch.no_grad():
                                # -> TD-error = TD_target - current_q
                                current_q1_new, current_q2_new = critic(obs, actions)
                                TD_error_1 = torch.abs(target_q - current_q1_new)
                                TD_error_2 = torch.abs(target_q - current_q2_new)

                                TD_update_priorities = ((TD_error_1 + TD_error_2) / 2.0)

                                # TD_update_priorities = TD_update_priorities * discounts.clamp(min=1e-6)
                                TD_update_priorities = TD_update_priorities.cpu().numpy().flatten()
                                TD_update_priorities += 1e-6
                                

                                if self.replay_buffer.stat_tracker:
                                    self.replay_buffer.stat_tracker.batch_update_TD_errors(self.replay_buffer.current_sampled_inds, TD_update_priorities)

                                # # NIKITA: TESTING INVERTED TD ERROR PRIORITIES
                                # TD_update_priorities = 1.0 / TD_update_priorities
                                # TD_update_priorities = TD_update_priorities / TD_update_priorities.max() #normalize :)

                                # optional clipping
                                if self.replay_buffer.clip_weights:
                                    TD_update_priorities = np.clip(TD_update_priorities, 1e-6, 1.0)

                                self.replay_buffer.update_TD_priorities(TD_update_inds = self.replay_buffer.current_sampled_inds, TD_update_priorities = TD_update_priorities)


                        # print(f"[server] Critic updated in {(time.time() - then):.5f} seconds.")

                        # NIKITA: testing seperate sample for actor and critic
                        data = self.replay_buffer.sample(safe_batch_size, invert_TD=self.actor_invert_TD)

                        obs      = data.observations
                        actions  = data.actions
                        next_obs = data.next_observations
                        rewards  = data.rewards  # shape handling below
                        dones    = data.dones
                        steps_taken = data.steps_taken

                        #TODO: Have the IS sampling weights be returned here
                        is_weights = data.is_weights

                        #NIKITA: get TD error wtice, for actor batch also
                        with torch.no_grad():
                            next_actions, next_log_prob = self.model.policy.actor.action_log_prob(next_obs)
                            # print(next_log_prob)
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

                            # print(next_log_prob)
                            """
                            R_t^n = sum_{i=0}^{n-1} gamma^i * r_{t+i} -> rewards handled when saving to replay buffer
                                  + alpha * sum_{i=1}^{n} gamma^i * H(pi(s_{t+i}))      -> here
                                  + gamma^n * Q(s_{t+n}) -> here
                            """
                            target_q = rewards + discounts * (1 - dones) * (target_q - ent_coef * next_log_prob)

                        if self.replay_buffer.custom_sampling:
                            with torch.no_grad():
                                # -> TD-error = TD_target - current_q
                                current_q1_new, current_q2_new = critic(obs, actions)
                                TD_error_1 = torch.abs(target_q - current_q1_new)
                                TD_error_2 = torch.abs(target_q - current_q2_new)

                                TD_update_priorities = ((TD_error_1 + TD_error_2) / 2.0)

                                # TD_update_priorities = TD_update_priorities * discounts.clamp(min=1e-6)
                                TD_update_priorities = TD_update_priorities.cpu().numpy().flatten()
                                TD_update_priorities += 1e-6
                                

                                if self.replay_buffer.stat_tracker:
                                    self.replay_buffer.stat_tracker.batch_update_TD_errors(self.replay_buffer.current_sampled_inds, TD_update_priorities)

                                # NIKITA: TESTING INVERTED TD ERROR PRIORITIES
                                # TD_update_priorities = 1.0 / TD_update_priorities
                                # TD_update_priorities = TD_update_priorities / TD_update_priorities.max() #normalize :)

                                # optional clipping
                                if self.replay_buffer.clip_weights:
                                    TD_update_priorities = np.clip(TD_update_priorities, 1e-6, 1.0)

                                self.replay_buffer.update_TD_priorities(TD_update_inds = self.replay_buffer.current_sampled_inds, TD_update_priorities = TD_update_priorities)

                        # Actor update
                        new_actions, log_prob = actor.action_log_prob(obs)
                        q1_new, q2_new = critic(obs, new_actions)
                        q_new = torch.min(q1_new, q2_new)

                        ####Nikita: this is my actor loss
                        if self.replay_buffer.use_is_weights_for_actor:
                            actor_loss = (is_weights * (ent_coef * log_prob - q_new)).mean()
                        else:
                            actor_loss = (ent_coef * log_prob - q_new).mean()
                        
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        # print(f"[server] Actor updated in {(time.time() - then):.5f} seconds.")

                        # Entropy coefficient update
                        if log_ent_coef is not None and ent_coef_optimizer is not None:
                            ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
                            ent_coef_optimizer.zero_grad()
                            ent_coef_loss.backward()
                            ent_coef_optimizer.step()

                        # print(f"[server] Entropy coefficient updated in {(time.time() - then):.5f} seconds.")

                        # Soft update of target network
                        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                        # print(f"[server] Step {step+1}/{grad_steps} completed in {(time.time() - then):.5f} seconds.")
                        self.total_weight_updates += 1
                        training_steps_done += 1

                        # Free unused memory
                        # if self.device == "cuda":
                        #     torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[server] CUDA OOM during training step {step}: {e}")
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        continue

                
                
                # Check if we should terminate before saving/broadcasting
                async with self._terminate_lock:
                    if self._should_terminate:
                        print("[server] Termination requested, skipping post-training save/broadcast")
                        return

                if training_steps_done == 0:
                    print("[server] No gradient steps completed (early exit or repeated OOM); skipping log/broadcast.")
                    continue

                ent_coef_loss_val = (
                    float(ent_coef_loss.detach().item())
                    if log_ent_coef is not None and ent_coef_optimizer is not None
                    else 0.0
                )
                log_dict = {
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "ent_coef_loss": ent_coef_loss_val,
                    "ent_coef": ent_coef if isinstance(ent_coef, float) else ent_coef.item(),
                    "total_weight_updates": self.total_weight_updates,
                    "training_duration": time.time() - time_start_training,
                    "total_timesteps": self.total_actor_timesteps,
                    "UDT": self.total_weight_updates / max(1, self.total_actor_timesteps),
                }

                training_time = time.time() - time_start_training
                if Settings.LEARNER_SERVER_DEBUG:
                    print(f"[server] Training completed in {(time.time() - time_start_training):.2f} seconds.")
                save_time = time.time()

                # Nikita: stat tracker save
                if self.replay_buffer.stat_tracker is not None:
                    self.replay_buffer.stat_tracker.update_training_length_list(training_time)
                    now = time.time()
                    if now - self._last_stat_save_ts > 60.0:  # save every 5 minutes
                        self.replay_buffer.stat_tracker.print_stats()
                        # self.replay_buffer.stat_tracker.save_csv(append = False)
                        self._last_stat_save_ts = now

                
                if(len(episodes) > 0):
                    print(log_dict)
                    
                    # Time CSV logging
                    csv_start = time.time()
                    self.trainingLogHelper.log_to_csv(self.model, episodes, log_dict)
                    csv_duration = time.time() - csv_start
                    if self.replay_buffer.stat_tracker is not None:
                        self.replay_buffer.stat_tracker.update_csv_logging_time(csv_duration)

                # Time state dict serialization
                serial_start = time.time()
                new_blob = SacUtilities.state_dict_to_bytes(self.model.policy.actor.state_dict())
                self._weights_blob = new_blob
                await self._broadcast_weights(new_blob)
                current_udt = self.total_weight_updates / max(1, self.total_actor_timesteps)
                training_info_payload = self._build_training_info_payload(
                    current_udt=current_udt,
                    training_steps_done=training_steps_done,
                )
                await self._broadcast_training_info(training_info_payload)
                print("[server] Trained SAC and broadcast updated actor weights.")

                self._weights_blob = new_blob

                # Time weight broadcasting
                broadcast_start = time.time()
                await self._broadcast_weights(new_blob)
                broadcast_duration = time.time() - broadcast_start
                if self.replay_buffer.stat_tracker is not None:
                    self.replay_buffer.stat_tracker.update_broadcast_time(broadcast_duration)

                print("\n[server] Trained SAC and broadcast updated actor weights.")

                # Nikita: debug
                try:
                    self.model.save(self.save_model_path)
                    # print(f"[server] Auto-saved model to {self.save_model_path}")
                except Exception as e:
                    print(f"[server] Error auto-saving model: {e}")
                print(f"[server] Rest of train loop after training completed: {(time.time() - save_time):.2f} seconds.")
                # print(f"[server] 1 to 2: {(debug_time_2 - debug_time_1):.2f} seconds.")
                # print(f"[server] 1 to 3: {(debug_time_3 - debug_time_1):.2f} seconds.")
                # print(f"[server] 1 to 4: {(debug_time_4 - debug_time_1):.2f} seconds.")
            else:
                needed = self.learning_starts - (self.replay_buffer.size() if self.replay_buffer else 0)
                print(f"[server] Not training yet. Need {max(0, needed)} more samples.")


    # ---------- networking ----------
    def _can_start_training(self) -> bool:
        if self.replay_buffer is None:
            return False
        if self.pre_fill_with_pp and (not self._bc_prefill_done or self._bc_in_progress):
            return False
        return self.replay_buffer.size() >= self.learning_starts

    def _training_status_frame(self) -> dict:
        return {
            "type": "training_status",
            "data": {
                "training_ready": self._can_start_training(),
                "replay_size": self.replay_buffer.size() if self.replay_buffer is not None else 0,
                "learning_starts": self.learning_starts,
                "prefill_with_pp": self.pre_fill_with_pp,
                "bc_in_progress": self._bc_in_progress,
                "bc_prefill_done": self._bc_prefill_done,
            },
        }

    async def _broadcast_training_status(self, force: bool = False):
        training_ready = self._can_start_training()
        if not force and self._last_broadcast_training_ready == training_ready:
            return

        frame = self._training_status_frame()
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

        self._last_broadcast_training_ready = training_ready

    async def _broadcast_weights(self, blob: bytes):
        """Send weights to all currently connected clients."""
        frame = {
            "type": "weights",
            "data": {
                "blob": blob,
                "format": "torch_state_dict",
                "algo": "SAC",
                "module": "actor",
                "fingerprint": self._actor_fingerprint(),
            },
        }
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

    async def _broadcast_training_info(self, payload: dict):
        """Send post-training telemetry as an extensible JSON payload."""
        frame = {"type": "training_info", "data": payload}
        self._latest_training_info_payload = payload
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

    def _build_training_info_payload(self, current_udt: float, training_steps_done: int) -> dict:
        """Build an extensible training telemetry payload."""
        target_udt = getattr(Settings, "SAC_TARGET_UDT", None)
        udt_control = None
        if target_udt is not None:
            target_udt = float(target_udt)
            deadband_ratio = float(getattr(Settings, "SAC_UDT_DEADBAND_RATIO", 0.1))
            deadband_ratio = min(max(deadband_ratio, 0.0), 0.95)
            resume_udt = target_udt * (1.0 - deadband_ratio)
            udt_control = {
                "target_udt": target_udt,
                "resume_udt": resume_udt,
                "training_paused": bool(self._training_paused_for_udt),
                "deadband_ratio": deadband_ratio,
                "freq_adjust_step_ratio": float(getattr(Settings, "SAC_UDT_FREQ_ADJUST_STEP_RATIO", 0.05)),
                "min_sim_frequency_hz": float(getattr(Settings, "SAC_MIN_SIM_FREQUENCY", 20.0)),
            }
        payload = {
            "schema_version": 1,
            "event": "post_training_update",
            "timestamp": time.time(),
            "metrics": {
                "current_udt": float(current_udt),
            },
            "counters": {
                "total_weight_updates": int(self.total_weight_updates),
                "total_actor_timesteps": int(self.total_actor_timesteps),
                "training_steps_done": int(training_steps_done),
                "replay_buffer_size": int(self.replay_buffer.size() if self.replay_buffer is not None else 0),
            },
        }
        if udt_control is not None:
            payload["udt_control"] = udt_control
        return payload

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

        # Tell clients where to mirror model artifacts from.
        try:
            writer.write(
                pack_frame(
                    {
                        "type": "model_sync",
                        "data": {
                            "model_name": self.save_model_name,
                            "source_model_dir": self.model_dir,
                        },
                    }
                )
            )
            await writer.drain()
        except Exception as e:
            print(f"[server] Failed to send model sync info to {addr}: {e}")

        # ALWAYS send current weights if we have them (even when init_from_scratch=True)
        if self._weights_blob is not None:
            try:
                await self._broadcast_weights(self._weights_blob)
                print(f"[server] Weights sent to {addr} (bytes={len(self._weights_blob)})")
            except Exception as e:
                print(f"[server] Failed to send initial weights to {addr}: {e}")
        # Send latest training telemetry (if available) to reconnecting actors.
        if self._latest_training_info_payload is not None:
            try:
                writer.write(pack_frame({"type": "training_info", "data": self._latest_training_info_payload}))
                await writer.drain()
            except Exception as e:
                print(f"[server] Failed to send training info to {addr}: {e}")
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

                    # Lazily init model/replay once we see the first obs vector.
                    if self.model is None:
                        try:
                            first_obs = obs_list[0]
                            obs_dim = int(np.asarray(first_obs, dtype=np.float32).reshape(-1).shape[0])
                            self._initialize_model(obs_dim=obs_dim)
                            # Broadcast initial weights right after init so the actor can leave warmup.
                            if self._weights_blob is not None:
                                await self._broadcast_weights(self._weights_blob)
                        except Exception as e:
                            print(f"[server] Failed to init model from first obs: {e}")

                    if Settings.RL_CRASH_REWQARD_RAMPING:
                        episode = self._apply_crash_ramp(episode, ramp_steps=20, max_ramp_value=15.0)
                        episode = self._apply_crash_ramp(episode, ramp_steps=20, max_ramp_value=15.0)

                    # initialize shapes if this is the very first episode
                    if self.model is None:
                        self._initialize_model()

                    self.last_episode_time = time.time()

                    if self.pre_fill_with_pp and self._bc_in_progress:
                        try:
                            writer.write(pack_frame({"type": "episode_ack", "data": {"n": len(episode), "ignored": True}}))
                            await writer.drain()
                        except Exception:
                            pass
                        continue

                    self.episode_buffer.add_episode(episode)
                    # print(f"[server] Stored episode: {len(episode)} transitions "
                    #     f"(total episodes pending train: {len(self.episode_buffer.episodes)})")
                    if self.pre_fill_with_pp and not self._bc_prefill_done:
                        self.total_prefill_timesteps += len(episode)
                    else:
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
                        self._last_broadcast_training_ready = None
                        print(f"[server] Cleared replay buffer (had {replay_size_before} transitions)")
                    
                    print(f"[server] Cleared {episodes_cleared} episodes from buffer (requested by actor {actor_id})")
                    
                    # Send acknowledgment
                    try:
                        writer.write(pack_frame({"type": "clear_buffer_ack", "data": {"episodes_cleared": episodes_cleared}}))
                        await writer.drain()
                    except Exception:
                        pass

                    await self._broadcast_training_status(force=True)
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
        # If loading a pretrained model, initialize immediately from that model's obs space.
        # This prevents bootstrap transitions with stale obs builders from forcing a wrong obs_dim.
        if self.load_model_name is not None and self.model is None:
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                probe_model = SAC.load(self.load_model_path, device=self.device)
                obs_dim = int(probe_model.observation_space.shape[0])
                del probe_model
                self._initialize_model(obs_dim=obs_dim)
            except Exception as e:
                print(
                    "[server] Warning: could not pre-initialize from load model "
                    f"'{self.load_model_name}'. Falling back to lazy init from actor obs. Error: {e}"
                )

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
            
            # Save model one last time
            print("[server] Saving model before exit...")
            self._save_model()
            
            print("[server] Shutdown complete")

