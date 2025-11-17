#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import os
import subprocess
from typing import List, Optional

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

from utilities.Settings import Settings

class CustomReplayBuffer(ReplayBuffer):
    """Class to extend SB3 replaybuffer, to enable custom weighted sampling, and other additional helping functions"""
    def __init__(self, *args, sample_weighting=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weighting = sample_weighting
        self.w_d = Settings.SAC_WP_OFFSET_WEIGHT
        self.w_e = Settings.SAC_WP_HEADING_ERROR_WEIGHT
        self.reward_weight = Settings.SAC_REWARD_WEIGHT
        self.alpha = Settings.SAC_PRIORITY_FACTOR
        self.custom_sampling = Settings.USE_CUSTOM_SAC_SAMPLING
        # Initialize with small non-zero values to avoid zero-sum edge cases (DO I ACTUALLY NEED THIS?)
        self._weights = np.ones(self.buffer_size, dtype=np.float64) * 1e-6
        
        if self.custom_sampling:
            print("Using custom SAC replay buffer sampling with weights:")
            print(f"Offset weight: {self.w_d}, Heading error weight: {self.w_e}, Reward weight: {self.reward_weight}, Priority factor: {self.alpha}")

    def add(self, *args, **kwargs):
        """Override SB3 add so that the weight can be computed once per transition, and then stored in the buffer"""
    
        # --> actual weighting computation is only within this function
        super().add(*args, **kwargs)

        try:
            idx = (self.pos - 1) % self.buffer_size
        except Exception:
            return

        # Extract only the observation that was just added (at index idx)
        obs = self.observations[idx, 0, :]

        d = obs[-2]
        e = obs[-1]

        w = self.w_d * abs(d) + self.w_e * abs(e)
        w = np.clip(w, 1e-6, 1e3)


        self._weights[idx] = w

    def sample(self, safe_batch_size: int, env=None):
        """custom sample function, if none then use default SB3"""
        #weights already calculated in add(), here we just normalize and sample

        # self.weight_func = weight_func_for_test

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


        # Use stored per-transition weights for sampling instead of recomputing from obs
        w_vec = self._weights[possible_inds].astype(np.float64)

        # Clamp extremely small/large values
        # w_vec = np.clip(w_vec, 1e-6, 1e3)

        # ==== Final normalization ====
        total_w = np.sum(w_vec)
        if total_w <= 0 or not np.isfinite(total_w):
            priority_p = np.ones_like(w_vec) / len(w_vec)
        else:
            priority_p = w_vec / total_w

        length = len(possible_inds)
        uniform_p = np.ones(length) / length
        p = self.alpha * priority_p + (1.0 - self.alpha) * uniform_p

        # Safety normalization
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p_tot = p.sum()
        if p_tot <= 0 or not np.isfinite(p_tot):
            p = np.ones_like(p) / len(p)
        else:
            p = p / p_tot

        batch_inds = np.random.choice(possible_inds, size=safe_batch_size, p=p)
        return self._get_samples(batch_inds, env=env)

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
        while True:
            # Check termination flag at the start of each iteration
            async with self._terminate_lock:
                if self._should_terminate:
                    print("[server] Training loop detected termination flag, exiting...")
                    return
            
            await asyncio.sleep(self.train_every_seconds)
            # Drain whatever the actor sent since last round
            episodes = self.episode_buffer.drain_all()

            n_added = self._ingest_episodes_into_replay(episodes)
            if( n_added > 0 ):
                print(f"[server] Ingested {len(episodes)} episodes / {n_added} transitions into replay "
                    f"(size={self.replay_buffer.size() if self.replay_buffer else 0}).")

            # Calculate trainign steps per sample
            current_udt = self.total_weight_updates / max(1, self.total_actor_timesteps)
            if current_udt > 10.:
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
                # gradient steps proportional to newly ingested data (UTD ≈ 4)
                grad_steps = self.grad_steps


                print(f"[server] Training SAC... steps={grad_steps} | bs={self.model.batch_size} | buffer size={self.replay_buffer.size()} | UDT={current_udt:.4f}")
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
                tau = self.model.tau
                ent_coef = self.model.ent_coef
                target_entropy = self.model.target_entropy
                log_ent_coef = getattr(self.model, "log_ent_coef", None)
                ent_coef_optimizer = getattr(self.model, "ent_coef_optimizer", None)
                                
                for step in range(grad_steps):
                    # Check termination flag periodically during training (every 10 steps)
                    if step % 10 == 0:
                        async with self._terminate_lock:
                            if self._should_terminate:
                                print(f"[server] Training interrupted at step {step}/{grad_steps} due to termination request")
                                break
                    
                    then = time.time()
                    try:
                        # Reduce batch size to avoid OOM
                        safe_batch_size = min(batch_size, 4096)
                        data = replay_buffer.sample(safe_batch_size)

                        '''make a custom sample function here:
                        #TODO: bind this to settings.py'''
                        # data_testing = replay_buffer.sample(safe_batch_size, weight_func_for_test = 1)

                        """TODO: compare data to data_testing"""

                        obs      = data.observations
                        actions  = data.actions
                        next_obs = data.next_observations
                        rewards  = data.rewards  # shape handling below
                        dones    = data.dones
                   

                        # print(f"[server] Sampled batch in {(time.time() - then):.5f} seconds.")

                        # Critic update
                        with torch.no_grad():
                            next_actions, next_log_prob = self.model.policy.actor.action_log_prob(next_obs)
                            target_q1, target_q2 = critic_target(next_obs, next_actions)
                            target_q = torch.min(target_q1, target_q2)
                            if log_ent_coef is not None:
                                ent_coef = torch.exp(log_ent_coef.detach())
                            # Ensure next_log_prob shape is [batch_size, 1]
                            if next_log_prob.dim() == 2 and next_log_prob.shape[1] != 1:
                                next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
                            else:
                                next_log_prob = next_log_prob.view(-1, 1)
                            target_q = rewards + gamma * (1 - dones) * (target_q - ent_coef * next_log_prob)
                        # print(f"[server] Critic lost time: {(time.time() - then):.5f} seconds.")

                        current_q1, current_q2 = critic(obs, actions)
                        critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(current_q2, target_q)
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()

                        # print(f"[server] Critic updated in {(time.time() - then):.5f} seconds.")

                        # Actor update
                        new_actions, log_prob = actor.action_log_prob(obs)
                        q1_new, q2_new = critic(obs, new_actions)
                        q_new = torch.min(q1_new, q2_new)
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
                
                # You can now interfere with actor, critic, optimizers, etc. in this loop
                log_dict = {
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "ent_coef_loss": ent_coef_loss.item(),
                    "ent_coef": ent_coef if isinstance(ent_coef, float) else ent_coef.item(),
                    "total_weight_updates": self.total_weight_updates,
                    "training_duration": time.time() - time_start_training,
                    "total_timesteps": self.total_actor_timesteps,
                    "UDT": self.total_weight_updates / max(1, self.total_actor_timesteps),
                }
                print(f"[server] Training completed in {(time.time() - time_start_training):.2f} seconds.")
                if(len(episodes) > 0):
                    self.trainingLogHelper.log_to_csv(self.model, episodes, log_dict)

                new_blob = SacUtilities.state_dict_to_bytes(self.model.policy.actor.state_dict())
                self._weights_blob = new_blob
                await self._broadcast_weights(new_blob)
                print("[server] Trained SAC and broadcast updated actor weights.")

                try:
                    target = os.path.join(self.model_dir, str(self.save_model_name))
                    self.model.save(target)
                    # print(f"[server] Auto-saved model to {target}")
                except Exception as e:
                    print(f"[server] Error auto-saving model: {e}")
            else:
                needed = self.learning_starts - (self.replay_buffer.size() if self.replay_buffer else 0)
                print(f"[server] Not training yet. Need {max(0, needed)} more samples.")


    # ---------- networking ----------
    async def _broadcast_weights(self, blob: bytes):
        """Send weights to all currently connected clients."""
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
            
            # Save model one last time
            print("[server] Saving model before exit...")
            self._save_model()
            
            print("[server] Shutdown complete")

