#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import ast
import os
import subprocess
import shutil
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

        self.save_model_checkpoints = Settings.SAC_SAVE_MODEL_CHECKPOINTS
        self.checkpoint_frequency = Settings.SAC_CHECKPOINT_FREQUENCY
        self.last_checkpoint_timestep = 0


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
            self.load_model_path = load_model_path_root if os.path.exists(load_model_path_root + ".zip") else load_model_path_server
            self._sync_save_model_from_load_model()

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

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.replay_capacity,
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            device=self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        self.model.replay_buffer = self.replay_buffer
        self._load_replay_buffer()
        # Restore historical counters from metrics, then ensure loaded replay
        # samples are reflected in the UDT denominator.
        self._restore_training_counters_from_metrics()
        self.total_actor_timesteps = max(
            self.total_actor_timesteps,
            int(self.replay_buffer.size()),
        )

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
                target = os.path.join(self.model_dir, self.save_model_name)
                self.model.save(target)
                print(f"[server] Model saved to {target}")
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
                self.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos={},  # could pass TimeLimit info here if you track it
                )
                n_added += 1
        if dropped_for_dim_mismatch > 0:
            print(
                f"[server] Dropped {dropped_for_dim_mismatch} transition(s) due to "
                f"observation dim mismatch (expected {expected_obs_dim})."
            )
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


    async def _train_loop(self):
        """Periodic training loop: drain new episodes -> add to replay -> train -> broadcast new weights."""
        while True:
            # Check termination flag at the start of each iteration
            async with self._terminate_lock:
                if self._should_terminate:
                    print("[server] Training loop detected termination flag, exiting...")
                    return

            if self.save_model_checkpoints and self.model is not None:
                if self.total_actor_timesteps  - self.last_checkpoint_timestep >= self.checkpoint_frequency:
                    checkpoint_name = f"{self.save_model_name}_ckpt_{self.total_actor_timesteps}"
                    self._save_model_checkpoint(checkpoint_name)

            
            
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

            # Calculate trainign steps per sample
            current_udt = self.total_weight_updates / max(1, self.total_actor_timesteps)


            if( self.replay_buffer.size() < self.learning_starts ):
                needed = self.learning_starts - self.replay_buffer.size()
                print(f"[server] Not training yet. Need {max(0, needed)} more samples.")
                await asyncio.sleep(self.train_every_seconds)
                continue

            target_udt = getattr(Settings, "SAC_TARGET_UDT", None)
            if target_udt is not None:
                target_udt = float(target_udt)
                if target_udt > 0.0:
                    deadband_ratio = float(getattr(Settings, "SAC_UDT_DEADBAND_RATIO", 0.1))
                    deadband_ratio = min(max(deadband_ratio, 0.0), 0.95)
                    resume_udt = target_udt * (1.0 - deadband_ratio)

                    if self._training_paused_for_udt:
                        if current_udt <= resume_udt:
                            self._training_paused_for_udt = False
                            print(
                                f"[server] UDT {current_udt:.4f} <= resume threshold "
                                f"{resume_udt:.4f}; resuming training."
                            )
                        else:
                            print(
                                f"[server] UDT {current_udt:.4f} above resume threshold "
                                f"{resume_udt:.4f}; waiting."
                            )
                            continue
                    elif current_udt >= target_udt:
                        self._training_paused_for_udt = True
                        print(
                            f"[server] UDT {current_udt:.4f} reached target "
                            f"{target_udt:.4f}; pausing until below {resume_udt:.4f}."
                        )
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
                replay_buffer = self.model.replay_buffer
                gamma = self.model.gamma
                tau = self.model.tau
                ent_coef = self.model.ent_coef
                target_entropy = self.model.target_entropy
                log_ent_coef = getattr(self.model, "log_ent_coef", None)
                ent_coef_optimizer = getattr(self.model, "ent_coef_optimizer", None)

                training_steps_done = 0
                for step in range(grad_steps):
                    # Check termination flag periodically during training (every 10 steps)
                    if step % 10 == 0:
                        async with self._terminate_lock:
                            if self._should_terminate:
                                print(f"[server] Training interrupted at step {step}/{grad_steps} due to termination request")
                                break
                    
                    then = time.time()
                    try:
                        # Reduce batch size to avoid OOM (read from model so a bad merge can't leave `batch_size` undefined)
                        safe_batch_size = min(int(self.model.batch_size), 4096)
                        data = replay_buffer.sample(safe_batch_size)
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
                if Settings.LEARNER_SERVER_DEBUG:
                    print(f"[server] Training completed in {(time.time() - time_start_training):.2f} seconds.")
                if(len(episodes) > 0):
                    self.trainingLogHelper.log_to_csv(self.model, episodes, log_dict)

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

