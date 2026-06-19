#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import ast
import os
import subprocess
import shutil
import threading
from typing import Any, Callable, Dict, List, Optional

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
from sac_utilities import _SpacesOnlyEnv, SacUtilities, EpisodeReplayBuffer, TrainingLogHelper, ObsRewardTracker
from metrics_http import MetricsHttpServer

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
        batch_size: Optional[int] = None,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        train_frequency: int = 1,
        save_replay_buffer: bool = False,
        load_replay_buffer: bool = False,
        status_line_callback: Optional[Callable[[str], None]] = None,
    ):
        self.host = host
        self.port = port
        self.load_model_name = load_model_name
        self.save_model_name = save_model_name
        self.device = device
        self.train_every_seconds = train_every_seconds
        self.replay_capacity = replay_capacity
        self.batch_size = int(
            batch_size if batch_size is not None else getattr(Settings, "SAC_BATCH_SIZE", 256)
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.train_frequency = train_frequency
        self.save_replay_buffer_enabled = bool(save_replay_buffer)
        self.load_replay_buffer_enabled = bool(load_replay_buffer)
        self._latest_training_info_payload: Optional[dict] = None
        self.max_utd = 4.0  # max updates per data point

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
        # Reassemble streamed batches into full episodes: (actor_id, episode_id) -> transitions.
        self._episode_accum: Dict[tuple, List[dict]] = {}
        self._max_episode_id_per_actor: Dict[int, int] = {}
        # TCP stream batch sizes per (actor_id, episode_id); finalized when episode completes.
        self._stream_batches_by_episode: Dict[tuple, List[int]] = {}
        self._finalized_stream_batches: Dict[tuple, List[int]] = {}
        self._weights_blob: Optional[bytes] = None
        lt = getattr(Settings, "SAC_TERMINATE_BELOW_LAPTIME", None)
        self._terminate_below_lap_s: Optional[float] = float(lt) if lt is not None else None
        self._lap_terminate_triggered = False
        self._best_lap_time_seen: Optional[float] = None
        if self._terminate_below_lap_s is not None:
            print(
                f"[server] Lap-time stop enabled: terminate when >=2 laps are faster than "
                f"{self._terminate_below_lap_s:.3f}s (Settings.SAC_TERMINATE_BELOW_LAPTIME)"
            )

        # RL bits (lazy init fallback; but we’ll eager-init in run())
        self.model: Optional[SAC] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.vecnorm: Optional[VecNormalize] = None
        self.total_actor_timesteps = 0
        self.total_weight_updates = 0
        self._training_paused_for_udt = False
        self._train_event = asyncio.Event()
        self._pending_samples = 0
        self._sample_lock = asyncio.Lock()
        self._last_sec_per_grad_step: Optional[float] = None
        # Replay + SAC updates can run in a worker thread; TCP ingest stays on the event loop.
        self._replay_lock = threading.Lock()
        self._train_job_lock = asyncio.Lock()
        self._model_autosave_interval_s = float(getattr(Settings, "SAC_MODEL_AUTOSAVE_INTERVAL_S", 60.0))
        self._last_model_autosave_time = 0.0
        self._last_train_log_time = 0.0
        self._status_line_callback = status_line_callback
        self._checkpoint_thread: Optional[threading.Thread] = None

        self.last_episode_time = None
        self.episode_timeout = 100.0
        self.episode_inactivity_timeout = 60.0
        self._has_received_episode = False

        #used for n-step buffer, for standard buffer set = 1
        self.n_step = getattr(Settings, "SAC_N_STEP", 1)
        # self.n_step_discount_factor = self.discount_factor ** self.n_step
        self.custom_sampling = Settings.USE_CUSTOM_SAC_SAMPLING

        self.save_model_checkpoints = bool(Settings.SAC_SAVE_MODEL_CHECKPOINTS)
        self.checkpoint_frequency = max(1, int(Settings.SAC_CHECKPOINT_FREQUENCY))
        self.last_checkpoint_timestep = 0
        print(
            f"[server] SAC training batch_size={self.batch_size} "
            f"(not SAC_STREAM_BATCH_SIZE)"
        )
        if self.save_model_checkpoints:
            print(
                f"[server] Model checkpoints enabled: every "
                f"{self.checkpoint_frequency} actor timesteps "
                f"(Settings.SAC_CHECKPOINT_FREQUENCY)"
            )


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
        self.obs_tracker = ObsRewardTracker(
            model_dir=self.model_dir,
            enabled=bool(getattr(Settings, "SAC_OBS_TRACKING_ENABLED", True)),
            flush_every=int(getattr(Settings, "SAC_OBS_TRACKING_FLUSH_EVERY", 10_000)),
            hist_bins=int(getattr(Settings, "SAC_OBS_HIST_BINS", 40)),
            hist_sample_cap=int(getattr(Settings, "SAC_OBS_HIST_SAMPLE_CAP", 20_000)),
            hist_max_dims=int(getattr(Settings, "SAC_OBS_HIST_MAX_DIMS", 256)),
        )

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
                exit(1)

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
        if self.load_replay_buffer_enabled and self.load_model_name is not None:
            self._load_replay_buffer()
        elif self.load_replay_buffer_enabled:
            print(
                "[server] Replay-buffer loading skipped: no load_model_name set "
                "(starting from scratch)."
            )
        else:
            print("[server] Replay-buffer loading disabled; starting with empty replay buffer.")
        # Restore historical counters from metrics, then ensure loaded replay
        # samples are reflected in the UDT denominator.
        self._restore_training_counters_from_metrics()
        if self.load_model_name is None:
            self.total_actor_timesteps = max(
                self.total_actor_timesteps,
                int(self.replay_buffer.size()),
            )

        # Save model info
        info = {
            "grad_steps": self.grad_steps,
            "batch_size": self.batch_size,
            "sac_checkpoint_frequency": self.checkpoint_frequency,
            "sac_terminate_below_laptime": self._terminate_below_lap_s,
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

    def _udt_denominator_timesteps(self) -> int:
        """
        Timesteps denominator used for UDT control/logging.
        When finetuning with a loaded replay buffer while counters are reset,
        include replay size to avoid artificial UDT spikes at startup.
        """
        denom = int(self.total_actor_timesteps)
        if self.load_replay_buffer_enabled and self.load_model_name is not None and self.replay_buffer is not None:
            denom = max(denom, int(self.replay_buffer.size()))
        return max(1, denom)

    def _current_udt(self) -> float:
        return float(self.total_weight_updates) / float(self._udt_denominator_timesteps())

    def _compute_grad_steps(self, new_samples: int) -> int:
        """Gradient steps for this round: UTD-targeted, capped, only when samples arrived."""
        if new_samples <= 0 or self.model is None or self.replay_buffer is None:
            return 0
        if self.replay_buffer.size() < self.learning_starts:
            return 0

        current_udt = self._current_udt()
        max_udt = getattr(Settings, "SAC_MAX_UTD", None)
        if max_udt is not None and float(max_udt) > 0.0 and current_udt >= float(max_udt):
            self._training_paused_for_udt = True
            return 0
        self._training_paused_for_udt = False

        target_utd = getattr(Settings, "SAC_TARGET_UTD", None)
        if target_utd is None:
            return int(self.grad_steps)

        target_utd = float(target_utd)
        desired_updates = target_utd * float(self._udt_denominator_timesteps())
        utd_debt = desired_updates - float(self.total_weight_updates)
        if utd_debt < 1.0:
            return 0

        burst_cap = int(self.grad_steps)
        if current_udt < target_utd * 0.98 and new_samples > 0:
            pace = max(1, int(target_utd * new_samples))
            mult = int(getattr(Settings, "SAC_GRAD_BURST_MULTIPLIER", 4))
            burst_cap = max(burst_cap, pace, self.grad_steps * mult)
        burst_cap = min(burst_cap, int(getattr(Settings, "SAC_MAX_GRAD_BURST", 2048)))
        return int(min(burst_cap, utd_debt))

    async def _take_pending_samples(self) -> int:
        async with self._sample_lock:
            n = self._pending_samples
            self._pending_samples = 0
        return n

    def _maybe_log_train_round(
        self,
        *,
        grad_steps: int,
        training_steps_done: int,
        new_samples: int,
        current_udt: float,
        train_duration: float,
        post_duration: float,
        round_wall_s: float,
    ) -> None:
        interval = float(getattr(Settings, "LEARNER_SERVER_TRAIN_LOG_INTERVAL_S", 1.0))
        now = time.time()
        if self._status_line_callback is None and not Settings.LEARNER_SERVER_DEBUG and interval > 0.0:
            if (now - self._last_train_log_time) < interval:
                return
        self._last_train_log_time = now
        buf = int(self.replay_buffer.size() if self.replay_buffer is not None else 0)
        msg = (
            f"grad={training_steps_done}/{grad_steps} "
            f"new={new_samples} UDT={current_udt:.3f} buf={buf} "
            f"train={train_duration:.2f}s"
        )
        if self._status_line_callback is not None:
            self._status_line_callback(msg)
        else:
            print(f"[server] Train: {msg}")

    async def _notify_new_samples(self, n: int) -> None:
        if n <= 0:
            return
        async with self._sample_lock:
            self._pending_samples += n
        self._train_event.set()

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
                self._sync_model_timestep_counter()
                target = os.path.join(self.model_dir, self.save_model_name)
                self.model.save(target)
                print(f"[server] Model saved to {target}")
            except Exception as e:
                print(f"[server] Error saving model: {e}")
        self.save_replay_buffer()
        self.obs_tracker.flush(render_png=False)
        self.trainingLogHelper.maybe_plot_training_metrics_final()

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
        """Initialize training counters for this run (never resume from metrics CSV)."""
        self.total_weight_updates = 0
        self.total_actor_timesteps = 0
        if self.load_model_name is not None:
            print(
                "[server] load_model_name is set; starting with reset counters: "
                "total_weight_updates=0, total_actor_timesteps=0"
            )
        else:
            print(
                "[server] No load_model_name set; starting with fresh counters "
                "(not restoring from learning_metrics.csv)."
            )

    def _maybe_schedule_actor_timestep_checkpoints(self) -> None:
        """Save checkpoints at Settings.SAC_CHECKPOINT_FREQUENCY actor timesteps."""
        if not self.save_model_checkpoints or self.model is None:
            return
        if self.checkpoint_frequency <= 0:
            return
        while self.total_actor_timesteps - self.last_checkpoint_timestep >= self.checkpoint_frequency:
            self.last_checkpoint_timestep += self.checkpoint_frequency
            scheduled = self._schedule_checkpoint_async(
                self._save_checkpoint_locked, self.last_checkpoint_timestep
            )
            if not scheduled:
                self.last_checkpoint_timestep -= self.checkpoint_frequency
                break
    
    def _save_checkpoint(self, timesteps: int):
        """Save a checkpoint with a unique name including timestep count."""
        if self.model is not None:
            try:
                checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_name = f"{self.save_model_name}_checkpoint_{timesteps}"
                target = os.path.join(checkpoint_dir, checkpoint_name)
                self._sync_model_timestep_counter()
                self.model.save(target)
                if getattr(Settings, "SAC_CHECKPOINT_SAVE_REPLAY", False):
                    self.save_replay_buffer()
                self.obs_tracker.flush(render_png=True)
                print(f"[server] Checkpoint saved to {target} (timesteps={timesteps})")
            except Exception as e:
                print(f"[server] Error saving checkpoint: {e}")

    def _schedule_checkpoint_async(self, save_fn, *args) -> bool:
        """Run checkpoint I/O on a dedicated thread (does not block the asyncio train loop)."""
        if self._checkpoint_thread is not None and self._checkpoint_thread.is_alive():
            return False

        def _run():
            try:
                save_fn(*args)
            except Exception as e:
                print(f"[server] Background checkpoint failed: {e}")

        self._checkpoint_thread = threading.Thread(
            target=_run, name="checkpoint-save", daemon=True
        )
        self._checkpoint_thread.start()
        print("[server] Checkpoint save started (background thread).")
        return True

    async def _await_pending_checkpoint(self) -> None:
        thread = self._checkpoint_thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join)

    # ---------- ingestion + training ----------
    # No normalization for now: obs arrive normalized already
    def _normalize_obs(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def _dummy_vec_from_spaces(self, obs_space: spaces.Box, act_space: spaces.Box):
        return DummyVecEnv([lambda: _SpacesOnlyEnv(obs_space, act_space)])

    def _ingest_transitions(self, transitions: List[dict]) -> int:
        if self.model is None or self.replay_buffer is None:
            return 0
        with self._replay_lock:
            n_added, flush_obs_tracker = self._ingest_transitions_locked(transitions)
        if flush_obs_tracker:
            self.obs_tracker.flush(render_png=False)
        return n_added

    def _ingest_transitions_locked(self, transitions: List[dict]) -> tuple[int, bool]:
        expected_obs_dim = self._expected_obs_dim()
        dropped_for_dim_mismatch = 0
        n_added = 0
        flush_obs_tracker = False
        for t in transitions:
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
            obs = self._normalize_obs(obs)
            next_obs = self._normalize_obs(next_obs)
            self.obs_tracker.track(obs, reward)
            self.replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos={},
            )
            n_added += 1
            flush_obs_tracker = flush_obs_tracker or self.obs_tracker.should_flush()
        if dropped_for_dim_mismatch > 0:
            print(
                f"[server] Dropped {dropped_for_dim_mismatch} transition(s) due to "
                f"observation dim mismatch (expected {expected_obs_dim})."
            )
        return n_added, flush_obs_tracker

    def _record_stream_batch(self, actor_id: int, episode_id: int, batch_len: int) -> None:
        if batch_len <= 0:
            return
        key = (int(actor_id), int(episode_id))
        self._stream_batches_by_episode.setdefault(key, []).append(int(batch_len))

    def _finalize_stream_batches_for_episode(self, actor_id: int, episode_id: int) -> None:
        """Freeze TCP batch-size list when an episode completes (prevents late stray batches)."""
        key = (int(actor_id), int(episode_id))
        if key in self._finalized_stream_batches:
            return
        live = self._stream_batches_by_episode.pop(key, [])
        self._finalized_stream_batches[key] = live

    def _stream_batch_sizes_for_episodes(self, episodes: List[List[dict]]) -> List[int]:
        """All stream TCP batch sizes for episodes completed in this log row."""
        sizes: List[int] = []
        for ep in episodes:
            if not ep:
                continue
            key = (int(ep[0].get("actor_id", 0)), int(ep[0].get("episode_id", 0)))
            sizes.extend(self._finalized_stream_batches.pop(key, []))
        return sizes

    def _accumulate_episode_batch(
        self,
        actor_id: int,
        episode_id: int,
        batch: List[dict],
        episode_end: bool = False,
    ) -> List[List[dict]]:
        """
        Append batch to (actor_id, episode_id). On done/episode_end, return the
        full episode (all batches with this id). When episode_id advances, flush
        any still-open older episodes.
        """
        completed: List[List[dict]] = []
        prev_max = self._max_episode_id_per_actor.get(actor_id, -1)
        if episode_id > prev_max:
            stale_keys = sorted(
                (k for k in self._episode_accum if k[0] == actor_id and k[1] < episode_id),
                key=lambda k: k[1],
            )
            for key in stale_keys:
                ep = self._episode_accum.pop(key, None)
                if ep:
                    completed.append(ep)
            self._max_episode_id_per_actor[actor_id] = episode_id

        key = (actor_id, episode_id)
        buf = self._episode_accum.setdefault(key, [])
        buf.extend(batch)

        if episode_end or any(t.get("done") for t in buf):
            ep = self._episode_accum.pop(key, None)
            if ep:
                completed.append(ep)
        return completed

    def _ingest_episodes_into_replay(self, episodes: List[List[dict]]) -> int:
        n_added = 0
        for ep in episodes:
            n_added += self._ingest_transitions(ep)
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

    def _sync_model_timestep_counter(self) -> None:
        """Keep SB3 model.num_timesteps aligned with server actor timesteps for saves/plots."""
        if self.model is not None:
            self.model.num_timesteps = int(self.total_actor_timesteps)

    def _ent_coef_scalar(self, ent_coef, log_ent_coef) -> float:
        """SB3 may keep ent_coef as the string 'auto' until the first successful grad step."""
        if log_ent_coef is not None:
            return float(torch.exp(log_ent_coef.detach()).item())
        if isinstance(ent_coef, (int, float)):
            return float(ent_coef)
        if hasattr(ent_coef, "item"):
            return float(ent_coef.item())
        return 0.0

    def _run_gradient_training(self, grad_steps: int) -> Dict[str, Any]:
        """SAC gradient updates (runs in worker thread; do not call from the event loop)."""
        if self.model is None or self.replay_buffer is None:
            return {"training_steps_done": 0}

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
        actor_loss = critic_loss = ent_coef_loss = 0.0

        with self._replay_lock:
            for step in range(grad_steps):
                if step % 10 == 0 and self._should_terminate:
                    break

                try:
                    data = replay_buffer.sample(int(self.model.batch_size))
                    obs = data.observations
                    actions = data.actions
                    next_obs = data.next_observations
                    rewards = data.rewards
                    dones = data.dones

                    with torch.no_grad():
                        next_actions, next_log_prob = self.model.policy.actor.action_log_prob(next_obs)
                        target_q1, target_q2 = critic_target(next_obs, next_actions)
                        target_q = torch.min(target_q1, target_q2)
                        if log_ent_coef is not None:
                            ent_coef = torch.exp(log_ent_coef.detach())
                        if next_log_prob.dim() == 2 and next_log_prob.shape[1] != 1:
                            next_log_prob = next_log_prob.sum(dim=1, keepdim=True)
                        else:
                            next_log_prob = next_log_prob.view(-1, 1)
                        target_q = rewards + gamma * (1 - dones) * (target_q - ent_coef * next_log_prob)

                    current_q1, current_q2 = critic(obs, actions)
                    critic_loss_t = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(
                        current_q2, target_q
                    )
                    critic_optimizer.zero_grad()
                    critic_loss_t.backward()
                    critic_optimizer.step()

                    new_actions, log_prob = actor.action_log_prob(obs)
                    q1_new, q2_new = critic(obs, new_actions)
                    q_new = torch.min(q1_new, q2_new)
                    actor_loss_t = (ent_coef * log_prob - q_new).mean()
                    actor_optimizer.zero_grad()
                    actor_loss_t.backward()
                    actor_optimizer.step()

                    if log_ent_coef is not None and ent_coef_optimizer is not None:
                        ent_coef_loss_t = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
                        ent_coef_optimizer.zero_grad()
                        ent_coef_loss_t.backward()
                        ent_coef_optimizer.step()
                    else:
                        ent_coef_loss_t = None

                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    self.total_weight_updates += 1
                    training_steps_done += 1
                    actor_loss = float(actor_loss_t.detach().item())
                    critic_loss = float(critic_loss_t.detach().item())
                    ent_coef_loss = (
                        float(ent_coef_loss_t.detach().item())
                        if ent_coef_loss_t is not None
                        else 0.0
                    )
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[server] CUDA OOM during training step {step}: {e}")
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    continue

        ent_coef_out = self._ent_coef_scalar(ent_coef, log_ent_coef)
        return {
            "training_steps_done": training_steps_done,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "ent_coef_loss": ent_coef_loss,
            "ent_coef": ent_coef_out,
        }

    def _post_training_sync(
        self,
        episodes_for_log: List[List[dict]],
        log_dict: Dict[str, Any],
        training_duration: float,
    ) -> Optional[bytes]:
        """CSV/plot/model save (runs in worker thread alongside training)."""
        if episodes_for_log:
            log_dict = dict(log_dict)
            log_dict["training_duration"] = training_duration
            log_dict["post_process_duration"] = float(log_dict.get("post_process_duration", 0.0))
            self.trainingLogHelper.log_to_csv(self.model, episodes_for_log, log_dict)

        now = time.time()
        if (
            self._model_autosave_interval_s > 0
            and (now - self._last_model_autosave_time) >= self._model_autosave_interval_s
        ):
            try:
                self._sync_model_timestep_counter()
                target = os.path.join(self.model_dir, str(self.save_model_name))
                self.model.save(target)
                self._last_model_autosave_time = now
            except Exception as e:
                print(f"[server] Error auto-saving model: {e}")

        return SacUtilities.state_dict_to_bytes(self.model.policy.actor.state_dict())

    def _train_round_sync(
        self,
        grad_steps: int,
        episodes_for_log: List[List[dict]],
        log_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Gradient updates + heavy I/O in one worker-thread call."""
        t0 = time.time()
        train_out = self._run_gradient_training(grad_steps)
        train_duration = time.time() - t0
        train_out["train_duration"] = train_duration
        train_out["post_duration"] = 0.0
        train_out["weights_blob"] = None
        if int(train_out.get("training_steps_done", 0)) > 0:
            log_dict.update(
                {
                    "actor_loss": train_out["actor_loss"],
                    "critic_loss": train_out["critic_loss"],
                    "ent_coef_loss": train_out["ent_coef_loss"],
                    "ent_coef": train_out["ent_coef"],
                    "total_weight_updates": self.total_weight_updates,
                    "total_timesteps": self.total_actor_timesteps,
                    "replay_buffer_size": int(
                        self.replay_buffer.size() if self.replay_buffer is not None else 0
                    ),
                    "UDT": self._current_udt(),
                }
            )
            t1 = time.time()
            log_dict["post_process_duration"] = 0.0
            train_out["weights_blob"] = self._post_training_sync(
                episodes_for_log, log_dict, train_duration
            )
            train_out["post_duration"] = time.time() - t1
            log_dict["post_process_duration"] = train_out["post_duration"]
        return train_out

    def _save_checkpoint_locked(self, timesteps: int):
        with self._replay_lock:
            self._sync_model_timestep_counter()
        self._save_checkpoint(timesteps)

 


    async def _train_loop(self):
        """Sample-driven training: wake on streamed transitions, train to target UTD."""
        while True:
            async with self._terminate_lock:
                if self._should_terminate:
                    print("[server] Training loop detected termination flag, exiting...")
                    return

            if self._has_received_episode and self.last_episode_time is not None:
                idle_for = time.time() - self.last_episode_time
                if idle_for >= self.episode_inactivity_timeout:
                    print(
                        "[server] No samples received for "
                        f"{idle_for:.1f}s (timeout={self.episode_inactivity_timeout:.1f}s). "
                        "Terminating server."
                    )
                    await self._await_pending_checkpoint()
                    self._save_model()
                    async with self._terminate_lock:
                        self._should_terminate = True
                    return

            try:
                await asyncio.wait_for(self._train_event.wait(), timeout=self.train_every_seconds)
            except asyncio.TimeoutError:
                pass

            if self.model is None or self.replay_buffer is None:
                continue

            new_samples = await self._take_pending_samples()
            self._train_event.clear()
            if new_samples <= 0:
                continue
            episodes_for_log = self.episode_buffer.drain_all()
            current_udt = self._current_udt()

            if self.replay_buffer.size() < self.learning_starts:
                if new_samples > 0 and Settings.LEARNER_SERVER_DEBUG:
                    needed = self.learning_starts - self.replay_buffer.size()
                    print(f"[server] Collecting... need {max(0, needed)} more samples before training.")
                continue

            grad_steps = self._compute_grad_steps(new_samples)
            if grad_steps <= 0:
                if self._training_paused_for_udt and Settings.LEARNER_SERVER_DEBUG:
                    max_udt = getattr(Settings, "SAC_MAX_UTD", None)
                    print(
                        f"[server] UDT {current_udt:.4f} at/above cap {max_udt}; "
                        "waiting for more samples."
                    )
                continue

            if self.model is not None and self.replay_buffer is not None:
                if Settings.LEARNER_SERVER_DEBUG:
                    print(
                        f"[server] Training SAC... steps={grad_steps} (new_samples={new_samples}) | "
                        f"bs={self.model.batch_size} | buffer={self.replay_buffer.size()} | UDT={current_udt:.4f}"
                    )
                time_start_round = time.time()
                stream_batch_sizes = self._stream_batch_sizes_for_episodes(episodes_for_log)
                log_dict = {
                    "actor_loss": 0.0,
                    "critic_loss": 0.0,
                    "ent_coef_loss": 0.0,
                    "ent_coef": 0.0,
                    "total_weight_updates": self.total_weight_updates,
                    "total_timesteps": self.total_actor_timesteps,
                    "replay_buffer_size": int(self.replay_buffer.size() if self.replay_buffer is not None else 0),
                    "UDT": current_udt,
                    "stream_batch_sizes": stream_batch_sizes,
                }

                async with self._train_job_lock:
                    train_out = await asyncio.to_thread(
                        self._train_round_sync, grad_steps, episodes_for_log, log_dict
                    )

                async with self._terminate_lock:
                    if self._should_terminate:
                        print("[server] Termination requested, skipping post-training save/broadcast")
                        return

                training_steps_done = int(train_out.get("training_steps_done", 0))
                if training_steps_done == 0:
                    print("[server] No gradient steps completed (early exit or repeated OOM); skipping log/broadcast.")
                    continue

                train_duration = float(train_out.get("train_duration", 0.0))
                post_duration = float(train_out.get("post_duration", 0.0))
                if training_steps_done > 0:
                    self._last_sec_per_grad_step = train_duration / float(training_steps_done)

                new_blob = train_out.get("weights_blob")
                if new_blob is not None:
                    self._weights_blob = new_blob
                    await self._broadcast_weights(new_blob)

                current_udt = self._current_udt()
                round_s = time.time() - time_start_round
                training_info_payload = self._build_training_info_payload(
                    current_udt=current_udt,
                    training_steps_done=training_steps_done,
                    training_duration=train_duration,
                )
                await self._broadcast_training_info(training_info_payload)

                self._maybe_log_train_round(
                    grad_steps=grad_steps,
                    training_steps_done=training_steps_done,
                    new_samples=new_samples,
                    current_udt=current_udt,
                    train_duration=train_duration,
                    post_duration=post_duration,
                    round_wall_s=round_s,
                )

                self._maybe_schedule_actor_timestep_checkpoints()

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

    async def _broadcast_terminate(self, reason: str) -> None:
        """Notify all connected clients to terminate their process."""
        frame = {"type": "terminate", "data": {"reason": str(reason)}}
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

    def _flush_pending_episodes_to_metrics(self) -> None:
        """
        Persist any pending, not-yet-trained episodes so terminate-triggering lap
        times are visible in learning_metrics.csv and training_metrics.png.
        """
        if self.model is None:
            return
        pending = self.episode_buffer.drain_all()
        if not pending:
            return
        try:
            log_dict = {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "ent_coef_loss": 0.0,
                "ent_coef": 0.0,
                "total_weight_updates": self.total_weight_updates,
                "training_duration": 0.0,
                "total_timesteps": self.total_actor_timesteps,
                "replay_buffer_size": int(self.replay_buffer.size() if self.replay_buffer is not None else 0),
                "UDT": self._current_udt(),
            }
            self.trainingLogHelper.log_to_csv(self.model, pending, log_dict)
            print(
                f"[server] Flushed {len(pending)} pending episode(s) to metrics before termination."
            )
        except Exception as e:
            print(f"[server] Failed to flush pending episodes before termination: {e}")

    @staticmethod
    def _parse_lap_times(laps_raw: Any) -> List[float]:
        if laps_raw is None:
            return []
        if isinstance(laps_raw, np.ndarray):
            laps_iter = laps_raw.reshape(-1).tolist()
        elif isinstance(laps_raw, (list, tuple)):
            laps_iter = laps_raw
        else:
            return []
        parsed: List[float] = []
        for lap in laps_iter:
            try:
                parsed.append(float(lap))
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def _extract_lap_times_from_episode(episode: List[dict]) -> List[float]:
        """Extract the most recent non-empty lap_times array from one episode."""
        for t in reversed(episode):
            info = t.get("info", {}) or {}
            parsed = LearnerServer._parse_lap_times(info.get("lap_times", None))
            if parsed:
                return parsed
        return []

    def _episode_hits_lap_terminate_threshold(self, episode: List[dict]) -> bool:
        """
        True when at least two completed laps in this episode are strictly faster than
        Settings.SAC_TERMINATE_BELOW_LAPTIME (seconds). Checked on streamed batches.
        """
        if self._terminate_below_lap_s is None or not episode:
            return False
        lap_times = self._extract_lap_times_from_episode(episode)
        if not lap_times:
            return False
        min_lap = min(lap_times)
        if self._best_lap_time_seen is None or min_lap < self._best_lap_time_seen:
            self._best_lap_time_seen = min_lap
        threshold = self._terminate_below_lap_s
        fast_laps = [t for t in lap_times if t < threshold]
        return len(fast_laps) >= 2

    def _episodes_to_check_for_lap_threshold(
        self,
        actor_id: int,
        episode_id: int,
        completed_episodes: List[List[dict]],
    ) -> List[List[dict]]:
        """Completed episodes from this batch plus the in-progress episode buffer."""
        candidates = list(completed_episodes)
        open_ep = self._episode_accum.get((actor_id, episode_id))
        if open_ep:
            candidates.append(open_ep)
        return candidates

    async def _trigger_lap_terminate_shutdown(self, writer: asyncio.StreamWriter) -> None:
        self._lap_terminate_triggered = True
        threshold = self._terminate_below_lap_s
        best = self._best_lap_time_seen
        print(
            "[server] Lap-time terminate condition reached: "
            f"best_lap={best:.3f}s <= threshold={threshold:.3f}s. "
            "Flushing pending episodes to metrics before shutdown."
        )
        self._flush_pending_episodes_to_metrics()
        await self._await_pending_checkpoint()
        self._save_model()
        await self._broadcast_terminate(
            reason=f"lap_time_threshold_reached(best={best:.3f},threshold={threshold:.3f})"
        )
        async with self._terminate_lock:
            self._should_terminate = True
        try:
            writer.write(
                pack_frame(
                    {
                        "type": "terminate_ack",
                        "data": {"msg": "terminating (lap-time threshold reached)"},
                    }
                )
            )
            await writer.drain()
        except Exception:
            pass

    def _build_training_info_payload(
        self,
        current_udt: float,
        training_steps_done: int,
        training_duration: float = 0.0,
    ) -> dict:
        """Build an extensible training telemetry payload."""
        target_udt = getattr(Settings, "SAC_TARGET_UTD", None)
        udt_control = None
        if target_udt is not None:
            target_udt = float(target_udt)
            fmin = float(getattr(Settings, "SAC_MIN_SIM_FREQUENCY", 20.0))
            suggested_hz = None
            if target_udt > 0 and current_udt < target_udt:
                ref_hz = float(
                    getattr(Settings, "MAX_SIM_FREQUENCY", None)
                    or getattr(Settings, "SAC_UDT_REF_SIM_FREQUENCY", 200.0)
                )
                suggested_hz = float(
                    np.clip(ref_hz * max(0.25, float(current_udt) / target_udt), fmin, ref_hz)
                )
            udt_control = {
                "target_udt": target_udt,
                "resume_udt": target_udt,
                "training_paused": bool(self._training_paused_for_udt),
                "min_sim_frequency_hz": fmin,
                "suggested_sim_frequency_hz": suggested_hz,
                "sec_per_grad_step": self._last_sec_per_grad_step,
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

                    actor_id = int(d.get("actor_id", -1))
                    episode_id = int(d.get("episode_id", 0))
                    episode_end = bool(d.get("episode_end", False))

                    batch = []
                    for i in range(len(reward_list)):
                        batch.append({
                            "obs":      obs_list[i],
                            "action":   act_list[i],
                            "next_obs": next_obs_list[i],
                            "reward":   float(reward_list[i]),
                            "done":     bool(done_list[i]),
                            "info":     info_list[i] if i < len(info_list) else {},
                            "actor_id": actor_id,
                            "episode_id": episode_id,
                        })

                    self._record_stream_batch(actor_id, episode_id, len(batch))

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
                        batch = self._apply_crash_ramp(batch, ramp_steps=20, max_ramp_value=15.0)

                    if self.model is not None:
                        n_ingested = self._ingest_transitions(batch)
                        if n_ingested > 0:
                            await self._notify_new_samples(n_ingested)
                            if Settings.LEARNER_SERVER_DEBUG:
                                print(
                                    f"[server] Ingested {n_ingested} transition(s) "
                                    f"(replay={self.replay_buffer.size()})."
                                )

                    self.total_actor_timesteps += len(batch)
                    self.last_episode_time = time.time()
                    self._has_received_episode = True

                    completed_episodes = self._accumulate_episode_batch(
                        actor_id, episode_id, batch, episode_end=episode_end
                    )
                    for completed_episode in completed_episodes:
                        if completed_episode:
                            ep_actor = int(completed_episode[0].get("actor_id", actor_id))
                            ep_id = int(completed_episode[0].get("episode_id", episode_id))
                            self._finalize_stream_batches_for_episode(ep_actor, ep_id)
                        self.episode_buffer.add_episode(completed_episode)

                    if not self._lap_terminate_triggered:
                        for ep in self._episodes_to_check_for_lap_threshold(
                            actor_id, episode_id, completed_episodes
                        ):
                            if self._episode_hits_lap_terminate_threshold(ep):
                                await self._trigger_lap_terminate_shutdown(writer)
                                break
                    if self._lap_terminate_triggered:
                        break
                    # optional ack
                    try:
                        writer.write(
                            pack_frame(
                                {
                                    "type": "episode_ack",
                                    "data": {
                                        "n": len(batch),
                                        "episode_id": episode_id,
                                        "episode_done": bool(completed_episodes),
                                        "episode_len": (
                                            len(completed_episodes[-1]) if completed_episodes else None
                                        ),
                                    },
                                }
                            )
                        )
                        await writer.drain()
                    except Exception:
                        pass
                elif msg.get("type") == "clear_buffer":
                    d = msg.get("data", {})
                    actor_id = int(d.get("actor_id", -1))

                    # Clear the episode buffer and in-flight episode assembly
                    episodes_cleared = len(self.episode_buffer.episodes)
                    self.episode_buffer.episodes.clear()
                    self._episode_accum.clear()
                    self._max_episode_id_per_actor.clear()
                    self._stream_batches_by_episode.clear()
                    self._finalized_stream_batches.clear()

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

                    # Ensure final lap times are persisted even if terminate arrives
                    # before the next train-loop tick.
                    self._flush_pending_episodes_to_metrics()
                    await self._await_pending_checkpoint()
                    self._save_model()
                    await self._broadcast_terminate(reason=f"terminate_requested_by_actor_{actor_id}")
                    
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

        metrics_http: Optional[MetricsHttpServer] = None
        if getattr(Settings, "LEARNER_METRICS_HTTP_ENABLED", True):
            metrics_http = MetricsHttpServer(
                host=self.host,
                port=int(getattr(Settings, "LEARNER_METRICS_HTTP_PORT", 5556)),
                csv_path=self.trainingLogHelper.csv_path,
                model_name=self.save_model_name,
                poll_hint_s=float(getattr(Settings, "LEARNER_METRICS_HTTP_POLL_S", 2.0)),
            )
            await metrics_http.start()

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

            if metrics_http is not None:
                await metrics_http.close()
            
            print("[server] Saving model before exit...")
            await self._await_pending_checkpoint()
            self._save_model()
            
            print("[server] Shutdown complete")

