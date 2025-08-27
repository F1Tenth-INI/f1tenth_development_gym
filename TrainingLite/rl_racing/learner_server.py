#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import io
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.logger import configure

from tcp_utilities import pack_frame, read_frame, blob_to_np  # shared utils (JSON + base64 framing)
import time
# ------------------------------
# Tiny env just to define spaces (no stepping)
# ------------------------------
class _SpacesOnlyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_space: spaces.Box, act_space: spaces.Box):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):  # never used
        raise RuntimeError("_SpacesOnlyEnv is not meant to be stepped")
    



# ------------------------------
# Simple in-memory episode buffer
# ------------------------------
class EpisodeReplayBuffer:
    def __init__(self, capacity_episodes: int = 2000):
        self.episodes: deque[List[dict]] = deque(maxlen=capacity_episodes)
        self.total_transitions = 0

    def add_episode(self, episode: List[dict]):
        self.episodes.append(episode)
        self.total_transitions += len(episode)

    def drain_all(self) -> List[List[dict]]:
        """Pop all stored episodes and return them."""
        items = list(self.episodes)
        self.episodes.clear()
        return items

# ------------------------------
# Torch serialization helpers
# ------------------------------
def state_dict_to_bytes(sd: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    cpu_sd = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
    torch.save(cpu_sd, buf)
    return buf.getvalue()

# ------------------------------
# Model path helpers
# ------------------------------
def resolve_model_paths(model_name: str) -> Tuple[str, str]:
    """
    Return (model_path, model_dir)
    Layout:
      root/TrainingLite/rl_racing/models/{model_name}/{model_name}.zip
      root/TrainingLite/rl_racing/models/{model_name}/vecnormalize.pkl (optional)
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(root_dir, "TrainingLite", "rl_racing", "models", model_name)
    model_path = os.path.join(model_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    # if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model not found: {model_path}(.zip)")
    return model_path, model_dir

# ------------------------------
# Core server with training
# ------------------------------
class LearnerServer:
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        device: str = "cpu",
        train_every_seconds: float = 10.0,
        gradient_steps_per_round: int = 1024,
        replay_capacity: int = 100_000,
        init_from_scratch: bool = False,
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
    ):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.device = device
        self.train_every_seconds = train_every_seconds
        self.gradient_steps_per_round = gradient_steps_per_round
        self.replay_capacity = replay_capacity

        # easier cold start; you can restore 10000 later
        self.learning_starts = 2000
        self.utd_ratio = 4.0
        self.min_grad_steps = 16
        self.max_grad_steps = 4096

        self.init_from_scratch = init_from_scratch
        # optional bootstrap dims when starting from scratch (required then)
        self._boot_obs_dim = obs_dim
        self._boot_act_dim = act_dim
        
        # networking
        self._clients: set[asyncio.StreamWriter] = set()
        self._client_lock = asyncio.Lock()

        # data
        self.episode_buffer = EpisodeReplayBuffer(capacity_episodes=2000)
        self._weights_blob: Optional[bytes] = None

        # RL bits (lazy init fallback; but we’ll eager-init in run())
        self.model: Optional[SAC] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.vecnorm: Optional[VecNormalize] = None
        self._obs_dim: Optional[int] = None
        self._act_dim: Optional[int] = None

        # file paths
        self.model_path, self.model_dir = resolve_model_paths(self.model_name)

    # ---------- setup / init ----------
    def _load_base_model(self, obs_dim: Optional[int], act_dim: Optional[int]):
        """
        Initialize the model + replay buffer.

        - If init_from_scratch: build fresh SAC with spaces from (obs_dim, act_dim).
        (Both must be provided.)
        - Else: load from zip and bind a matching dummy env.
        """
        if self.init_from_scratch:
            if obs_dim is None or act_dim is None:
                raise ValueError("When --init-from-scratch is set, you must provide --obs-dim and --act-dim.")
            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            dummy_env = self._dummy_vec_from_spaces(obs_space, act_space)

            policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh)
            print("[server] Initializing fresh SAC from scratch (no zip).")
            self.model = SAC(
                "MlpPolicy",
                env=dummy_env,
                verbose=0,
                train_freq=1,
                gamma=0.99,
                learning_rate=1e-3,
                policy_kwargs=policy_kwargs,
                buffer_size=self.replay_capacity,
                device=self.device,
                batch_size=256,
            )
            self._obs_dim = obs_dim
            self._act_dim = act_dim
        else:
            print(f"[server] Loading SAC base model from {self.model_path}")
            probe = SAC.load(self.model_path, device=self.device)
            exp_obs_space: spaces.Box = probe.observation_space
            exp_act_space: spaces.Box = probe.action_space
            self._obs_dim = int(np.prod(exp_obs_space.shape))
            self._act_dim = int(np.prod(exp_act_space.shape))
            del probe
            if self.device == "cuda":
                torch.cuda.empty_cache()
            dummy_env = self._dummy_vec_from_spaces(exp_obs_space, exp_act_space)
            self.model = SAC.load(self.model_path, env=dummy_env, device=self.device)

        # Minimal logger so .train() works outside .learn()
        self.model._logger = configure(folder=None, format_strings=[])

        # Fresh replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.replay_capacity,
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            device=self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        self.model.replay_buffer = self.replay_buffer

        # Cache weights for initial broadcast (random if scratch)
        self._weights_blob = state_dict_to_bytes(self.model.policy.actor.state_dict())

        print("[server] Base model + replay buffer initialized.",
            f"Mode={'scratch' if self.init_from_scratch else 'from-zip'}",
            f"obs_dim={self._obs_dim}, act_dim={self._act_dim}")




    def _ensure_initialized(self, episode: List[dict]):
        if self.model is not None or not episode: return
        obs0 = episode[0]["obs"]; act0 = episode[0]["action"]
        self._obs_dim = int(obs0.shape[-1]); self._act_dim = int(act0.shape[-1])
        self._load_base_model(self._obs_dim, self._act_dim)
        print(f"[server] Initialized model with obs_dim={self._obs_dim}, act_dim={self._act_dim}")

    # ---------- ingestion + training ----------
    def _normalize_obs(self, x: np.ndarray) -> np.ndarray:
        if self.vecnorm is None:
            return x.astype(np.float32)
        # VecNormalize expects (n_envs, obs_dim)
        return self.vecnorm.normalize_obs(x[None, :]).astype(np.float32)[0]
    
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
                self.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos={},  # could pass TimeLimit info here if you track it
                )
                n_added += 1
        return n_added

    async def _train_loop(self):
        """Periodic training loop: drain new episodes -> add to replay -> train -> broadcast new weights."""
        while True:
            await asyncio.sleep(self.train_every_seconds)

            # Drain whatever the actor sent since last round
            episodes = self.episode_buffer.drain_all()
            if not episodes:
                continue

            # Lazy init model on first batch (we need shapes)
            self._ensure_initialized(episodes[0])

            n_added = self._ingest_episodes_into_replay(episodes)
            print(f"[server] Ingested {len(episodes)} episodes / {n_added} transitions into replay "
                f"(size={self.replay_buffer.size() if self.replay_buffer else 0}).")


            # Train if we have enough samples
            # Train if we have enough samples
            if self.model is not None and self.replay_buffer is not None and self.replay_buffer.size() >= self.learning_starts:
                # gradient steps proportional to newly ingested data (UTD ≈ 4)
                grad_steps = int(self.utd_ratio * max(1, n_added))
                grad_steps = max(self.min_grad_steps, min(grad_steps, self.max_grad_steps))

                # initialize logger/progress (if not already)
                if not hasattr(self.model, "_logger") or self.model._logger is None:
                    self.model._logger = configure(folder=None, format_strings=[])
                if not hasattr(self.model, "_n_updates"):
                    self.model._n_updates = 0
                if not hasattr(self.model, "_current_progress_remaining"):
                    self.model._current_progress_remaining = 1.0

                print(f"[server] Training SAC... steps={grad_steps} | buffer size={self.replay_buffer.size()}")
                time_start_training = time.time()
                self.model.train(
                    gradient_steps=grad_steps, 
                    batch_size=self.model.batch_size
                    )

                # Print useful training info
                logger = self.model.logger
                def get_metric(name):
                    # Try to get the last recorded value for a metric
                    if hasattr(logger, 'name_to_value') and name in logger.name_to_value:
                        return logger.name_to_value[name]
                    if hasattr(logger, 'recorded_values') and name in logger.recorded_values:
                        return logger.recorded_values[name][-1]
                    return None

                actor_loss = get_metric('train/actor_loss')
                critic_loss = get_metric('train/critic_loss')
                ent_coef = get_metric('train/ent_coef')
                ent_coef_loss = get_metric('train/ent_coef_loss')
                total_updates = getattr(self.model, '_n_updates', None)
                num_episodes = len(self.episode_buffer.episodes)

                print(f"[server] Training completed in {(time.time() - time_start_training):.2f} seconds.")
                print(f"[server] Metrics: actor_loss={actor_loss}, critic_loss={critic_loss}, ent_coef={ent_coef}, ent_coef_loss={ent_coef_loss}, total_updates={total_updates}, episodes={num_episodes}")

                new_blob = state_dict_to_bytes(self.model.policy.actor.state_dict())
                self._weights_blob = new_blob
                await self._broadcast_weights(new_blob)
                print("[server] Trained SAC and broadcast updated actor weights.")

                self.model.save(os.path.join(self.model_dir, self.model_name))
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

                    # initialize shapes if this is the very first episode
                    if self.model is None:
                        self._ensure_initialized(episode)

                    self.episode_buffer.add_episode(episode)
                    print(f"[server] Stored episode: {len(episode)} transitions "
                        f"(total episodes pending train: {len(self.episode_buffer.episodes)})")

                    # optional ack
                    try:
                        writer.write(pack_frame({"type": "episode_ack", "data": {"n": len(episode)}}))
                        await writer.drain()
                    except Exception:
                        pass
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
            if self.init_from_scratch:
                self._load_base_model(self._boot_obs_dim, self._boot_act_dim)
            else:
                # For non-scratch, load from existing zip (shapes inferred inside)
                self._load_base_model(None, None)

        # start trainer task
        asyncio.create_task(self._train_loop())

        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        print(f"[server] Listening on {addrs} | model='{self.model_name}' | device='{self.device}'")
        async with server:
            await server.serve_forever()


# ------------------------------
# CLI
# ------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Learner server: collect episodes, train SAC, broadcast weights.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-name", default="SAC_RCA1_wpts_lidar_50_async")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--train-every-seconds", type=float, default=10.0)
    parser.add_argument("--gradient-steps", type=int, default=1024)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--init-from-scratch", action="store_true",
                        help="Initialize a fresh SAC (no zip). Requires --obs-dim and --act-dim.")
    parser.add_argument("--obs-dim", type=int, default=80, help="Observation dimension (required for --init-from-scratch)")
    parser.add_argument("--act-dim", type=int, default=2, help="Action dimension (required for --init-from-scratch)")

    args = parser.parse_args()

    srv = LearnerServer(
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        device=args.device,
        train_every_seconds=args.train_every_seconds,
        gradient_steps_per_round=args.gradient_steps,
        replay_capacity=args.replay_capacity,
        init_from_scratch=args.init_from_scratch,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
    )
    asyncio.run(srv.run())

if __name__ == "__main__":
    main()
