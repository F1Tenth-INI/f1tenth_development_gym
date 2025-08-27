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
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}(.zip)")
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
    ):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.device = device
        self.train_every_seconds = train_every_seconds
        self.gradient_steps_per_round = gradient_steps_per_round
        self.replay_capacity = replay_capacity


        self.learning_starts = 10000
        self.utd_ratio = 4.0
        self.min_grad_steps = 16
        self.max_grad_steps = 4096

        # networking
        self._clients: set[asyncio.StreamWriter] = set()
        self._client_lock = asyncio.Lock()

        # data
        self.episode_buffer = EpisodeReplayBuffer(capacity_episodes=2000)
        self._weights_blob: Optional[bytes] = None

        # RL bits (lazy init for shapes that we infer from first episode)
        self.model: Optional[SAC] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.vecnorm: Optional[VecNormalize] = None
        self._obs_dim: Optional[int] = None
        self._act_dim: Optional[int] = None

        # file paths
        self.model_path, self.model_dir = resolve_model_paths(self.model_name)

    # ---------- setup / init ----------
    def _load_base_model(self, _obs_dim_ignored: int, _act_dim_ignored: int):
        """
        Probe the saved model to get the exact observation/action spaces,
        then load again with a DummyVecEnv that advertises those spaces.
        """
        print(f"[server] Loading SAC base model from {self.model_path}")

        # 1) Probe to read saved spaces (no env binding checks here)
        probe = SAC.load(self.model_path, device=self.device)
        exp_obs_space: spaces.Box = probe.observation_space
        exp_act_space: spaces.Box = probe.action_space
        # remember dims (optional)
        self._obs_dim = int(np.prod(exp_obs_space.shape))
        self._act_dim = int(np.prod(exp_act_space.shape))
        # free probe asap (esp. on CUDA)
        del probe
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # 2) Build a 1-env DummyVecEnv with EXACT same spaces
        dummy_env = self._dummy_vec_from_spaces(exp_obs_space, exp_act_space)

        # 3) Load again, now binding env for SB3 validation
        self.model = SAC.load(self.model_path, env=dummy_env, device=self.device)

        # Minimal logger so `.train()` works outside `.learn()`
        self.model._logger = configure(folder=None, format_strings=[])

        # Fresh replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.replay_capacity,
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            device=self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        self.model.replay_buffer = self.replay_buffer

        # Cache weights for initial broadcast
        self._weights_blob = state_dict_to_bytes(self.model.policy.actor.state_dict())

        # (Optional) sanity: verify spaces match
        ms = self.model.observation_space
        assert np.allclose(ms.low,  dummy_env.observation_space.low)
        assert np.allclose(ms.high, dummy_env.observation_space.high)

        print("[server] Base model + replay buffer initialized with matched spaces.")


    def _ensure_initialized(self, episode: List[dict]):
        """Lazy init once we know obs/action dims from the first received episode."""
        if self.model is not None:
            return
        if not episode:
            return
        obs0 = episode[0]["obs"]
        act0 = episode[0]["action"]
        self._obs_dim = int(obs0.shape[-1])
        self._act_dim = int(act0.shape[-1])
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
                self.model.train(gradient_steps=grad_steps, batch_size=self.model.batch_size)

                new_blob = state_dict_to_bytes(self.model.policy.actor.state_dict())
                self._weights_blob = new_blob
                await self._broadcast_weights(new_blob)
                print("[server] Trained SAC and broadcast updated actor weights.")
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

        # send current weights (base or latest)
        if self._weights_blob is None:
            # If training hasn't initialized yet, we can proactively load base model
            # with guessed dimensions (your defaults): obs=80, act=2.
            # Or just wait until first episode arrives. We'll try to send nothing if unknown.
            pass
        else:
            try:
                await self._broadcast_weights(self._weights_blob)
                print(f"[server] Weights sent to {addr}")
            except Exception:
                pass

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
    parser.add_argument("--model-name", default="SAC_RCA1_wpts_lidar_38_unnormalized")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--train-every-seconds", type=float, default=10.0)
    parser.add_argument("--gradient-steps", type=int, default=1024)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    args = parser.parse_args()

    srv = LearnerServer(
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        device=args.device,
        train_every_seconds=args.train_every_seconds,
        gradient_steps_per_round=args.gradient_steps,
        replay_capacity=args.replay_capacity,
    )
    asyncio.run(srv.run())

if __name__ == "__main__":
    main()
