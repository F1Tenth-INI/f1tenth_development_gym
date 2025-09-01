#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import os
import sys
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

class LearnerServer:
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        device: str = "cpu",
        train_every_seconds: float = 10.0,
        grad_steps: int = 1024,
        replay_capacity: int = 100_000,
        learning_starts: int = 2000
    ):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.device = device
        self.train_every_seconds = train_every_seconds
        self.replay_capacity = replay_capacity

        # Settings
        self.learning_starts = learning_starts
        self.grad_steps = grad_steps

        self.init_from_scratch = None
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
        self.total_actor_timesteps = 0
        self.total_weight_updates = 0


        # file paths
        self.model_path, self.model_dir = SacUtilities.resolve_model_paths(self.model_name)

        self.trainingLogHelper = TrainingLogHelper(self.model_name,self.model_dir)

        self._initialize_model()

    # ---------- setup / init ----------
    def _load_base_model(self):
        """
        Initialize the model + replay buffer.

        - Check if model already exists
        - Else: load from zip and bind a matching dummy env.
        """
        dummy_env = SacUtilities.create_vec_env()
     
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.model = SAC.load(self.model_path, env=dummy_env, device=self.device)
            print(f"[server] Success: Loaded SAC model: {self.model_name}")

        except Exception as e:
            print(f"[server] No SAC model located at {self.model_path}.")
            print("Creating new model.")
            self.model = SacUtilities.create_model(
                env=dummy_env, 
                buffer_size=self.replay_capacity, 
                device=self.device
            )
            print(f"[server] Success: Created new SAC model.")

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
        self._weights_blob = SacUtilities.state_dict_to_bytes(self.model.policy.actor.state_dict())

        print("[server] Base model + replay buffer initialized.",
            f"Mode={'scratch' if self.init_from_scratch else 'from-zip'}")


    def _initialize_model(self):
        if self.model is not None: return
        self._load_base_model()
        print(f"[server] Initialized model")

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

            n_added = self._ingest_episodes_into_replay(episodes)
            print(f"[server] Ingested {len(episodes)} episodes / {n_added} transitions into replay "
                f"(size={self.replay_buffer.size() if self.replay_buffer else 0}).")


            # Train if we have enough samples
            if self.model is not None and self.replay_buffer is not None and self.replay_buffer.size() >= self.learning_starts:
                # gradient steps proportional to newly ingested data (UTD ≈ 4)
                grad_steps = self.grad_steps


                print(f"[server] Training SAC... steps={grad_steps} | buffer size={self.replay_buffer.size()}")
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
                    then = time.time()
                    try:
                        # Reduce batch size to avoid OOM
                        safe_batch_size = min(batch_size, 4096)
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

                        # Free unused memory
                        # if self.device == "cuda":
                        #     torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[server] CUDA OOM during training step {step}: {e}")
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        continue
                # You can now interfere with actor, critic, optimizers, etc. in this loop
                
                self.model.actor_loss = actor_loss.item()
                self.model.critic_loss = critic_loss.item()
                self.model.ent_coef_loss = ent_coef_loss.item()
                self.model.ent_coef = ent_coef if isinstance(ent_coef, float) else ent_coef.item()
                self.model.total_weight_updates = self.total_weight_updates
                self.model.training_duration = time.time() - time_start_training
                self.model._total_timesteps = self.total_actor_timesteps
                print(f"[server] Training completed in {(time.time() - time_start_training):.2f} seconds.")

                self.trainingLogHelper.log_to_csv(self.model, episodes)

                new_blob = SacUtilities.state_dict_to_bytes(self.model.policy.actor.state_dict())
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
                        self._initialize_model()

                    self.episode_buffer.add_episode(episode)
                    print(f"[server] Stored episode: {len(episode)} transitions "
                        f"(total episodes pending train: {len(self.episode_buffer.episodes)})")
                    self.total_actor_timesteps += len(episode)
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
            self._load_base_model() 

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

# python TrainingLite/rl_racing/learner_server.py   --model-name SAC_RCA1_wpts_lidar_50
# python TrainingLite/rl_racing/learner_server.py   --init-from-scratch --model-name SAC_RCA1_wpts_lidar_50

def main():

    model_name = "SAC_RCA1_wpts_lidar_50_async"
    grad_steps = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_every_seconds = 1.0
    replay_capacity = 100_000
    learning_starts = 2000



    import argparse
    parser = argparse.ArgumentParser(description="Learner server: collect episodes, train SAC, broadcast weights.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--device", default=device, choices=["cpu", "cuda"])
    parser.add_argument("--train-every-seconds", type=float, default=train_every_seconds)
    parser.add_argument("--gradient-steps", type=int, default=grad_steps)
    parser.add_argument("--replay-capacity", type=int, default=replay_capacity)
    parser.add_argument("--learning-starts", type=int, default=learning_starts)

    args = parser.parse_args()

    srv = LearnerServer(
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        device=args.device,
        train_every_seconds=args.train_every_seconds,
        grad_steps=args.gradient_steps,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts
    )
    asyncio.run(srv.run())

if __name__ == "__main__":
    main()
