#!/usr/bin/env python3
"""
Pretrain an SB3 SAC actor with Behavior Cloning (BC) on a transitions CSV.

CSV columns expected:
timestamp, obs, action, next_obs, reward, done, info
where `obs`, `action`, `next_obs` are stringified arrays like "[0.1, 0.2, ...]".

Usage:
  python pretrain_sac_bc.py --csv transitions.csv --epochs 50 --batch_size 1024

Outputs:
  - sac_pretrained_actor.zip (SB3 model with actor BC-pretrained)
  - bc_stats.txt (simple training/validation MSE log)
"""

import argparse
import ast
import math
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sac_utilities import SacUtilities

# ----------------------------
# Parsing utilities
# ----------------------------

def parse_array(s: str, dtype=np.float32) -> np.ndarray:
    """
    Robustly parse a string like "[0.1, 0.2, ...]" to a numpy array.
    """
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=dtype)
    s = str(s).strip()
    # Fast path: ast to list
    try:
        arr = np.asarray(ast.literal_eval(s), dtype=dtype)
        return arr
    except Exception:
        # Fallback: strip brackets and use fromstring with commas
        s = s.strip("[]")
        arr = np.fromstring(s, sep=",", dtype=dtype)
        return arr

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path, engine="python")
    obs = np.stack(df["obs"].apply(parse_array).to_list()).astype(np.float32)
    act = np.stack(df["action"].apply(parse_array).to_list()).astype(np.float32)
    next_obs = np.stack(df["next_obs"].apply(parse_array).to_list()).astype(np.float32)
    rew = df["reward"].astype(np.float32).to_numpy()
    done = df["done"].astype(bool).to_numpy()
    return obs, act, next_obs, rew, done



# ----------------------------
# Behavior Cloning trainer
# ----------------------------

def bc_pretrain_actor(
    model: SAC,
    obs: np.ndarray,
    act: np.ndarray,
    epochs: int = 50,
    batch_size: int = 1024,
    val_split: float = 0.1,
    lr: float = 3e-4,
    shuffle: bool = True,
    save_every: int = 0,
    out_prefix: str = "sac_pretrained_actor",
):
    """
    Train only the actor of an SB3 SAC model to imitate expert actions (MSE on deterministic actions).
    """
    device = th.device(model.device)
    N = obs.shape[0]

    # Train/val split
    n_val = max(1, int(val_split * N))
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    obs_tr = th.as_tensor(obs[tr_idx], device=device)
    act_tr = th.as_tensor(act[tr_idx], device=device)
    obs_val = th.as_tensor(obs[val_idx], device=device)
    act_val = th.as_tensor(act[val_idx], device=device)

    # Optimizer for actor only
    actor = model.policy.actor
    actor_opt = th.optim.Adam(actor.parameters(), lr=lr)

    def actor_predict(o_batch: th.Tensor) -> th.Tensor:
        """
        Deterministic policy action as a differentiable tensor in env action scale.
        Handles SB3 versions that return (action) or (action, ...).
        """
        actor = model.policy.actor  # or capture 'actor' from outer scope if you prefer
        actor.train()  # ensure gradients flow
        out = actor(o_batch, deterministic=True)
        if isinstance(out, (tuple, list)):
            pred = out[0]
        else:
            pred = out
        return pred

    # Simple minibatch iterator
    def iterate_minibatches(X: th.Tensor, Y: th.Tensor, batch: int):
        M = X.shape[0]
        order = th.randperm(M, device=X.device)
        for start in range(0, M, batch):
            end = min(start + batch, M)
            idx_mb = order[start:end]
            yield X[idx_mb], Y[idx_mb]

    # Training loop
    best_val = math.inf
    log_lines = []
    for ep in range(1, epochs + 1):
        # Train
        actor.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in iterate_minibatches(obs_tr, act_tr, batch_size):
            actor_opt.zero_grad(set_to_none=True)
            pred = actor_predict(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            th.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            actor_opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1

        train_mse = total_loss / max(1, n_batches)

        # Validate
        actor.eval()
        with th.no_grad():
            val_pred = actor_predict(obs_val)
            val_mse = F.mse_loss(val_pred, act_val).item()

        line = f"Epoch {ep:03d} | train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f}"
        print(line)
        log_lines.append(line)

        # Keep best
        if val_mse < best_val:
            best_val = val_mse
            model.save(f"{out_prefix}.zip")

        # Optional periodic saves
        if save_every and ep % save_every == 0:
            model.save(f"{out_prefix}_ep{ep}.zip")

    with open("bc_stats.txt", "w") as f:
        f.write("\n".join(log_lines))
    print(f"Best val MSE: {best_val:.6f} | Saved: {out_prefix}.zip")




def pretrain_critic(
    model: SAC,
    obs: np.ndarray,
    act: np.ndarray,
    next_obs: np.ndarray,
    rew: np.ndarray,
    done: np.ndarray,
    epochs: int = 10,
    batch_size: int = 1024,
    gamma: float = 0.99,
):
    """
    Offline SAC critic pretraining: runs Q-updates on logged transitions.
    """
    device = th.device(model.device)

    obs = th.as_tensor(obs, device=device)
    act = th.as_tensor(act, device=device)
    next_obs = th.as_tensor(next_obs, device=device)
    rew = th.as_tensor(rew, device=device).unsqueeze(-1)  # shape (N,1)
    done = th.as_tensor(done.astype(np.float32), device=device).unsqueeze(-1)

    critic = model.policy.critic
    critic_target = model.policy.critic_target
    critic_opt = model.policy.critic.optimizer
    alpha = th.exp(model.log_alpha.detach()) if hasattr(model, "log_alpha") else 0.2

    N = obs.shape[0]

    def iterate_minibatches():
        idx = th.randperm(N, device=device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            yield idx[start:end]

    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for mb_idx in iterate_minibatches():
            obs_b, act_b, next_obs_b, rew_b, done_b = (
                obs[mb_idx],
                act[mb_idx],
                next_obs[mb_idx],
                rew[mb_idx],
                done[mb_idx],
            )

            with th.no_grad():
                # Sample next action from current actor
                next_action, next_logp = model.policy.actor.action_log_prob(next_obs_b)
                q1_next, q2_next = critic_target(next_obs_b, next_action)
                q_next = th.min(q1_next, q2_next) - alpha * next_logp.unsqueeze(-1)
                target_q = rew_b + gamma * (1 - done_b) * q_next

            q1, q2 = critic(obs_b, act_b)
            loss_q = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            critic_opt.zero_grad(set_to_none=True)
            loss_q.backward()
            th.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            critic_opt.step()

            total_loss += float(loss_q.detach().cpu())
            n_batches += 1

        print(f"[Critic Pretrain] Epoch {ep+1}/{epochs} | loss_q: {total_loss/n_batches:.6f}")

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to transitions CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=str, default="auto")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_kwargs", type=str, default="", help="e.g. 'net_arch=[256,256]'")
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    obs, act, next_obs, rew, done = load_dataset(args.csv)
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]
    print(f"Dataset: {obs.shape[0]} samples | obs_dim={obs_dim} | act_dim={act_dim}")

    # Infer action bounds from data (robust for mixed ranges like steer in [-1,1], throttle in [0,1])
    a_low = act.min(axis=0)
    a_high = act.max(axis=0)

    # Safety margins to avoid zero-width bounds
    eps = 1e-3
    a_low = np.minimum(a_low, a_high - eps).astype(np.float32)
    a_high = np.maximum(a_high, a_low + eps).astype(np.float32)

    # Build dummy env
    env = SacUtilities.make_env()
    # Initialize SAC
    model = SacUtilities.create_model(env)
    
    # Behavior Cloning pretrain of the actor
    bc_pretrain_actor(
        model=model,
        obs=obs,
        act=act,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.actor_lr,
        out_prefix="sac_pretrained_actor",
    )
    
    
    # Critic offline pretrain
    pretrain_critic(
        model=model,
        obs=obs,
        act=act,
        next_obs=next_obs,
        rew=rew,
        done=done,
        epochs=10,           # you can increase this (e.g. 50)
        batch_size=args.batch_size,
        gamma=args.gamma,
    )


    print("\nDone. You can now fine-tune online with SAC, e.g.:")
    print("    model = SAC.load('sac_pretrained_actor.zip', env=your_real_env)")
    print("    model.learn(total_timesteps=..., reset_num_timesteps=False)")

if __name__ == "__main__":
    main()
