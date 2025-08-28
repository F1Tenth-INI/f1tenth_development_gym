import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyparsing import deque
import io
import torch
from stable_baselines3 import SAC
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from stable_baselines3.common.vec_env import DummyVecEnv


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




        
class SacUtilities:

    # --- define spaces ---
    obs_low  = np.array([-1, -1, -1, -1] + [-1]*30 + [0]*40 + [-1]*6, dtype=np.float32)
    obs_high = np.array([ 1,  1,  1,  1] + [ 1]*30 + [1]*40 + [ 1]*6, dtype=np.float32)
    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    act_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32), high=np.array([ 1,  1], dtype=np.float32), dtype=np.float32)


    def make_env():
        return _SpacesOnlyEnv(SacUtilities.obs_space, SacUtilities.act_space)

    def create_vec_env():
        return DummyVecEnv([SacUtilities.make_env])

    @staticmethod
    def create_model(env, buffer_size=100_000, device="cpu"):
        policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh)

        model = SAC(
                    "MlpPolicy",
                    env=env,
                    verbose=0,
                    train_freq=1,
                    gamma=0.99,
                    learning_rate=1e-3,
                    policy_kwargs=policy_kwargs,
                    buffer_size=buffer_size,
                    device=device,
                    batch_size=256,
                )
        return model
    

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
    # Torch serialization helpers
    # ------------------------------
    def state_dict_to_bytes(sd: Dict[str, Any]) -> bytes:
        buf = io.BytesIO() 
        cpu_sd = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
        torch.save(cpu_sd, buf)
        return buf.getvalue()


    


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