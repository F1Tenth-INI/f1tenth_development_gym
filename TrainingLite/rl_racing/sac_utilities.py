import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import SAC

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