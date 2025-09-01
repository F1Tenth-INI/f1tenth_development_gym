import time
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyparsing import deque
import io
import torch
from stable_baselines3 import SAC
import os
import sys
import csv
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
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
    obs_low  = np.array([-1, -1, -1, -1] + [-1]*30 + [0]*40 + [-1]*6 + [-1]*2, dtype=np.float32)
    obs_high = np.array([ 1,  1,  1,  1] + [ 1]*30 + [1]*40 + [ 1]*6 + [ 1]*2, dtype=np.float32)
    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    act_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32), high=np.array([ 1,  1], dtype=np.float32), dtype=np.float32)

    @staticmethod
    def make_env():
        return _SpacesOnlyEnv(SacUtilities.obs_space, SacUtilities.act_space)

    @staticmethod
    def create_vec_env():
        return DummyVecEnv([SacUtilities.make_env])

    @staticmethod
    def create_model(env, buffer_size=100_000, device="cpu") -> SAC:
        policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh)
        #  log_std_init=-3.5

        model = SAC(
                    "MlpPolicy",
                    env=env,
                    verbose=0,
                    # ent_coef=0.01,
                    train_freq=1,
                    gamma=0.99,
                    learning_rate=1e-3,
                    policy_kwargs=policy_kwargs,
                    buffer_size=buffer_size,
                    device=device,
                    batch_size=1024,
                )
        return model
    
    @staticmethod
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
    @staticmethod
    def state_dict_to_bytes(sd: Dict[str, Any]) -> bytes:
        buf = io.BytesIO() 
        cpu_sd = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
        torch.save(cpu_sd, buf)
        return buf.getvalue()
    


    

class TrainingLogHelper():
    def __init__(self, model_name: str, model_dir: str):
        self.model_name = model_name
        self.model_dir = model_dir
        self.csv_path = os.path.join(model_dir, f"learning_metrics.csv")

        self.training_index = 0
        self.plot_every = 5

    def log_to_csv(self, model, episodes):

        file_exists = os.path.isfile(self.csv_path)
        
        # Gather all available metrics from logger
        metric_dict = {}

        # Add lap_times, min_laptime, avg_laptime from the last episode's info if available
        last_episode = episodes[-1] if episodes else None
        last_info = last_episode[-1]["info"] if last_episode and last_episode[-1] and "info" in last_episode[-1] else {}
        lap_times = last_info.get("lap_times", None)
        min_laptime = last_info.get("min_laptime", None)
        avg_laptime = last_info.get("avg_laptime", None)
        metric_dict['lap_times'] = str(lap_times) if lap_times is not None else ""
        metric_dict['min_laptime'] = min_laptime if min_laptime is not None else ""
        metric_dict['avg_laptime'] = avg_laptime if avg_laptime is not None else ""
        metric_dict['training_duration'] = getattr(model, 'training_duration', None)

        metric_dict['total_timesteps'] = getattr(model, '_total_timesteps', None)
        metric_dict['actor_loss'] = getattr(model, 'actor_loss', None)
        metric_dict['critic_loss'] = getattr(model, 'critic_loss', None)
        metric_dict['ent_coef'] = getattr(model, 'ent_coef', None)
        metric_dict['ent_coef_loss'] = getattr(model, 'ent_coef_loss', None)
        metric_dict['total_weight_updates'] = getattr(model, 'total_weight_updates', None)

        # Write to CSV
        with open(self.csv_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp"] + list(metric_dict.keys()))
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S")
            ] + [metric_dict[k] for k in metric_dict.keys()])

        self.training_index += 1

        if self.training_index % self.plot_every == 0:
            self.plot_training_metrics()

    def plot_training_metrics(self):
        model_dir = self.model_dir
        csv_path=self.csv_path

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        # Load the CSV
        df = pd.read_csv(csv_path)
        print(df.columns)


        # Downsample and window settings
        downsample_step = 1


        # Downsample
        df_plot = df.iloc[::downsample_step].copy()

        # Timestep is saved as date string 2025-08-28 23:09:26 and needs to be converted to seconds
        # Convert to datetime
        df_plot['timestamp_dt'] = pd.to_datetime(df_plot['timestamp'])
        # Calculate seconds since first timestamp
        x_vals = (df_plot['timestamp_dt'] - df_plot['timestamp_dt'].iloc[0]).dt.total_seconds().values


        # Remove 'timestamp' from columns to plot
        columns_to_plot = [col for col in df_plot.columns if col not in ['lap_times', 'timestamp']]
        n_cols = len(columns_to_plot)
        fig, axs = plt.subplots(n_cols, 1, figsize=(10, 3 * n_cols), sharex=True)

        if n_cols == 1:
            axs = [axs]


        for i, col in enumerate(columns_to_plot):
            y_vals = df_plot[col].values
            if col in ['min_laptime', 'avg_laptime']:
                axs[i].scatter(x_vals, y_vals, label=col, s=10)
            else:
                axs[i].plot(x_vals, y_vals, label=col)
            axs[i].set_ylabel(col)
            axs[i].legend()
            axs[i].grid(True)

        axs[-1].set_xlabel('timestamp')

        # Improve x-axis readability: fewer ticks, rotated labels
        import matplotlib.ticker as ticker
        max_xticks = 10
        axs[-1].xaxis.set_major_locator(ticker.MaxNLocator(max_xticks))
        for ax in axs:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_metrics.png'))




class TransitionLogger:
    def __init__(self):
        self.transitions = []
        self.file_path = "transitions_log.csv"

    def log(self, transition: dict):
        self.transitions.append(transition)

    def save_csv(self):
        
        if not self.transitions:
            return
        import time
        file_exists = os.path.isfile(self.file_path)
        # Use keys from the first transition for header
        if not self.transitions:
            return
        keys = list(self.transitions[0].keys())
        with open(self.file_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp"] + keys)
            for t in self.transitions:
                row = []
                row.append(time.strftime("%Y-%m-%d %H:%M:%S"))
                for k in keys:
                    v = t[k]
                    if isinstance(v, np.ndarray):
                        row.append(np.array2string(v, separator=',', max_line_width=10000))
                    else:
                        row.append(str(v))
                writer.writerow(row)


    def clear(self):
        self.transitions = []

    def get_logs(self):
        return self.transitions
    

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