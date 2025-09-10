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
import yaml


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
    obs_low  = np.array(
        [-1, -1, -1, -1] +
        [-1]*30 +
        # [0]*40 + 
        [0.]*60 +
        [-1]*6 +
        [-1]*2, dtype=np.float32)
    
    obs_high = np.array(
        [ 1,  1,  1,  1] +
        [ 1]*30 + 
        # [1]*40 + 
        [1.0]*60 +
        [ 1]*6 + 
        [ 1]*2, dtype=np.float32)
    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    act_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32), high=np.array([ 1,  1], dtype=np.float32), dtype=np.float32)

    @staticmethod
    def make_env():
        return _SpacesOnlyEnv(SacUtilities.obs_space, SacUtilities.act_space)

    @staticmethod
    def create_vec_env():
        return DummyVecEnv([SacUtilities.make_env])

    @staticmethod
    def create_model(env, 
                     buffer_size=100_000, 
                     device="cpu",
                     learning_rate=3e-4,
                     discount_factor=0.99,
                     batch_size=256,
                     train_freq=1
                     ) -> SAC:
        policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh)
        #  log_std_init=-3.5

        model = SAC(
                    "MlpPolicy",
                    env=env,
                    verbose=0,
                    # ent_coef=0.01,
                    train_freq=train_freq,
                    gamma=discount_factor,
                    learning_rate=learning_rate,
                    policy_kwargs=policy_kwargs,
                    buffer_size=buffer_size,
                    device=device,
                    batch_size=batch_size,
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
    
    @staticmethod
    def zip_relevant_files(model_dir: str) -> str:
        import zipfile
        this_files_path = os.path.abspath(__file__)
        rl_racing_dir = os.path.dirname(this_files_path)
        gym_dir = os.path.join(rl_racing_dir, "..", "..")
        paths = [
            os.path.join(rl_racing_dir, "learner_server.py"),
            os.path.join(rl_racing_dir, "sac_utilities.py"),
            os.path.join(rl_racing_dir, "sac_agent_planner.py"),
            os.path.join(gym_dir, "utilities", "Settings.py"),
        ]

        zip_path = os.path.join(model_dir, "training_files.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in paths:
                if os.path.isfile(file_path):
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname=arcname)
        return zip_path


    

class TrainingLogHelper():
    def __init__(self, model_name: str, model_dir: str):
        self.model_name = model_name
        self.model_dir = model_dir
        self.csv_path = os.path.join(model_dir, f"learning_metrics.csv")
        self.start_time = self.initialize_start_time() # Training time start

        self.training_index = 0
        self.plot_every = 5


    
    def initialize_start_time(self):
        # Initialize start time, possibly resuming from existing CSV

        self.start_time = time.time()
        # check if csv exists. if exists, load it and read last time
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                if not df.empty and 'time' in df.columns:
                    print(f"Existing CSV file found: {self.csv_path}, resuming time from last entry. Start time was {self.start_time}")
                    last_time = df['time'].iloc[-1]
                    self.start_time -= last_time
            except Exception as e:
                print(f"Error reading existing CSV file: {e}")
        else:
            print(f"CSV file does not exist, starting fresh: {self.csv_path} with start time {self.start_time}")

        return self.start_time
    

    def save_meta_info(self, model: SAC, info: dict):
        info_file = os.path.join(self.model_dir, "info.yaml")

        info_dict = {
            "model_name": self.model_name,
            "batch_size": model.batch_size,
            "buffer_size": model.replay_buffer.buffer_size,
            "observation_space": model.observation_space.shape[0],
            "action_space": model.action_space.shape[0],
            "gamma": model.gamma,
            "tau": model.tau,
            "target_entropy": model.target_entropy,
            "policy_type": type(model.policy).__name__,
            "actor_arch": str(model.policy.actor),
            "critic_arch": str(model.policy.critic),
        }
        info_dict.update(info)

        with open(info_file, "w") as f:
            yaml.dump(info_dict, f, sort_keys=False)

    def load_meta_info(self) -> dict:
        info_file = os.path.join(self.model_path, "info.yaml")
        if not os.path.exists(info_file):
            return {}
        with open(info_file, "r") as f:
            return yaml.safe_load(f)


    def log_to_csv(self, model, episodes, log_dict):

        file_exists = os.path.isfile(self.csv_path)
        
        # Gather all available metrics from logger
        metric_dict = {}

        # Add lap_times, min_laptime, avg_laptime from the last episode's info if available
        last_episode = episodes[-1] if episodes else None
        last_info = last_episode[-1]["info"] if last_episode and last_episode[-1] and "info" in last_episode[-1] else {}
        lap_times = last_info.get("lap_times", None)
        min_laptime = last_info.get("min_laptime", None)

        episode_lengths = [len(episode) for episode in episodes]
        episode_rewards = [sum(transition["reward"] for transition in episode) for episode in episodes]
        episode_mean_step_rewards = [np.mean([transition["reward"] for transition in episode]) if len(episode)>0 else 0.0 for episode in episodes]

        current_time = time.time()
        training_time = current_time - self.start_time

        metric_dict['time'] = training_time
        metric_dict['episode_lengths'] = episode_lengths
        metric_dict['episode_rewards'] = episode_rewards
        metric_dict['episode_mean_step_rewards'] = episode_mean_step_rewards
        metric_dict['lap_times'] = str(lap_times) if lap_times is not None else ""
        metric_dict['total_timesteps'] = getattr(model, '_total_timesteps', None)
        metric_dict['training_duration'] = getattr(model, 'training_duration', None)
        

        metric_dict['replay_buffer_size'] = model.replay_buffer.size() if hasattr(model, 'replay_buffer') else None
        metric_dict['batch_size'] = getattr(model, 'batch_size', None)
        metric_dict['gradient_steps'] = getattr(model, 'gradient_steps', None)
        metric_dict['learning_rate'] = getattr(model, 'learning_rate', None) 
        
        metric_dict.update(log_dict)


        # if(min_laptime is not None):
        #     if min_laptime < 21.0:
        #         print("Training done.")
        #         exit()
        # if(metric_dict['total_timesteps'] > 2_000_000):
        #     print("Max timesteps reached.")
        #     exit()

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
        x_vals = df_plot['time']

        # Remove 'timestamp' from columns to plot
        columns_to_plot = [col for col in df_plot.columns if col not in ['timestamp','training_duration', 'replay_buffer_size', 'batch_size', 'gradient_steps', 'learning_rate']]


        n_cols = len(columns_to_plot)
        fig, axs = plt.subplots(n_cols, 1, figsize=(10, 3 * n_cols), sharex=True)

        if n_cols == 1:
            axs = [axs]

        for i, col in enumerate(columns_to_plot):
            # Check if the value of col is an array (list or tuple)
            import ast
            first_val = df_plot[col].iloc[0]
            # Try to detect string representations of lists/tuples for any column
            is_array_like = False
            arr_sample = None
            if isinstance(first_val, (list, tuple)):
                is_array_like = True
                arr_sample = first_val
            elif isinstance(first_val, str):
                try:
                    arr_sample = ast.literal_eval(first_val)
                    if isinstance(arr_sample, (list, tuple)):
                        is_array_like = True
                except Exception:
                    pass
            if is_array_like:
                all_vals = []
                all_x = []
                for idx, arr_str in enumerate(df_plot[col].values):
                    try:
                        arr = arr_str
                        if isinstance(arr, str):
                            arr = ast.literal_eval(arr)
                        if isinstance(arr, (list, tuple)):
                            all_vals.extend(arr)
                            all_x.extend([x_vals[idx]] * len(arr))
                    except Exception:
                        continue
                axs[i].scatter(all_x, all_vals, label=col, s=10)
            else:
                y_vals = df_plot[col].values
                if col in ['min_laptime', 'avg_laptime']:
                    axs[i].scatter(x_vals, y_vals, label=col, s=10)
                else:
                    axs[i].plot(x_vals, y_vals, label=col)
            axs[i].set_ylabel(col)
            axs[i].legend()
            axs[i].grid(True)

        import matplotlib.ticker as ticker
        max_xticks = 10
        for idx, ax in enumerate(axs):
            ax.xaxis.set_major_locator(ticker.MaxNLocator(max_xticks))
            ax.xaxis.set_tick_params(labelbottom=True)
            if idx == len(axs) - 1:
                ax.set_xlabel('timestamp')

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