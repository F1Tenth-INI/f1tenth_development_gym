import time
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import io
import torch
from stable_baselines3 import SAC
import os
import sys
import csv
import json
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
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
        # [0.]*60 +
        [-1] * 40 + 
        [-1]*6 +
        [-1]*2 
        + [0] * 2
        ,dtype=np.float32)
    
    obs_high = np.array(
        [ 1,  1,  1,  1] +
        [ 1]*30 + 
        # [1]*40 + 
        # [1.0]*60 +
        [ 1] * 40 +
        [ 1]*6 + 
        [ 1]*2
        + [0] * 2
        ,dtype=np.float32)
    obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    act_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32), high=np.array([ 1,  1], dtype=np.float32), dtype=np.float32)

    @staticmethod
    def make_obs_space_from_dim(obs_dim: int, low: float = -1.0, high: float = 1.0) -> spaces.Box:
        """
        Create a Box observation space with the given dimension.
        We mainly need this for SB3 to build the policy networks.
        """
        obs_dim = int(obs_dim)
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
        return spaces.Box(
            low=np.full((obs_dim,), low, dtype=np.float32),
            high=np.full((obs_dim,), high, dtype=np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def create_vec_env_from_obs_space(obs_space: spaces.Box):
        return DummyVecEnv([lambda: _SpacesOnlyEnv(obs_space, SacUtilities.act_space)])

    @staticmethod
    def create_vec_env_from_obs_dim(obs_dim: int):
        obs_space = SacUtilities.make_obs_space_from_dim(obs_dim)
        return SacUtilities.create_vec_env_from_obs_space(obs_space)

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
        # policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU)

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
            os.path.join(rl_racing_dir, "RewardCalculator.py"),
        ]

        zip_path = os.path.join(model_dir, "training_files.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in paths:
                if os.path.isfile(file_path):
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname=arcname)
        return zip_path
class ObsRewardTracker:
    def __init__(
        self,
        model_dir: str,
        enabled: bool = True,
        flush_every: int = 10_000,
        hist_bins: int = 40,
        hist_sample_cap: int = 20_000,
        hist_max_dims: int = 256,
    ):
        self.enabled = bool(enabled)
        self.flush_every = int(flush_every)
        self.hist_bins = int(hist_bins)
        self.hist_sample_cap = int(hist_sample_cap)
        self.hist_max_dims = int(hist_max_dims)

        self.obs_tracking_dir = os.path.join(model_dir, "obs_tracking")
        os.makedirs(self.obs_tracking_dir, exist_ok=True)
        self.obs_stats_path = os.path.join(self.obs_tracking_dir, "obs_stats.csv")
        self.obs_hist_npz_path = os.path.join(self.obs_tracking_dir, "obs_histograms.npz")
        self.obs_hist_png_path = os.path.join(self.obs_tracking_dir, "obs_histograms.png")
        self.reward_stats_path = os.path.join(self.obs_tracking_dir, "reward_stats.csv")
        self.reward_hist_csv_path = os.path.join(self.obs_tracking_dir, "reward_histogram.csv")
        self.reward_hist_png_path = os.path.join(self.obs_tracking_dir, "reward_histogram.png")
        self.summary_path = os.path.join(self.obs_tracking_dir, "tracker_summary.json")

        self._obs_seen = 0
        self._obs_last_flush_seen = 0
        self._obs_dim: Optional[int] = None
        self._obs_mean: Optional[np.ndarray] = None
        self._obs_m2: Optional[np.ndarray] = None
        self._obs_min: Optional[np.ndarray] = None
        self._obs_max: Optional[np.ndarray] = None
        self._obs_reservoir: Optional[np.ndarray] = None
        self._obs_reservoir_fill = 0

        self._rew_seen = 0
        self._rew_mean = 0.0
        self._rew_m2 = 0.0
        self._rew_min = float("inf")
        self._rew_max = float("-inf")
        self._rew_samples: list[float] = []

    def _maybe_init(self, obs_dim: int) -> None:
        if self._obs_dim is not None:
            return
        self._obs_dim = int(obs_dim)
        self._obs_mean = np.zeros(self._obs_dim, dtype=np.float64)
        self._obs_m2 = np.zeros(self._obs_dim, dtype=np.float64)
        self._obs_min = np.full(self._obs_dim, np.inf, dtype=np.float64)
        self._obs_max = np.full(self._obs_dim, -np.inf, dtype=np.float64)
        self._obs_reservoir = np.zeros((self.hist_sample_cap, self._obs_dim), dtype=np.float32)

    def track(self, obs: np.ndarray, reward: float) -> None:
        if not self.enabled:
            return
        flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if flat_obs.size == 0:
            return
        self._maybe_init(flat_obs.shape[0])
        if self._obs_dim is None or self._obs_mean is None or self._obs_m2 is None:
            return
        if flat_obs.shape[0] != self._obs_dim:
            return

        self._obs_seen += 1
        obs_f64 = flat_obs.astype(np.float64, copy=False)
        delta = obs_f64 - self._obs_mean
        self._obs_mean += delta / self._obs_seen
        delta2 = obs_f64 - self._obs_mean
        self._obs_m2 += delta * delta2
        self._obs_min = np.minimum(self._obs_min, obs_f64)
        self._obs_max = np.maximum(self._obs_max, obs_f64)

        if self._obs_reservoir is not None:
            if self._obs_reservoir_fill < self.hist_sample_cap:
                self._obs_reservoir[self._obs_reservoir_fill, :] = flat_obs
                self._obs_reservoir_fill += 1
            else:
                j = np.random.randint(0, self._obs_seen)
                if j < self.hist_sample_cap:
                    self._obs_reservoir[j, :] = flat_obs

        r = float(reward)
        self._rew_seen += 1
        rew_delta = r - self._rew_mean
        self._rew_mean += rew_delta / self._rew_seen
        rew_delta2 = r - self._rew_mean
        self._rew_m2 += rew_delta * rew_delta2
        self._rew_min = min(self._rew_min, r)
        self._rew_max = max(self._rew_max, r)
        if len(self._rew_samples) < self.hist_sample_cap:
            self._rew_samples.append(r)
        else:
            j = np.random.randint(0, self._rew_seen)
            if j < self.hist_sample_cap:
                self._rew_samples[j] = r

    def should_flush(self) -> bool:
        if not self.enabled:
            return False
        return (self._obs_seen - self._obs_last_flush_seen) >= self.flush_every

    def flush(self, render_png: bool = False) -> None:
        if not self.enabled:
            return
        if self._obs_dim is None or self._obs_seen <= 1:
            return
        if self._obs_mean is None or self._obs_m2 is None or self._obs_min is None or self._obs_max is None:
            return

        obs_var = self._obs_m2 / max(1, self._obs_seen - 1)
        obs_std = np.sqrt(np.maximum(obs_var, 0.0))
        with open(self.obs_stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["obs_idx", "count", "mean", "std", "min", "max"])
            for i in range(self._obs_dim):
                writer.writerow([i, self._obs_seen, float(self._obs_mean[i]), float(obs_std[i]), float(self._obs_min[i]), float(self._obs_max[i])])

        if self._obs_reservoir is not None and self._obs_reservoir_fill > 0:
            sample = self._obs_reservoir[: self._obs_reservoir_fill, :]
            dim_limit = min(self._obs_dim, self.hist_max_dims)
            hist_payload = {}
            for i in range(dim_limit):
                col = sample[:, i]
                col_min = float(np.min(col))
                col_max = float(np.max(col))
                if col_max <= col_min:
                    col_max = col_min + 1e-6
                counts, edges = np.histogram(col, bins=self.hist_bins, range=(col_min, col_max))
                hist_payload[f"obs_{i}_counts"] = counts.astype(np.int64)
                hist_payload[f"obs_{i}_edges"] = edges.astype(np.float32)
            np.savez_compressed(self.obs_hist_npz_path, **hist_payload)

            if render_png:
                try:
                    n_cols = int(np.ceil(np.sqrt(dim_limit)))
                    n_rows = int(np.ceil(dim_limit / max(1, n_cols)))
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False, sharey=False)
                    for i in range(dim_limit):
                        ax = axes[i // n_cols][i % n_cols]
                        counts = hist_payload[f"obs_{i}_counts"]
                        edges = hist_payload[f"obs_{i}_edges"]
                        centers = 0.5 * (edges[:-1] + edges[1:])
                        widths = (edges[1:] - edges[:-1]) * 0.95
                        ax.bar(centers, counts, width=widths, align="center", color="#72B7B2", edgecolor="black", linewidth=0.2)
                        ax.set_title(f"obs[{i}]", fontsize=8)
                        ax.tick_params(axis="both", labelsize=7)
                    for j in range(dim_limit, n_rows * n_cols):
                        axes[j // n_cols][j % n_cols].axis("off")
                    fig.suptitle("Observation Histograms", fontsize=12)
                    plt.tight_layout(rect=[0, 0, 1, 0.98])
                    fig.savefig(self.obs_hist_png_path, dpi=140)
                    plt.close(fig)
                except Exception as e:
                    print(f"[tracker] Failed to render observation histograms PNG: {e}")

        rew_var = self._rew_m2 / max(1, self._rew_seen - 1) if self._rew_seen > 1 else 0.0
        rew_std = float(np.sqrt(max(rew_var, 0.0)))
        with open(self.reward_stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["count", "mean", "std", "min", "max"])
            writer.writerow([self._rew_seen, float(self._rew_mean), rew_std, float(self._rew_min if self._rew_seen > 0 else 0.0), float(self._rew_max if self._rew_seen > 0 else 0.0)])

        if len(self._rew_samples) > 0:
            rew_arr = np.asarray(self._rew_samples, dtype=np.float32)
            r_min = float(np.min(rew_arr))
            r_max = float(np.max(rew_arr))
            if r_max <= r_min:
                r_max = r_min + 1e-6
            r_counts, r_edges = np.histogram(rew_arr, bins=self.hist_bins, range=(r_min, r_max))
            with open(self.reward_hist_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["bin_left", "bin_right", "count"])
                for i in range(len(r_counts)):
                    writer.writerow([float(r_edges[i]), float(r_edges[i + 1]), int(r_counts[i])])

            if render_png:
                try:
                    centers = 0.5 * (r_edges[:-1] + r_edges[1:])
                    widths = (r_edges[1:] - r_edges[:-1]) * 0.95
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(centers, r_counts, width=widths, align="center", color="#4C78A8", edgecolor="black", linewidth=0.3)
                    ax.set_title("Reward Histogram")
                    ax.set_xlabel("Reward")
                    ax.set_ylabel("Count")
                    ax.grid(axis="y", alpha=0.25)
                    plt.tight_layout()
                    fig.savefig(self.reward_hist_png_path, dpi=160)
                    plt.close(fig)
                except Exception as e:
                    print(f"[tracker] Failed to render reward histogram PNG: {e}")

        summary = {
            "obs_seen": int(self._obs_seen),
            "obs_dim": int(self._obs_dim),
            "obs_hist_sample_count": int(self._obs_reservoir_fill),
            "obs_hist_bins": int(self.hist_bins),
            "obs_hist_max_dims": int(self.hist_max_dims),
            "reward_seen": int(self._rew_seen),
            "reward_min": float(self._rew_min if self._rew_seen > 0 else 0.0),
            "reward_max": float(self._rew_max if self._rew_seen > 0 else 0.0),
            "reward_mean": float(self._rew_mean if self._rew_seen > 0 else 0.0),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._obs_last_flush_seen = self._obs_seen



class TrainingLogHelper():
    def __init__(self, model_name: str, model_dir: str):
        self.model_name = model_name
        self.model_dir = model_dir
        self.csv_path = os.path.join(model_dir, f"learning_metrics.csv")
        self.start_time = self.initialize_start_time() # Training time start

        self.training_index = 0
        self.plot_every = 1  # Plot every log (was 5; 1 ensures metrics appear quickly)


    
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

        # Many actors send "bootstrap" transition batches before an episode ends
        # by sending only the first couple transitions (length ~= 2). Those batches
        # tend to have empty lap_times and near-zero accumulated rewards.
        #
        # For metrics we therefore ignore very short batches and only compute
        # lap_times / episode stats from episodes with at least `min_transitions`.
        min_transitions = 3
        metric_episodes = [ep for ep in episodes if ep and len(ep) >= min_transitions]
        if not metric_episodes:
            # Fallback: if everything is very short, log whatever we got.
            metric_episodes = episodes

        # Find the most recent non-empty lap_times from the metric episodes.
        lap_times = None
        last_info = {}
        for ep in reversed(metric_episodes):
            if not ep:
                continue
            # Track info from the last transition in the episode as a source of other keys.
            if "info" in ep[-1]:
                last_info = ep[-1]["info"] or {}

            for t in reversed(ep):
                info = t.get("info", {}) or {}
                candidate = info.get("lap_times", None)
                if candidate is None:
                    continue
                # Treat empty lists as "no lap times".
                if isinstance(candidate, (list, tuple)) and len(candidate) == 0:
                    continue
                # Found a meaningful lap_times value.
                lap_times = candidate
                break
            if lap_times is not None:
                break

        episode_lengths = [len(episode) for episode in metric_episodes]
        episode_rewards = [
            sum(transition["reward"] for transition in episode) for episode in metric_episodes
        ]
        episode_mean_step_rewards = [
            np.mean([transition["reward"] for transition in episode]) if len(episode) > 0 else 0.0
            for episode in metric_episodes
        ]

        current_time = time.time()
        training_time = current_time - self.start_time

        metric_dict['time'] = training_time
        metric_dict['episode_lengths'] = episode_lengths
        metric_dict['episode_rewards'] = episode_rewards
        metric_dict['episode_mean_step_rewards'] = episode_mean_step_rewards
        # Keep a consistent, parsable representation so `plot_training_metrics()`
        # can always detect `lap_times` as an array-like column.
        metric_dict['lap_times'] = str(lap_times) if lap_times is not None else "[]"
        metric_dict['reward_difficulty'] = last_info.get("reward_difficulty", None)
        metric_dict['difficulty'] = last_info.get("difficulty", None)  # curriculum difficulty
        
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
            try:
                self.plot_training_metrics()
            except Exception as e:
                print(f"[TrainingLogHelper] Failed to plot training metrics: {e}")

    def plot_training_metrics(self):
        model_dir = self.model_dir
        csv_path=self.csv_path

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        # Load the CSV
        df = pd.read_csv(csv_path)


        # Downsample and window settings
        downsample_step = 1


        # Downsample
        df_plot = df.iloc[::downsample_step].copy()
        x_vals = df_plot['time'].values  # Use .values to get numpy array for proper indexing

        # Remove non-timeseries metadata fields from plotting.
        columns_to_plot = [
            col
            for col in df_plot.columns
            if col not in ['timestamp', 'training_duration', 'batch_size', 'gradient_steps', 'learning_rate']
        ]


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
            
            # Force certain columns to be treated as array-like
            if col in ['episode_lengths', 'episode_rewards', 'episode_mean_step_rewards']:
                is_array_like = True
            elif isinstance(first_val, (list, tuple)):
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
        png_path = os.path.join(model_dir, 'training_metrics.png')
        plt.savefig(png_path)
        plt.close(fig)




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