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
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml

from utilities.Settings import Settings


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
    CONTROL_ACTION_DIM = 2

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
    act_space = spaces.Box(
        low=np.array([-1, -1], dtype=np.float32),
        high=np.array([1, 1], dtype=np.float32),
        dtype=np.float32,
    )

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
    REWARD_COMPONENT_KEYS = (
        "progress",
        "crash_reward",
        "wp_distance_penalty",
        "d_action_penality",
        "speed_cap_penalty",
        "proximity_penalty",
        "stuck_reward",
        "spin_reward",
    )

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
        self.reward_components_stats_path = os.path.join(
            self.obs_tracking_dir, "reward_components_stats.csv"
        )
        self.reward_components_hist_csv_path = os.path.join(
            self.obs_tracking_dir, "reward_components_histogram.csv"
        )
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
        self._rew_reservoir: Optional[np.ndarray] = None
        self._comp_seen = 0
        self._comp_stats: Dict[str, Dict[str, float]] = {}
        self._comp_reservoir: Dict[str, np.ndarray] = {}
        self._comp_reservoir_fill = 0

    def _maybe_init(self, obs_dim: int) -> None:
        if self._obs_dim is not None:
            return
        self._obs_dim = int(obs_dim)
        self._obs_mean = np.zeros(self._obs_dim, dtype=np.float64)
        self._obs_m2 = np.zeros(self._obs_dim, dtype=np.float64)
        self._obs_min = np.full(self._obs_dim, np.inf, dtype=np.float64)
        self._obs_max = np.full(self._obs_dim, -np.inf, dtype=np.float64)
        self._obs_reservoir = np.zeros((self.hist_sample_cap, self._obs_dim), dtype=np.float32)

    def _ensure_comp_reservoir(self) -> None:
        if self._comp_reservoir:
            return
        self._rew_reservoir = np.zeros(self.hist_sample_cap, dtype=np.float32)
        for key in self.REWARD_COMPONENT_KEYS:
            self._comp_reservoir[key] = np.zeros(self.hist_sample_cap, dtype=np.float32)
            self._comp_stats[key] = {
                "seen": 0,
                "mean": 0.0,
                "m2": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "mean_abs": 0.0,
            }

    def _reservoir_index(self) -> Optional[int]:
        if self._comp_reservoir_fill < self.hist_sample_cap:
            idx = self._comp_reservoir_fill
            self._comp_reservoir_fill += 1
            return idx
        j = np.random.randint(0, self._rew_seen)
        if j < self.hist_sample_cap:
            return j
        return None

    def _store_aligned_sample(self, reward: float, components: Dict[str, float]) -> None:
        self._ensure_comp_reservoir()
        idx = self._reservoir_index()
        if idx is None or self._rew_reservoir is None:
            return
        self._rew_reservoir[idx] = float(reward)
        for key in self.REWARD_COMPONENT_KEYS:
            self._comp_reservoir[key][idx] = float(components.get(key, 0.0))

    def _track_component(self, key: str, value: float) -> None:
        stats = self._comp_stats[key]
        stats["seen"] += 1
        n = stats["seen"]
        delta = value - stats["mean"]
        stats["mean"] += delta / n
        delta2 = value - stats["mean"]
        stats["m2"] += delta * delta2
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        abs_delta = abs(value) - stats["mean_abs"]
        stats["mean_abs"] += abs_delta / n

    def track(
        self,
        obs: np.ndarray,
        reward: float,
        reward_components: Optional[Dict[str, float]] = None,
    ) -> None:
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

        if isinstance(reward_components, dict) and reward_components:
            self._ensure_comp_reservoir()
            self._comp_seen += 1
            for key in self.REWARD_COMPONENT_KEYS:
                comp_val = float(reward_components.get(key, 0.0))
                self._track_component(key, comp_val)
            self._store_aligned_sample(r, reward_components)
        else:
            if len(self._rew_samples) < self.hist_sample_cap:
                self._rew_samples.append(r)
            else:
                j = np.random.randint(0, self._rew_seen)
                if j < self.hist_sample_cap:
                    self._rew_samples[j] = r

    def _component_accumulated(self, key: str) -> float:
        stats = self._comp_stats.get(key)
        if not stats or stats["seen"] <= 0:
            return 0.0
        return float(stats["mean"]) * float(stats["seen"])

    def _write_reward_component_stats(self) -> None:
        if self._comp_seen <= 0:
            return
        rows = []
        mean_abs_vals = []
        for key in self.REWARD_COMPONENT_KEYS:
            stats = self._comp_stats.get(key)
            if not stats or stats["seen"] <= 0:
                continue
            var = stats["m2"] / max(1, stats["seen"] - 1)
            std = float(np.sqrt(max(var, 0.0)))
            mean_abs = float(stats["mean_abs"])
            mean_abs_vals.append(mean_abs)
            accumulated = self._component_accumulated(key)
            rows.append(
                {
                    "component": key,
                    "count": stats["seen"],
                    "accumulated": accumulated,
                    "mean": float(stats["mean"]),
                    "std": std,
                    "min": float(stats["min"]),
                    "max": float(stats["max"]),
                    "mean_abs": mean_abs,
                }
            )
        total_abs = float(sum(mean_abs_vals)) or 1.0
        for row in rows:
            row["rel_share"] = row["mean_abs"] / total_abs
        with open(self.reward_components_stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "component",
                    "count",
                    "accumulated",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "mean_abs",
                    "rel_share",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_reward_component_histogram_csv(self) -> None:
        if self._comp_seen <= 0:
            return
        with open(self.reward_components_hist_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "accumulated", "count"])
            for key in self.REWARD_COMPONENT_KEYS:
                stats = self._comp_stats.get(key)
                if not stats or stats["seen"] <= 0:
                    continue
                writer.writerow([key, self._component_accumulated(key), stats["seen"]])

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

        self._write_reward_component_stats()

        if len(self._rew_samples) > 0 or self._comp_reservoir_fill > 0:
            has_components = self._comp_reservoir_fill > 0 and bool(self._comp_reservoir)
            if has_components and self._rew_reservoir is not None:
                n = int(self._comp_reservoir_fill)
                rew_arr = self._rew_reservoir[:n].astype(np.float32)
            elif len(self._rew_samples) > 0:
                rew_arr = np.asarray(self._rew_samples, dtype=np.float32)
                has_components = False
            else:
                rew_arr = np.array([], dtype=np.float32)

            if rew_arr.size > 0:
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

                if self._comp_seen > 0:
                    self._write_reward_component_histogram_csv()

                if render_png and self._comp_seen <= 0 and rew_arr.size > 0:
                    try:
                        centers = 0.5 * (r_edges[:-1] + r_edges[1:])
                        widths = (r_edges[1:] - r_edges[:-1]) * 0.95
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.bar(
                            centers,
                            r_counts,
                            width=widths,
                            align="center",
                            color="#4C78A8",
                            edgecolor="black",
                            linewidth=0.3,
                        )
                        ax.set_title("Reward Histogram (total only; no components in stream)")
                        ax.set_xlabel("Reward")
                        ax.set_ylabel("Count")
                        ax.grid(axis="y", alpha=0.25)
                        plt.tight_layout()
                        fig.savefig(
                            os.path.join(self.obs_tracking_dir, "reward_histogram.png"),
                            dpi=160,
                        )
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
            "reward_components_seen": int(self._comp_seen),
            "reward_hist_sample_count": int(
                self._comp_reservoir_fill if self._comp_reservoir_fill > 0 else len(self._rew_samples)
            ),
            "reward_min": float(self._rew_min if self._rew_seen > 0 else 0.0),
            "reward_max": float(self._rew_max if self._rew_seen > 0 else 0.0),
            "reward_mean": float(self._rew_mean if self._rew_seen > 0 else 0.0),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._obs_last_flush_seen = self._obs_seen



class IngestStatsTracker:
    """Append-only log of streamed transition batches (updates before SAC training runs)."""

    HEADER = [
        "time",
        "batch_size",
        "batch_mean_reward",
        "batch_min_reward",
        "batch_max_reward",
        "batch_done_count",
        "episodes_completed_total",
        "transitions_total",
        "replay_buffer_size",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._start = time.time()
        self._episodes_completed = 0
        self._transitions_total = 0
        self._batch_count = 0
        target_dir = os.path.dirname(csv_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        if not os.path.isfile(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def record_batch(
        self,
        batch: List[dict],
        *,
        episodes_completed: int,
        replay_buffer_size: int,
    ) -> None:
        if not batch:
            return

        rewards = [float(t.get("reward", 0.0)) for t in batch]
        done_count = sum(1 for t in batch if bool(t.get("done")))
        self._batch_count += 1
        self._episodes_completed += int(episodes_completed)
        self._transitions_total += len(batch)

        row = {
            "time": round(time.time() - self._start, 3),
            "batch_size": len(batch),
            "batch_mean_reward": float(np.mean(rewards)),
            "batch_min_reward": float(np.min(rewards)),
            "batch_max_reward": float(np.max(rewards)),
            "batch_done_count": int(done_count),
            "episodes_completed_total": int(self._episodes_completed),
            "transitions_total": int(self._transitions_total),
            "replay_buffer_size": int(replay_buffer_size),
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.HEADER).writerow(row)

        log_every = max(1, int(getattr(Settings, "LEARNER_INGEST_LOG_EVERY_BATCHES", 5)))
        if Settings.LEARNER_SERVER_DEBUG or (self._batch_count % log_every) == 0:
            print(
                "[server] Ingest batch "
                f"size={row['batch_size']} mean_r={row['batch_mean_reward']:.3f} "
                f"done={row['batch_done_count']} replay={row['replay_buffer_size']} "
                f"episodes_total={row['episodes_completed_total']}"
            )


class EpisodeLogTracker:
    """Append-only log of completed training episodes (one row per episode)."""

    COMPONENT_KEYS = ObsRewardTracker.REWARD_COMPONENT_KEYS

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._next_episode_index = 0
        target_dir = os.path.dirname(csv_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        if os.path.isfile(csv_path):
            self._resume_from_csv()
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames()).writeheader()

    def _fieldnames(self) -> List[str]:
        return [
            "episode_index",
            "timestamp",
            "time_s",
            "actor_id",
            "episode_id",
            "length",
            "total_reward",
            "mean_reward",
            "total_timesteps",
            "lap_times",
            "reward_difficulty",
            "difficulty",
            *[f"comp_{key}" for key in self.COMPONENT_KEYS],
        ]

    def _resume_from_csv(self) -> None:
        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                self._next_episode_index = int(rows[-1].get("episode_index", -1)) + 1
        except Exception as exc:
            print(f"[EpisodeLogTracker] Could not resume from {self.csv_path}: {exc}")
            self._next_episode_index = 0

    @staticmethod
    def _extract_lap_times(episode: List[dict]) -> List[float]:
        for transition in reversed(episode):
            info = transition.get("info", {}) or {}
            candidate = info.get("lap_times")
            if candidate is None:
                continue
            if isinstance(candidate, (list, tuple)) and len(candidate) == 0:
                continue
            if isinstance(candidate, (list, tuple)):
                out: List[float] = []
                for item in candidate:
                    try:
                        out.append(float(item))
                    except (TypeError, ValueError):
                        continue
                return out
        return []

    @staticmethod
    def _summarize_reward_components(episode: List[dict]) -> Dict[str, float]:
        totals = {key: 0.0 for key in EpisodeLogTracker.COMPONENT_KEYS}
        for transition in episode:
            info = transition.get("info", {}) or {}
            components = info.get("reward_components")
            if not isinstance(components, dict):
                continue
            for key in EpisodeLogTracker.COMPONENT_KEYS:
                try:
                    totals[key] += float(components.get(key, 0.0))
                except (TypeError, ValueError):
                    continue
        return totals

    def record_episode(
        self,
        episode: List[dict],
        *,
        total_timesteps: int,
        training_time_s: float,
    ) -> None:
        if not episode:
            return

        rewards = [float(t.get("reward", 0.0)) for t in episode]
        total_reward = float(sum(rewards))
        mean_reward = float(total_reward / max(1, len(rewards)))
        actor_id = int(episode[0].get("actor_id", 0))
        episode_id = int(episode[0].get("episode_id", 0))
        last_info = episode[-1].get("info", {}) or {}
        component_totals = self._summarize_reward_components(episode)
        lap_times = self._extract_lap_times(episode)

        row = {
            "episode_index": self._next_episode_index,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_s": round(float(training_time_s), 3),
            "actor_id": actor_id,
            "episode_id": episode_id,
            "length": len(episode),
            "total_reward": total_reward,
            "mean_reward": mean_reward,
            "total_timesteps": int(total_timesteps),
            "lap_times": json.dumps(lap_times),
            "reward_difficulty": last_info.get("reward_difficulty"),
            "difficulty": last_info.get("difficulty"),
        }
        for key in self.COMPONENT_KEYS:
            row[f"comp_{key}"] = float(component_totals.get(key, 0.0))

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames()).writerow(row)
        self._next_episode_index += 1


class TrainingLogHelper():
    def __init__(self, model_name: str, model_dir: str):
        self.model_name = model_name
        self.model_dir = model_dir
        self.csv_path = os.path.join(model_dir, f"learning_metrics.csv")
        self.start_time = self.initialize_start_time() # Training time start

        self.training_index = 0
        self.plot_every: Union[int, str] = self._parse_plot_every_setting()
        self._metrics_plot_interval_s = float(
            getattr(Settings, "SAC_METRICS_PLOT_INTERVAL_S", 0.0)
        )
        self._last_metrics_plot_time = 0.0
        self._final_metrics_png_done = False


    
    @staticmethod
    def _parse_plot_every_setting() -> Union[int, str]:
        raw = getattr(Settings, "SAC_METRICS_PLOT_EVERY", 5)
        if isinstance(raw, str) and raw.strip().lower() == "end":
            return "end"
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 5

    def maybe_plot_training_metrics_periodic(self) -> None:
        if self._metrics_plot_interval_s > 0:
            now = time.time()
            if (now - self._last_metrics_plot_time) < self._metrics_plot_interval_s:
                return
            self._last_metrics_plot_time = now
            self._plot_training_metrics_safe(final=False)
            return

        if self.plot_every == "end":
            return
        if self.training_index % int(self.plot_every) != 0:
            return
        self._plot_training_metrics_safe(final=False)

    def maybe_plot_training_metrics_final(self) -> None:
        """Always refresh training_metrics.png once when training stops."""
        if self._final_metrics_png_done:
            return
        self._final_metrics_png_done = True
        self._plot_training_metrics_safe(final=True)

    def _plot_training_metrics_safe(self, *, final: bool = False) -> None:
        try:
            png_path = self.plot_training_metrics()
            if png_path:
                tag = "final " if final else ""
                print(f"[TrainingLogHelper] Wrote {tag}training metrics plot: {png_path}")
        except Exception as e:
            print(f"[TrainingLogHelper] Failed to plot training metrics: {e}")

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

        # Ignore very short completed episodes (e.g. immediate crash) for lap-time stats.
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
        stream_batch_sizes = log_dict.get("stream_batch_sizes", [])

        current_time = time.time()
        training_time = current_time - self.start_time

        metric_dict['time'] = training_time
        metric_dict['episode_lengths'] = episode_lengths
        metric_dict['episode_rewards'] = episode_rewards
        metric_dict['episode_mean_step_rewards'] = episode_mean_step_rewards
        metric_dict['stream_batch_sizes'] = stream_batch_sizes
        # Keep a consistent, parsable representation so `plot_training_metrics()`
        # can always detect `lap_times` as an array-like column.
        metric_dict['lap_times'] = str(lap_times) if lap_times is not None else "[]"
        metric_dict['reward_difficulty'] = last_info.get("reward_difficulty", None)
        metric_dict['difficulty'] = last_info.get("difficulty", None)  # curriculum difficulty
        
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

        self.maybe_plot_training_metrics_periodic()

    def plot_training_metrics(self) -> Optional[str]:
        model_dir = self.model_dir
        csv_path=self.csv_path

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return None

        # Load the CSV
        df = pd.read_csv(csv_path)


        # Downsample and window settings
        downsample_step = 1


        # Downsample
        df_plot = df.iloc[::downsample_step].copy()
        if "time" in df_plot.columns:
            x_vals = df_plot["time"].values
            x_label = "time (s)"
        elif "total_timesteps" in df_plot.columns:
            x_vals = df_plot["total_timesteps"].values
            x_label = "total_timesteps"
        else:
            x_vals = np.arange(len(df_plot))
            x_label = "log_index"

        # Remove non-plotted metadata / timing fields (still kept in CSV).
        _skip_plot_cols = {
            "timestamp",
            "time",
            "training_duration",
            "post_process_duration",
            "batch_size",
            "gradient_steps",
            "learning_rate",
        }
        columns_to_plot = [
            col for col in df_plot.columns if col not in _skip_plot_cols
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
            if col in [
                'episode_lengths',
                'episode_rewards',
                'episode_mean_step_rewards',
                'stream_batch_sizes',
            ]:
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
        for ax in axs:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(max_xticks))
            ax.xaxis.set_tick_params(labelbottom=True)
        axs[-1].set_xlabel(x_label)

        plt.tight_layout()
        png_path = os.path.join(model_dir, 'training_metrics.png')
        plt.savefig(png_path)
        plt.close(fig)
        return png_path




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