#!/usr/bin/env python3
"""
Batch process plot_sample_frequency.py for multiple models and organize outputs.

Usage:
    python batch_plot_models.py --prefix "2602"
    python batch_plot_models.py --prefix "2602" --map-name RCA2
"""

import json
import os
import sys
import argparse
from pathlib import Path
import matplotlib
import torch


matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt

# Add root to path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

# Import the plotting functions
from plot_sample_frequency import (
    compute_sample_frequency,
    plot_sample_frequency_distribution_static,
    plot_spatial_heatmap_static,
    plot_reward_heatmap,
    plot_value_heatmap,
    load_map_image,
    world_to_pixel
)

import pandas as pd
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import DummyVecEnv
from TrainingLite.rl_racing.sac_utilities import SacUtilities


def compute_global_stats(csv_paths, min_possible_samples_percentile=5):
    """
    Load multiple CSVs and compute global statistics for consistent scaling.
    
    Computes stats on FILTERED data (bottom percentile removed) to match what's plotted.
    
    Returns:
        global_stats: dict with keys:
            - 'samples_per_batch_min/max': min/max across all models (filtered)
            - 'possible_samples_min/max': transition lifetime range (filtered)
            - 'sample_count_max': absolute sample count max (filtered)
            - 'num_transitions_min/max': range of transition counts per model (filtered)
            - 'reward_min/max': reward range across all transitions (filtered)
            - 'pose_x_min/max', 'pose_y_min/max': coordinate ranges for spatial plots (filtered)
    """
    all_dfs = []
    num_transitions_per_model = []
    
    print(f"  Applying {min_possible_samples_percentile}% possible_samples filtering to global stats...")
    
    for csv_path in csv_paths:
        try:
            df_raw = pd.read_csv(csv_path)
            before_count = len(df_raw)
            
            # Apply same filtering as compute_sample_frequency to match plotted data
            df = compute_sample_frequency(df_raw, csv_path=csv_path, 
                                        min_possible_samples_percentile=min_possible_samples_percentile)
            after_count = len(df)
            
            if before_count != after_count:
                print(f"    {csv_path.name}: {before_count} -> {after_count} transitions (filtered {before_count - after_count})")
            
            all_dfs.append(df)
            num_transitions_per_model.append(len(df))
        except Exception as e:
            print(f"  Warning: Failed to load {csv_path}: {e}")
    
    if not all_dfs:
        print("Error: No CSVs loaded successfully")
        return {}
    
    # Combine all data for global stats
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    global_stats = {
        'samples_per_batch_min': combined_df['samples_per_batch'].min(),
        'samples_per_batch_max': combined_df['samples_per_batch'].max(),
        'possible_samples_min': combined_df['possible_samples'].min(),
        'possible_samples_max': combined_df['possible_samples'].max(),
        'sample_count_max': combined_df['sample_count'].max(),
        'num_transitions_min': min(num_transitions_per_model),
        'num_transitions_max': max(num_transitions_per_model),
    }
    
    print(f"  After filtering, max sample_count in combined data: {global_stats['sample_count_max']:.0f}")
    
    # Reward range (if reward column exists)
    if 'reward' in combined_df.columns:
        df_with_reward = combined_df.dropna(subset=['reward'])
        if len(df_with_reward) > 0:
            global_stats['reward_min'] = df_with_reward['reward'].min()
            global_stats['reward_max'] = df_with_reward['reward'].max()
    
    # Spatial ranges (if position data exists)
    if 'pose_x' in combined_df.columns and 'pose_y' in combined_df.columns:
        df_with_pos = combined_df.dropna(subset=['pose_x', 'pose_y'])
        if len(df_with_pos) > 0:
            global_stats['pose_x_min'] = df_with_pos['pose_x'].min()
            global_stats['pose_x_max'] = df_with_pos['pose_x'].max()
            global_stats['pose_y_min'] = df_with_pos['pose_y'].min()
            global_stats['pose_y_max'] = df_with_pos['pose_y'].max()
    
    print(f"\n  Global statistics computed across {len(all_dfs)} models:")
    print(f"    Samples per batch: {global_stats['samples_per_batch_min']:.4f} - {global_stats['samples_per_batch_max']:.4f}")
    print(f"    Absolute sample count: 0 - {global_stats['sample_count_max']:.0f}")
    print(f"    Transition lifetime (possible samples): {global_stats['possible_samples_min']:.0f} - {global_stats['possible_samples_max']:.0f}")
    print(f"    Num transitions per model: {global_stats['num_transitions_min']} - {global_stats['num_transitions_max']}")
    if 'reward_min' in global_stats:
        print(f"    Reward range: {global_stats['reward_min']:.4f} - {global_stats['reward_max']:.4f}")
    if 'pose_x_min' in global_stats:
        print(f"    Pose X range: {global_stats['pose_x_min']:.2f} - {global_stats['pose_x_max']:.2f}")
        print(f"    Pose Y range: {global_stats['pose_y_min']:.2f} - {global_stats['pose_y_max']:.2f}")
    
    return global_stats


class BatchPlotter:
    """Batch process plotting for multiple models"""
    
    def __init__(self, prefix: str, map_name: str = 'RCA1', output_base_dir: str = None, plot_critic_output: bool = False):
        """
        Initialize batch plotter.
        
        Args:
            prefix: Model name prefix to filter (e.g., "2602")
            map_name: Map name for spatial heatmap
            output_base_dir: Base directory for organized outputs
            plot_critic_output: Whether to plot critic output
        """

        self.prefix = prefix
        self.map_name = map_name
        self.plot_critic_output = plot_critic_output
        
        self.models_dir = Path(root_dir) / "TrainingLite" / "rl_racing" / "models"
        
        if output_base_dir is None:
            output_base_dir = Path(root_dir) / "batch_plot_results" / f"batch_{prefix}"
        
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each plot type
        self.dist_dir = self.output_base_dir / "1_distribution_plots"
        self.heatmap_dir = self.output_base_dir / "2_spatial_heatmaps"
        self.reward_dir = self.output_base_dir / "3_reward_heatmaps"
        
        self.dist_dir.mkdir(exist_ok=True)
        self.heatmap_dir.mkdir(exist_ok=True)
        self.reward_dir.mkdir(exist_ok=True)
        
        print(f"Output directories created:")
        print(f"  Distribution plots: {self.dist_dir}")
        print(f"  Spatial heatmaps:   {self.heatmap_dir}")
        print(f"  Reward heatmaps:    {self.reward_dir}")
    
    def find_models_by_prefix(self):
        """Find all model directories matching the prefix"""
        if not self.models_dir.exists():
            print(f"Error: Models directory not found: {self.models_dir}")
            return []
        
        matching_models = []
        for model_dir in sorted(self.models_dir.iterdir()):
            if model_dir.is_dir() and model_dir.name.startswith(self.prefix):
                stats_csv = model_dir / "stat_logs" / "stats_log.csv"
                if stats_csv.exists():
                    matching_models.append(model_dir.name)
                else:
                    print(f"  Skipping {model_dir.name} (no stats_log.csv)")
        
        return matching_models
    
    def plot_model(self, model_name: str, global_stats: dict = None):
        """
        Generate all plots for a single model.
        
        Args:
            model_name: Name of the model directory
            global_stats: Global statistics for synchronized scaling across batch (optional)
        """
        print(f"\n{'='*80}")
        print(f"Processing: {model_name}")
        print(f"{'='*80}")
        
        # Load CSV
        csv_path = self.models_dir / model_name / "stat_logs" / "stats_log.csv"
        
        try:
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} transitions")
            
            # Compute sample frequency
            df = compute_sample_frequency(df, csv_path=str(csv_path))
            model_sample_count_max = float(df['sample_count'].max())
            print(f"  Filtered max absolute sample count: {model_sample_count_max:.0f}")
            
            # Plot 1: Sample frequency distribution
            print("  Generating distribution plot...")
            dist_save_path = self.dist_dir / f"{model_name}_distribution.png"
            try:
                plot_sample_frequency_distribution_static(df, save_path=str(dist_save_path), global_stats=global_stats)
                plt.close('all')  # Close figure to free memory
                print(f"    ✓ Saved to {dist_save_path.name}")
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            # Plot 2: Spatial heatmap
            print("  Generating spatial heatmap...")
            heatmap_save_path = self.heatmap_dir / f"{model_name}_spatial.png"
            try:
                # Filter out transitions without position data
                df_with_pos = df.dropna(subset=['pose_x', 'pose_y'])
                
                if len(df_with_pos) > 0:
                    # Load map
                    img_array, config = load_map_image(self.map_name)
                    origin = config['origin']
                    resolution = config['resolution']
                    
                    # Convert world coordinates to pixel coordinates
                    pixel_coords = []
                    for _, row in df_with_pos.iterrows():
                        px, py = world_to_pixel(row['pose_x'], row['pose_y'], origin, resolution)
                        pixel_coords.append((px, py))
                    
                    df_with_pos['pixel_x'] = [c[0] for c in pixel_coords]
                    df_with_pos['pixel_y'] = [c[1] for c in pixel_coords]
                    
                    plot_spatial_heatmap_static(
                        df_with_pos, img_array, config, self.map_name, 
                        save_path=str(heatmap_save_path),
                        global_stats=global_stats
                    )
                    plt.close('all')
                    print(f"    ✓ Saved to {heatmap_save_path.name}")
                else:
                    print(f"    ⊘ No position data available")
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            # Plot 3: Reward heatmap
            print("  Generating reward heatmap...")
            reward_save_path = self.reward_dir / f"{model_name}_reward.png"
            try:
                df_with_pos = df.dropna(subset=['pose_x', 'pose_y'])
                
                if len(df_with_pos) > 0 and 'reward' in df_with_pos.columns:
                    # Load map
                    img_array, config = load_map_image(self.map_name)
                    
                    # We need pixel coordinates - recompute if needed
                    if 'pixel_x' not in df_with_pos.columns:
                        origin = config['origin']
                        resolution = config['resolution']
                        pixel_coords = []
                        for _, row in df_with_pos.iterrows():
                            px, py = world_to_pixel(row['pose_x'], row['pose_y'], origin, resolution)
                            pixel_coords.append((px, py))
                        df_with_pos['pixel_x'] = [c[0] for c in pixel_coords]
                        df_with_pos['pixel_y'] = [c[1] for c in pixel_coords]
                    
                    # Create figure manually and save
                    plot_reward_heatmap(df_with_pos, img_array, self.map_name)
                    plt.savefig(str(reward_save_path), dpi=150, bbox_inches='tight')
                    plt.close('all')
                    print(f"    ✓ Saved to {reward_save_path.name}")
                else:
                    print(f"    ⊘ No reward data available")
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            print(f"  Done processing {model_name}")
            
        except Exception as e:
            print(f"  ✗ Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def create_critic_output_plot(self, model_name: str):

        print(f"\n{'='*80}")
        print(f"Calculating and plotting critic output: {model_name}")
        print(f"{'='*80}")

        csv_path = self.models_dir / model_name / "stat_logs" / "stats_log.csv"
        model_path = self.models_dir / model_name / f"{model_name}.zip"
        
        dummy_env = SacUtilities.create_vec_env()
        model = SAC.load(model_path, env=dummy_env, device="cpu")
        # actor = self.model.policy.actor
        critic = model.policy.critic

        try:
            df = pd.read_csv(csv_path)
            if 'obs' not in df.columns or 'action' not in df.columns:
                print(f"ERROR: CSV missing obs/action columns. Enable extended_obs_action_save.")
                return
            print(f"  Loaded {len(df)} transitions for critic output")
        except Exception as e:
            print(f"  ✗ Failed to load CSV for critic output: {e}")
            return
        
        try:
            print(f"  Parsing observations...")
            obs_list = [np.array(eval(x)) for x in df['obs']]
            # obs_list = [x for x in obs_list if x is not None]
            obs_np = np.vstack(obs_list).astype(np.float32)
            
            print(f"  Parsing actions...")
            act_list = [np.array(eval(x)) for x in df['action']]
            # act_list = [x for x in act_list if x is not None]
            act_np = np.vstack(act_list).astype(np.float32)
            
            print(f"  ✓ Parsed obs shape: {obs_np.shape}, action shape: {act_np.shape}")
        except Exception as e:
            print(f"  ✗ Failed to parse obs/action: {e}")
            import traceback
            traceback.print_exc()
            return
        # print(df['obs'])

        with torch.no_grad():
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device="cpu")
            act_t = torch.as_tensor(act_np, dtype=torch.float32, device="cpu")
            q1, q2 = critic(obs_t, act_t)
            q_values = torch.minimum(q1, q2).squeeze(-1).cpu().numpy()

        print(f"  Computed {len(q_values)} Q-values")

        #plot the things
        # Save CSV with Q-values
        df['critic_q_min'] = q_values
        
        critic_dir = self.output_base_dir / "4_critic_outputs"
        critic_dir.mkdir(exist_ok=True, parents=True)

        
        
        csv_out = critic_dir / f"{model_name}_critic_output.csv"
        png_out = critic_dir / f"{model_name}_critic_output.png"
        
        try:
            df.to_csv(csv_out, index=False)
            print(f"  ✓ Saved CSV: {csv_out.name}")
        except Exception as e:
            print(f"  ✗ Failed to save CSV: {e}")
        
        # Create plot
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # Plot 1: Q-values over time
            ax1.plot(q_values, linewidth=0.8, alpha=0.7, color='steelblue', label='Q(s,a)')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
            ax1.fill_between(range(len(q_values)), q_values, alpha=0.2, color='steelblue')
            ax1.set_title(f"{model_name} — Critic Output Q(s,a)", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Transition Index")
            ax1.set_ylabel("min(Q1, Q2)")
            ax1.grid(alpha=0.3, linestyle='--')
            ax1.legend()
            
            # Plot 2: Q-value distribution
            ax2.hist(q_values, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.axvline(x=q_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {q_values.mean():.4f}')
            ax2.set_title("Q-Value Distribution", fontsize=12)
            ax2.set_xlabel("Q-value")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(str(png_out), dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"  ✓ Saved plot: {png_out.name}")
        except Exception as e:
            print(f"  ✗ Failed to create plot: {e}")

        #Heatmap
        critic_heatmap_dir = self.output_base_dir / "5_critic_spatial_heatmaps"
        critic_heatmap_dir.mkdir(exist_ok=True, parents=True)
        png_out_heatmap = critic_heatmap_dir / f"{model_name}_critic_spatial_heatmap.png"

        print("  Generating critic Q-value heatmap...")
        try:
            df_with_pos = df.dropna(subset=['pose_x', 'pose_y'])
            
            if len(df_with_pos) > 0 and 'critic_q_min' in df_with_pos.columns:
                # Load map
                img_array, config = load_map_image(self.map_name)
                
                # We need pixel coordinates - recompute if needed
                if 'pixel_x' not in df_with_pos.columns:
                    origin = config['origin']
                    resolution = config['resolution']
                    pixel_coords = []
                    for _, row in df_with_pos.iterrows():
                        px, py = world_to_pixel(row['pose_x'], row['pose_y'], origin, resolution)
                        pixel_coords.append((px, py))
                    df_with_pos['pixel_x'] = [c[0] for c in pixel_coords]
                    df_with_pos['pixel_y'] = [c[1] for c in pixel_coords]
                
                # Plot the critic Q-value heatmap
                plot_value_heatmap(
                    df_with_pos, img_array, self.map_name, 
                    'critic_q_min', 'Q-value (min)',
                    save_path=str(png_out_heatmap)
                )
                plt.close('all')
                print(f"    ✓ Saved to {png_out_heatmap.name}")
            else:
                print(f"    ⊘ No critic Q-value data available")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
        # Print statistics
        print(f"\n  Critic Q-value statistics:")
        print(f"    Min:    {q_values.min():.6f}")
        print(f"    Max:    {q_values.max():.6f}")
        print(f"    Mean:   {q_values.mean():.6f}")
        print(f"    Median: {np.median(q_values):.6f}")
        print(f"    Std:    {q_values.std():.6f}")
        # print(obs_np, act_np)
        # critic(obs_t, act_t)
    
    def process_all(self):
        """Process all matching models with synchronized scales"""
        models = self.find_models_by_prefix()
        
        if not models:
            print(f"No models found matching prefix: {self.prefix}")
            return
        
        print(f"\nFound {len(models)} models matching prefix '{self.prefix}'")
        print(f"Models: {', '.join(models)}\n")
        
        # Compute global stats first for synchronized scaling
        print("Computing global statistics across all models...")
        csv_paths = [
            self.models_dir / model_name / "stat_logs" / "stats_log.csv"
            for model_name in models
        ]
        global_stats = compute_global_stats(csv_paths)
        
        for i, model_name in enumerate(models, 1):
            print(f"[{i}/{len(models)}]", end=" ")
            self.plot_model(model_name, global_stats=global_stats)
            if self.plot_critic_output:
                self.create_critic_output_plot(model_name)
        
        print(f"\n{'='*80}")
        print("BATCH PLOTTING COMPLETE")
        print(f"{'='*80}")
        print(f"All plots saved to: {self.output_base_dir}")
        print(f"\nYou can now browse:")
        print(f"  1. Distribution plots: {self.dist_dir}")
        print(f"  2. Spatial heatmaps:   {self.heatmap_dir}")
        print(f"  3. Reward heatmaps:    {self.reward_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process plot_sample_frequency.py for multiple models"
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Model name prefix to filter (e.g., '2602')"
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default='RCA1',
        help="Map name for spatial heatmap (default: RCA1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: batch_plot_results/batch_{prefix})"
    )
    parser.add_argument(
        "--plot-critic-output",
        type=bool,
        default=True,
        help="Whether to plot critic forward pass (default: True)"
    )
    
    args = parser.parse_args()
    
    plotter = BatchPlotter(
        prefix=args.prefix,
        map_name=args.map_name,
        output_base_dir=args.output_dir,
        plot_critic_output=args.plot_critic_output
    )
    plotter.process_all()



if __name__ == '__main__':
    main()
