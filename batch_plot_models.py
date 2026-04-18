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
import ast
import time
from pathlib import Path
import matplotlib
import torch


# Select matplotlib backend early (before pyplot/imports that may import pyplot).
# Default stays non-interactive for batch runs.
_backend_parser = argparse.ArgumentParser(add_help=False)
_backend_group = _backend_parser.add_mutually_exclusive_group()
_backend_group.add_argument("--interactive-plots", action="store_true")
_backend_group.add_argument("--non-interactive-plots", action="store_true")
_backend_args, _ = _backend_parser.parse_known_args()

if _backend_args.interactive_plots:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')

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


def compute_global_stats(csv_paths, min_possible_samples_percentile=5, stat_log_start_fraction=0.0):
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
    samples_per_batch_min = np.inf
    samples_per_batch_max = -np.inf
    possible_samples_min = np.inf
    possible_samples_max = -np.inf
    sample_count_max = -np.inf
    num_transitions_min = np.inf
    num_transitions_max = -np.inf

    reward_min = np.inf
    reward_max = -np.inf
    pose_x_min = np.inf
    pose_x_max = -np.inf
    pose_y_min = np.inf
    pose_y_max = -np.inf

    saw_any_models = False
    processed_models = 0
    saw_reward = False
    saw_pose = False
    
    print(f"  Applying {min_possible_samples_percentile}% possible_samples filtering to global stats...")
    if stat_log_start_fraction > 0.0:
        print(f"  Using tail of stat_logs with start fraction: {stat_log_start_fraction:.2f}")
    
    for csv_path in csv_paths:
        try:
            df_raw = pd.read_csv(
                csv_path,
                usecols=lambda col: col in {
                    'reward',
                    'pose_x',
                    'pose_y',
                    'sample_count',
                    'sample_calls_at_birth',
                    'sample_calls_at_death',
                },
            )
            loaded_count = len(df_raw)
            if stat_log_start_fraction > 0.0 and loaded_count > 0:
                start_idx = int(loaded_count * stat_log_start_fraction)
                df_raw = df_raw.iloc[start_idx:].copy()
                print(
                    f"    {csv_path.name}: using rows {start_idx}:{loaded_count} "
                    f"-> {len(df_raw)} transitions"
                )

            before_count = len(df_raw)
            
            # Apply same filtering as compute_sample_frequency to match plotted data
            df = compute_sample_frequency(df_raw, csv_path=csv_path, 
                                        min_possible_samples_percentile=min_possible_samples_percentile)
            after_count = len(df)
            
            if before_count != after_count:
                print(f"    {csv_path.name}: {before_count} -> {after_count} transitions (filtered {before_count - after_count})")

            saw_any_models = True
            processed_models += 1

            model_samples_min = float(df['samples_per_batch'].min())
            model_samples_max = float(df['samples_per_batch'].max())
            model_possible_min = float(df['possible_samples'].min())
            model_possible_max = float(df['possible_samples'].max())
            model_sample_count_max = float(df['sample_count'].max())

            samples_per_batch_min = min(samples_per_batch_min, model_samples_min)
            samples_per_batch_max = max(samples_per_batch_max, model_samples_max)
            possible_samples_min = min(possible_samples_min, model_possible_min)
            possible_samples_max = max(possible_samples_max, model_possible_max)
            sample_count_max = max(sample_count_max, model_sample_count_max)
            num_transitions_min = min(num_transitions_min, len(df))
            num_transitions_max = max(num_transitions_max, len(df))

            if 'reward' in df.columns:
                df_with_reward = df.dropna(subset=['reward'])
                if len(df_with_reward) > 0:
                    saw_reward = True
                    reward_min = min(reward_min, float(df_with_reward['reward'].min()))
                    reward_max = max(reward_max, float(df_with_reward['reward'].max()))

            if 'pose_x' in df.columns and 'pose_y' in df.columns:
                df_with_pos = df.dropna(subset=['pose_x', 'pose_y'])
                if len(df_with_pos) > 0:
                    saw_pose = True
                    pose_x_min = min(pose_x_min, float(df_with_pos['pose_x'].min()))
                    pose_x_max = max(pose_x_max, float(df_with_pos['pose_x'].max()))
                    pose_y_min = min(pose_y_min, float(df_with_pos['pose_y'].min()))
                    pose_y_max = max(pose_y_max, float(df_with_pos['pose_y'].max()))

            del df_raw
            del df
        except Exception as e:
            print(f"  Warning: Failed to load {csv_path}: {e}")
    
    if not saw_any_models:
        print("Error: No CSVs loaded successfully")
        return {}

    global_stats = {
        'samples_per_batch_min': samples_per_batch_min,
        'samples_per_batch_max': samples_per_batch_max,
        'possible_samples_min': possible_samples_min,
        'possible_samples_max': possible_samples_max,
        'sample_count_max': sample_count_max,
        'num_transitions_min': int(num_transitions_min),
        'num_transitions_max': int(num_transitions_max),
    }
    
    print(f"  After filtering, max sample_count in combined data: {global_stats['sample_count_max']:.0f}")
    
    # Reward range (if reward column exists)
    if saw_reward:
        global_stats['reward_min'] = reward_min
        global_stats['reward_max'] = reward_max
    
    # Spatial ranges (if position data exists)
    if saw_pose:
        global_stats['pose_x_min'] = pose_x_min
        global_stats['pose_x_max'] = pose_x_max
        global_stats['pose_y_min'] = pose_y_min
        global_stats['pose_y_max'] = pose_y_max
    
    print(f"\n  Global statistics computed across {processed_models} models:")
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
    
    def __init__(
        self,
        prefix: str,
        map_name: str = 'RCA1',
        output_base_dir: str = None,
        plot_base_metrics: bool = True,
        plot_critic_output: bool = False,
        plot_td_error: bool = False,
        plot_td_improvement: bool = False,
        td_sample_stride: int = 5,
        plot_td_extremes: bool = False,
        td_extreme_percent: float = 2.0,
        td_extremes_only: bool = False,
        stat_log_start_fraction: float = 0.0,
    ):
        """
        Initialize batch plotter.
        
        Args:
            prefix: Model name prefix to filter (e.g., "2602")
            map_name: Map name for spatial heatmap
            output_base_dir: Base directory for organized outputs
            plot_base_metrics: Whether to generate distribution/spatial/reward maps
            plot_critic_output: Whether to plot critic output
            plot_td_error: Whether to plot TD-error spatial heatmap
            plot_td_improvement: Whether to plot TD-error improvement heatmap
            td_sample_stride: Plot every Nth transition for TD-related maps
            plot_td_extremes: Whether to add maps for top/bottom TD extremes only
            td_extreme_percent: Percentile used for top and bottom TD extremes
            td_extremes_only: If True, skip normal TD maps and write only extremes maps
            stat_log_start_fraction: Keep only rows from this fraction to end (0.5 => second half)
        """

        if not (0.0 <= float(stat_log_start_fraction) < 1.0):
            raise ValueError("stat_log_start_fraction must be in [0.0, 1.0)")

        self.prefix = prefix
        self.map_name = map_name
        self.plot_base_metrics = plot_base_metrics
        self.plot_critic_output = plot_critic_output
        self.plot_td_error = plot_td_error
        self.plot_td_improvement = plot_td_improvement
        self.td_sample_stride = max(1, int(td_sample_stride))
        self.plot_td_extremes = plot_td_extremes
        self.td_extreme_percent = float(np.clip(td_extreme_percent, 0.0, 49.9))
        self.td_extremes_only = td_extremes_only
        self.stat_log_start_fraction = float(stat_log_start_fraction)
        
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
        print(f"  TD sample stride:   {self.td_sample_stride}")
        if self.stat_log_start_fraction > 0.0:
            print(f"  Stat log segment:   rows [{self.stat_log_start_fraction:.2f}, 1.00)")
        if self.plot_td_extremes:
            print(f"  TD extremes:        bottom/top {self.td_extreme_percent:.2f}%")
        if self.td_extremes_only:
            print("  TD mode:            extremes-only")

    def _slice_stat_log_frame(self, df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
        """Keep only the tail of a stat_log dataframe based on configured start fraction."""
        if self.stat_log_start_fraction <= 0.0 or len(df) == 0:
            return df

        total = len(df)
        start_idx = int(total * self.stat_log_start_fraction)
        df_sliced = df.iloc[start_idx:].copy()
        print(
            f"  Applying stat_log_start_fraction={self.stat_log_start_fraction:.2f} "
            f"to {csv_path.name}: rows {start_idx}:{total} -> {len(df_sliced)} transitions"
        )
        return df_sliced

    def _resolve_td_error_series(self, df: pd.DataFrame):
        """Resolve TD-error and improvement series from available stat tracker columns."""
        preferred_columns = [
            "TD_error_latest",
            "TD_error_mean",
            "TD_error_max",
            "TD_error_min",
            "TD_error_first",
        ]

        for col in preferred_columns:
            if col in df.columns and df[col].notna().any():
                td_series = pd.to_numeric(df[col], errors='coerce')
                improvement_series = None
                improvement_source = None

                if "TD_error_first" in df.columns and "TD_error_latest" in df.columns:
                    td_first = pd.to_numeric(df["TD_error_first"], errors='coerce')
                    td_latest = pd.to_numeric(df["TD_error_latest"], errors='coerce')
                    improvement_series = td_first - td_latest
                    improvement_source = "TD_error_first_minus_latest"

                return td_series, col, improvement_series, improvement_source

        if "TD_error_list" in df.columns:
            def parse_list_metrics(value):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return np.nan, np.nan
                if isinstance(value, list):
                    if len(value) == 0:
                        return np.nan, np.nan
                    first_val = float(value[0])
                    last_val = float(value[-1])
                    return last_val, first_val - last_val
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        return np.nan, np.nan
                    try:
                        parsed = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        return np.nan, np.nan
                    if isinstance(parsed, list) and len(parsed) > 0:
                        first_val = float(parsed[0])
                        last_val = float(parsed[-1])
                        return last_val, first_val - last_val
                return np.nan, np.nan

            parsed_pairs = df["TD_error_list"].apply(parse_list_metrics)
            latest_series = parsed_pairs.apply(lambda x: x[0])
            improvement_series = parsed_pairs.apply(lambda x: x[1])
            if latest_series.notna().any():
                return latest_series, "TD_error_list_latest", improvement_series, "TD_error_list_first_minus_latest"

        return None, None, None, None

    def _prepare_td_spatial_frame(self, df: pd.DataFrame, value_column: str) -> pd.DataFrame:
        """Filter and downsample a TD-related frame for spatial plotting."""
        df_with_pos = df.dropna(subset=["pose_x", "pose_y", value_column]).copy()
        if len(df_with_pos) == 0:
            return df_with_pos

        if self.td_sample_stride > 1:
            df_with_pos = df_with_pos.iloc[::self.td_sample_stride].copy()

        return df_with_pos

    def _resolve_td_error_mean_series(self, df: pd.DataFrame):
        """Resolve average TD error per transition from stat tracker columns."""
        if "TD_error_mean" in df.columns and df["TD_error_mean"].notna().any():
            return pd.to_numeric(df["TD_error_mean"], errors='coerce'), "TD_error_mean"

        if "TD_error_list" in df.columns:
            def parse_mean(value):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return np.nan
                if isinstance(value, list):
                    return float(np.mean(value)) if len(value) > 0 else np.nan
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        return np.nan
                    try:
                        parsed = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        return np.nan
                    if isinstance(parsed, list) and len(parsed) > 0:
                        return float(np.mean(parsed))
                return np.nan

            series = df["TD_error_list"].apply(parse_mean)
            if series.notna().any():
                return series, "TD_error_list_mean"

        return None, None

    def _resolve_model_checkpoint_path(self, model_name: str):
        """Return checkpoint path without extension (preferred by SB3 load API)."""
        model_dir = self.models_dir / model_name
        base_path = model_dir / model_name
        zip_path = model_dir / f"{model_name}.zip"

        # SB3's loader may append ".zip" internally, so pass the base stem when possible.
        if zip_path.exists() or base_path.exists():
            return base_path

        return None

    @staticmethod
    def _signed_log1p(values: pd.Series) -> pd.Series:
        return np.sign(values) * np.log1p(np.abs(values))

    @staticmethod
    def _signed_expm1(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * (np.expm1(np.abs(values)))

    def _build_log_colorbar_ticks(self, real_values: pd.Series):
        """Build colorbar tick positions in log space and labels in real TD units."""
        clean = pd.to_numeric(real_values, errors='coerce').dropna()
        if len(clean) == 0:
            return None, None

        real_min = float(clean.min())
        real_max = float(clean.max())

        if np.isclose(real_min, real_max):
            ticks_real = np.array([real_min], dtype=float)
        elif real_min < 0 < real_max:
            neg = np.linspace(real_min, 0.0, 4, dtype=float)
            pos = np.linspace(0.0, real_max, 4, dtype=float)
            ticks_real = np.unique(np.concatenate([neg[:-1], pos]))
        else:
            ticks_real = np.linspace(real_min, real_max, 7, dtype=float)

        ticks_log = self._signed_log1p(pd.Series(ticks_real)).to_numpy()
        tick_labels = [f"{v:.3g}" for v in ticks_real]
        return ticks_log, tick_labels

    @staticmethod
    def _print_timing(label: str, start_time: float) -> None:
        elapsed = time.perf_counter() - start_time
        print(f"    [timing] {label}: {elapsed:.2f}s")

    def _select_td_extremes(self, df: pd.DataFrame, value_column: str) -> pd.DataFrame:
        """Keep TD extreme rows for the specified value.

        In regular mode this keeps bottom+top tails.
        In td-extremes-only mode this keeps only the top tail.
        """
        if self.td_extreme_percent <= 0:
            return df.iloc[0:0].copy()

        values = pd.to_numeric(df[value_column], errors='coerce').dropna()
        if len(values) == 0:
            return df.iloc[0:0].copy()

        high_cut = np.percentile(values.to_numpy(), 100.0 - self.td_extreme_percent)
        if self.td_extremes_only:
            mask = df[value_column] >= high_cut
        else:
            low_cut = np.percentile(values.to_numpy(), self.td_extreme_percent)
            mask = (df[value_column] <= low_cut) | (df[value_column] >= high_cut)
        return df.loc[mask].copy()

    def _plot_td_extremes_maps(
        self,
        model_name: str,
        img_array,
        df_with_pos: pd.DataFrame,
        value_column: str,
        label: str,
        prefix: str,
    ) -> None:
        """Plot maps containing only bottom/top TD extremes."""
        extremes_df = self._select_td_extremes(df_with_pos, value_column)
        if len(extremes_df) == 0:
            print(f"  ⊘ No TD extremes found for {prefix}")
            return

        tail_label = (
            f"top {self.td_extreme_percent:.1f}%"
            if self.td_extremes_only
            else f"bottom/top {self.td_extreme_percent:.1f}%"
        )

        extremes_dir = self.output_base_dir / "10_td_error_extremes"
        extremes_dir.mkdir(exist_ok=True, parents=True)

        signed_path = extremes_dir / f"{model_name}_{prefix}_extremes_signed.png"
        abs_path = extremes_dir / f"{model_name}_{prefix}_extremes_abs.png"
        print(f"  TD extremes rows ({prefix}): {len(extremes_df)}")

        try:
            start_time = time.perf_counter()
            plot_value_heatmap(
                extremes_df,
                img_array,
                self.map_name,
                value_column,
                f"{label} extremes ({tail_label})",
                save_path=str(signed_path),
            )
            plt.close('all')
            print(f"  ✓ Saved TD extremes map: {signed_path.name}")
            self._print_timing(f"TD extremes map ({prefix}, signed)", start_time)
        except Exception as e:
            print(f"  ✗ Failed to create TD extremes signed map ({prefix}): {e}")

        try:
            start_time = time.perf_counter()
            extremes_abs = extremes_df.copy()
            abs_col = f"{value_column}_abs"
            extremes_abs[abs_col] = np.abs(extremes_abs[value_column])
            plot_value_heatmap(
                extremes_abs,
                img_array,
                self.map_name,
                abs_col,
                f"|{label}| extremes ({tail_label})",
                save_path=str(abs_path),
            )
            plt.close('all')
            print(f"  ✓ Saved TD extremes |value| map: {abs_path.name}")
            self._print_timing(f"TD extremes map ({prefix}, abs)", start_time)
        except Exception as e:
            print(f"  ✗ Failed to create TD extremes |value| map ({prefix}): {e}")

    def create_td_error_spatial_plot(self, model_name: str):
        """Generate spatial heatmap of TD error using logged stat tracker values."""
        print(f"\n{'='*80}")
        print(f"Generating TD-error spatial heatmap: {model_name}")
        print(f"{'='*80}")

        csv_path = self.models_dir / model_name / "stat_logs" / "stats_log.csv"

        try:
            df = pd.read_csv(csv_path)
            df = self._slice_stat_log_frame(df, csv_path)
            if len(df) == 0:
                print("  ⊘ No transitions left after stat_log slicing")
                return
        except Exception as e:
            print(f"  ✗ Failed to load CSV for TD-error heatmap: {e}")
            return

        td_series, td_source, td_improvement_series, td_improvement_source = self._resolve_td_error_series(df)
        if td_series is None:
            print("  ⊘ No TD-error data available (checked latest/mean/max/min/first/list)")
            return

        df = df.copy()
        df["td_error_for_plot"] = td_series
        df_with_pos = self._prepare_td_spatial_frame(df, "td_error_for_plot")

        if len(df_with_pos) == 0:
            print("  ⊘ No rows with valid pose_x/pose_y/TD error")
            return

        td_dir = self.output_base_dir / "6_td_error_spatial_heatmaps"
        td_dir.mkdir(exist_ok=True, parents=True)
        png_out = td_dir / f"{model_name}_td_error_spatial_heatmap.png"
        td_raw_dir = self.output_base_dir / "7_td_error_heatmaps_raw"
        td_raw_dir.mkdir(exist_ok=True, parents=True)
        png_out_raw = td_raw_dir / f"{model_name}_td_error_spatial_heatmap_raw.png"
        td_log_dir = self.output_base_dir / "8_td_error_heatmaps_log"
        td_log_dir.mkdir(exist_ok=True, parents=True)
        png_out_log = td_log_dir / f"{model_name}_td_error_spatial_heatmap_log.png"
        avg_td_dir = self.output_base_dir / "9_avg_td_error_heatmaps"
        avg_td_dir.mkdir(exist_ok=True, parents=True)

        print(f"  Using TD source: {td_source}")
        print(f"  Rows plotted: {len(df_with_pos)}")
        if self.td_sample_stride > 1:
            print(f"  Downsampled by stride: every {self.td_sample_stride}th transition")

        try:
            img_array, config = load_map_image(self.map_name)
        except Exception as e:
            print(f"  ✗ Failed to load map image: {e}")
            return

        if 'pixel_x' not in df_with_pos.columns:
            origin = config['origin']
            resolution = config['resolution']
            pixel_coords = []
            for _, row in df_with_pos.iterrows():
                px, py = world_to_pixel(row['pose_x'], row['pose_y'], origin, resolution)
                pixel_coords.append((px, py))
            df_with_pos['pixel_x'] = [c[0] for c in pixel_coords]
            df_with_pos['pixel_y'] = [c[1] for c in pixel_coords]

        if not self.td_extremes_only:
            try:
                start_time = time.perf_counter()

                plot_value_heatmap(
                    df_with_pos,
                    img_array,
                    self.map_name,
                    'td_error_for_plot',
                    f'TD Error ({td_source})',
                    save_path=str(png_out),
                    robust_percentiles=(2, 98),
                )
                plt.close('all')
                print(f"  ✓ Saved TD-error heatmap: {png_out.name}")
                self._print_timing("TD-error heatmap", start_time)
            except Exception as e:
                print(f"  ✗ Failed to create TD-error heatmap: {e}")

            try:
                start_time = time.perf_counter()
                plot_value_heatmap(
                    df_with_pos,
                    img_array,
                    self.map_name,
                    'td_error_for_plot',
                    f'TD Error ({td_source}) RAW',
                    save_path=str(png_out_raw)
                )
                plt.close('all')
                print(f"  ✓ Saved RAW TD-error heatmap: {png_out_raw.name}")
                self._print_timing("TD-error heatmap raw", start_time)
            except Exception as e:
                print(f"  ✗ Failed to create RAW TD-error heatmap: {e}")

            try:
                start_time = time.perf_counter()
                df_with_pos_log = df_with_pos.copy()
                df_with_pos_log['td_error_for_plot_log'] = self._signed_log1p(df_with_pos_log['td_error_for_plot'])
                tick_values, tick_labels = self._build_log_colorbar_ticks(df_with_pos_log['td_error_for_plot'])
                plot_value_heatmap(
                    df_with_pos_log,
                    img_array,
                    self.map_name,
                    'td_error_for_plot_log',
                    f'TD Error ({td_source}) [signed log1p color scale, ticks=real TD]',
                    save_path=str(png_out_log),
                    colorbar_tick_values=tick_values,
                    colorbar_tick_labels=tick_labels,
                )
                plt.close('all')
                print(f"  ✓ Saved LOG TD-error heatmap: {png_out_log.name}")
                self._print_timing("TD-error heatmap log", start_time)
            except Exception as e:
                print(f"  ✗ Failed to create LOG TD-error heatmap: {e}")

        # Dedicated average TD-error maps
        td_mean_series, td_mean_source = self._resolve_td_error_mean_series(df)
        if td_mean_series is not None:
            df["td_mean_for_plot"] = td_mean_series
            df_td_mean = self._prepare_td_spatial_frame(df, "td_mean_for_plot")

            if len(df_td_mean) > 0:
                if 'pixel_x' not in df_td_mean.columns:
                    origin = config['origin']
                    resolution = config['resolution']
                    pixel_coords = []
                    for _, row in df_td_mean.iterrows():
                        px, py = world_to_pixel(row['pose_x'], row['pose_y'], origin, resolution)
                        pixel_coords.append((px, py))
                    df_td_mean['pixel_x'] = [c[0] for c in pixel_coords]
                    df_td_mean['pixel_y'] = [c[1] for c in pixel_coords]

                avg_png = avg_td_dir / f"{model_name}_avg_td_error_spatial_heatmap.png"
                avg_png_raw = avg_td_dir / f"{model_name}_avg_td_error_spatial_heatmap_raw.png"
                avg_png_log = avg_td_dir / f"{model_name}_avg_td_error_spatial_heatmap_log.png"

                if not self.td_extremes_only:
                    try:
                        start_time = time.perf_counter()
                        plot_value_heatmap(
                            df_td_mean,
                            img_array,
                            self.map_name,
                            'td_mean_for_plot',
                            f'Average TD Error ({td_mean_source})',
                            save_path=str(avg_png),
                            robust_percentiles=(2, 98),
                        )
                        plt.close('all')
                        print(f"  ✓ Saved average TD heatmap: {avg_png.name}")
                        self._print_timing("average TD heatmap", start_time)
                    except Exception as e:
                        print(f"  ✗ Failed to create average TD heatmap: {e}")

                    try:
                        start_time = time.perf_counter()
                        plot_value_heatmap(
                            df_td_mean,
                            img_array,
                            self.map_name,
                            'td_mean_for_plot',
                            f'Average TD Error ({td_mean_source}) RAW',
                            save_path=str(avg_png_raw),
                        )
                        plt.close('all')
                        print(f"  ✓ Saved RAW average TD heatmap: {avg_png_raw.name}")
                        self._print_timing("average TD heatmap raw", start_time)
                    except Exception as e:
                        print(f"  ✗ Failed to create RAW average TD heatmap: {e}")

                    try:
                        start_time = time.perf_counter()
                        df_td_mean_log = df_td_mean.copy()
                        df_td_mean_log['td_mean_for_plot_log'] = self._signed_log1p(df_td_mean_log['td_mean_for_plot'])
                        tick_values, tick_labels = self._build_log_colorbar_ticks(df_td_mean_log['td_mean_for_plot'])
                        plot_value_heatmap(
                            df_td_mean_log,
                            img_array,
                            self.map_name,
                            'td_mean_for_plot_log',
                            f'Average TD Error ({td_mean_source}) [signed log1p color scale, ticks=real TD]',
                            save_path=str(avg_png_log),
                            colorbar_tick_values=tick_values,
                            colorbar_tick_labels=tick_labels,
                        )
                        plt.close('all')
                        print(f"  ✓ Saved LOG average TD heatmap: {avg_png_log.name}")
                        self._print_timing("average TD heatmap log", start_time)
                    except Exception as e:
                        print(f"  ✗ Failed to create LOG average TD heatmap: {e}")

                if self.plot_td_extremes:
                    self._plot_td_extremes_maps(
                        model_name=model_name,
                        img_array=img_array,
                        df_with_pos=df_td_mean,
                        value_column='td_mean_for_plot',
                        label=f'Average TD Error ({td_mean_source})',
                        prefix='avg_td_error',
                    )
            else:
                print("  ⊘ No rows with valid pose_x/pose_y/average TD error")
        else:
            print("  ⊘ No average TD-error data available")

        if self.plot_td_extremes:
            self._plot_td_extremes_maps(
                model_name=model_name,
                img_array=img_array,
                df_with_pos=df_with_pos,
                value_column='td_error_for_plot',
                label=f'TD Error ({td_source})',
                prefix='td_error',
            )

        if not self.plot_td_improvement:
            return

        if td_improvement_series is None or not td_improvement_series.notna().any():
            print("  ⊘ No TD-error improvement data available")
            return

        df["td_improvement_for_plot"] = td_improvement_series
        df_improvement = self._prepare_td_spatial_frame(df, "td_improvement_for_plot")
        if len(df_improvement) == 0:
            print("  ⊘ No rows with valid pose_x/pose_y/TD improvement")
            return

        png_out_improvement = td_dir / f"{model_name}_td_error_improvement_spatial_heatmap.png"
        png_out_improvement_raw = td_raw_dir / f"{model_name}_td_error_improvement_spatial_heatmap_raw.png"
        png_out_improvement_log = td_log_dir / f"{model_name}_td_error_improvement_spatial_heatmap_log.png"
        print(f"  Using TD improvement source: {td_improvement_source}")
        print(f"  Rows plotted for improvement: {len(df_improvement)}")

        try:
            start_time = time.perf_counter()
            img_array, config = load_map_image(self.map_name)

            if 'pixel_x' not in df_improvement.columns:
                origin = config['origin']
                resolution = config['resolution']
                pixel_coords = []
                for _, row in df_improvement.iterrows():
                    px, py = world_to_pixel(row['pose_x'], row['pose_y'], origin, resolution)
                    pixel_coords.append((px, py))
                df_improvement['pixel_x'] = [c[0] for c in pixel_coords]
                df_improvement['pixel_y'] = [c[1] for c in pixel_coords]

            plot_value_heatmap(
                df_improvement,
                img_array,
                self.map_name,
                'td_improvement_for_plot',
                f'TD Improvement ({td_improvement_source})',
                save_path=str(png_out_improvement),
                robust_percentiles=(2, 98),
            )
            plt.close('all')
            print(f"  ✓ Saved TD-error improvement heatmap: {png_out_improvement.name}")
            self._print_timing("TD-error improvement heatmap", start_time)
        except Exception as e:
            print(f"  ✗ Failed to create TD-error improvement heatmap: {e}")

        try:
            start_time = time.perf_counter()
            plot_value_heatmap(
                df_improvement,
                img_array,
                self.map_name,
                'td_improvement_for_plot',
                f'TD Improvement ({td_improvement_source}) RAW',
                save_path=str(png_out_improvement_raw)
            )
            plt.close('all')
            print(f"  ✓ Saved RAW TD-error improvement heatmap: {png_out_improvement_raw.name}")
            self._print_timing("TD-error improvement heatmap raw", start_time)
        except Exception as e:
            print(f"  ✗ Failed to create RAW TD-error improvement heatmap: {e}")

        try:
            start_time = time.perf_counter()
            df_improvement_log = df_improvement.copy()
            df_improvement_log['td_improvement_for_plot_log'] = self._signed_log1p(df_improvement_log['td_improvement_for_plot'])
            tick_values, tick_labels = self._build_log_colorbar_ticks(df_improvement_log['td_improvement_for_plot'])
            plot_value_heatmap(
                df_improvement_log,
                img_array,
                self.map_name,
                'td_improvement_for_plot_log',
                f'TD Improvement ({td_improvement_source}) [signed log1p color scale, ticks=real TD]',
                save_path=str(png_out_improvement_log),
                colorbar_tick_values=tick_values,
                colorbar_tick_labels=tick_labels,
            )
            plt.close('all')
            print(f"  ✓ Saved LOG TD-error improvement heatmap: {png_out_improvement_log.name}")
            self._print_timing("TD-error improvement heatmap log", start_time)
        except Exception as e:
            print(f"  ✗ Failed to create LOG TD-error improvement heatmap: {e}")
    
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
            loaded_total = len(df)
            df = self._slice_stat_log_frame(df, csv_path)
            if len(df) == 0:
                print("  ⊘ No transitions left after stat_log slicing")
                return
            if self.stat_log_start_fraction > 0.0:
                print(f"  Loaded {len(df)} transitions after slicing (from {loaded_total})")
            else:
                print(f"  Loaded {len(df)} transitions")
            
            # Compute sample frequency
            df = compute_sample_frequency(df, csv_path=str(csv_path))
            model_sample_count_max = float(df['sample_count'].max())
            print(f"  Filtered max absolute sample count: {model_sample_count_max:.0f}")
            
            # Plot 1: Sample frequency distribution
            print("  Generating distribution plot...")
            start_time = time.perf_counter()
            dist_save_path = self.dist_dir / f"{model_name}_distribution.png"
            try:
                plot_sample_frequency_distribution_static(df, save_path=str(dist_save_path), global_stats=global_stats)
                plt.close('all')  # Close figure to free memory
                print(f"    ✓ Saved to {dist_save_path.name}")
                self._print_timing("distribution plot", start_time)
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            # Plot 2: Spatial heatmap
            print("  Generating spatial heatmap...")
            start_time = time.perf_counter()
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
                    self._print_timing("spatial heatmap", start_time)
                else:
                    print(f"    ⊘ No position data available")
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            # Plot 3: Reward heatmap
            print("  Generating reward heatmap...")
            start_time = time.perf_counter()
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
                    reward_vmin = global_stats.get('reward_min') if global_stats else None
                    reward_vmax = global_stats.get('reward_max') if global_stats else None
                    plot_reward_heatmap(
                        df_with_pos,
                        img_array,
                        self.map_name,
                        value_min=reward_vmin,
                        value_max=reward_vmax,
                    )
                    plt.savefig(str(reward_save_path), dpi=150, bbox_inches='tight')
                    plt.close('all')
                    print(f"    ✓ Saved to {reward_save_path.name}")
                    self._print_timing("reward heatmap", start_time)
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

        from stable_baselines3 import SAC
        from TrainingLite.rl_racing.sac_utilities import SacUtilities

        print(f"\n{'='*80}")
        print(f"Calculating and plotting critic output: {model_name}")
        print(f"{'='*80}")

        csv_path = self.models_dir / model_name / "stat_logs" / "stats_log.csv"
        model_path = self._resolve_model_checkpoint_path(model_name)

        if model_path is None:
            print(f"  ✗ Checkpoint missing for {model_name}")
            print(f"    Looked for: {self.models_dir / model_name / model_name}")
            print(f"    and:       {self.models_dir / model_name / (model_name + '.zip')}")
            return
        
        dummy_env = SacUtilities.create_vec_env()
        start_time = time.perf_counter()
        model = SAC.load(model_path, env=dummy_env, device="cpu")
        # actor = self.model.policy.actor
        critic = model.policy.critic

        try:
            df_full = pd.read_csv(csv_path)
            if 'obs' not in df_full.columns or 'action' not in df_full.columns:
                print(f"ERROR: CSV missing obs/action columns. Enable extended_obs_action_save.")
                return

            df = self._slice_stat_log_frame(df_full, csv_path)
            if len(df) == 0:
                print("  ⊘ No transitions left after stat_log slicing")
                return
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
            plot_start_time = time.perf_counter()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # Plot 1: Q-values over time
            ax1.plot(q_values, linewidth=0.8, alpha=0.7, color='steelblue', label='Q(s,a)')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
            ax1.fill_between(range(len(q_values)), q_values, alpha=0.2, color='steelblue')
            ax1.set_title(f"{model_name} — Critic Output Q(s,a) (Selected Segment)", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Transition Index (selected segment)")
            ax1.set_ylabel("min(Q1, Q2)")
            ax1.grid(alpha=0.3, linestyle='--')
            ax1.legend()
            
            # Plot 2: Q-value distribution
            ax2.hist(q_values, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.axvline(x=q_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {q_values.mean():.4f}')
            ax2.set_title("Q-Value Distribution (Selected Segment)", fontsize=12)
            ax2.set_xlabel("Q-value")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(str(png_out), dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"  ✓ Saved plot: {png_out.name}")
            self._print_timing("critic output plot", plot_start_time)
        except Exception as e:
            print(f"  ✗ Failed to create plot: {e}")

        #Heatmap
        critic_heatmap_dir = self.output_base_dir / "5_critic_spatial_heatmaps"
        critic_heatmap_dir.mkdir(exist_ok=True, parents=True)
        png_out_heatmap = critic_heatmap_dir / f"{model_name}_critic_spatial_heatmap.png"

        print("  Generating critic Q-value heatmap...")
        try:
            heatmap_start_time = time.perf_counter()
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
                self._print_timing("critic spatial heatmap", heatmap_start_time)
            else:
                print(f"    ⊘ No critic Q-value data available")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
        # Print statistics
        print(f"\n  Critic Q-value statistics (selected segment):")
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
        
        global_stats = None
        if self.plot_base_metrics:
            # Compute global stats first for synchronized scaling
            print("Computing global statistics across all models...")
            csv_paths = [
                self.models_dir / model_name / "stat_logs" / "stats_log.csv"
                for model_name in models
            ]
            global_stats = compute_global_stats(
                csv_paths,
                stat_log_start_fraction=self.stat_log_start_fraction,
            )
        
        for i, model_name in enumerate(models, 1):
            print(f"[{i}/{len(models)}]", end=" ")
            if self.plot_base_metrics:
                self.plot_model(model_name, global_stats=global_stats)
            if self.plot_critic_output:
                self.create_critic_output_plot(model_name)
            if self.plot_td_error:
                self.create_td_error_spatial_plot(model_name)
        
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to plot critic forward pass (default: True)"
    )
    parser.add_argument(
        "--plot-td-error",
        action="store_true",
        help="Generate TD-error spatial heatmaps from stat_logs (default: off)"
    )
    parser.add_argument(
        "--plot-td-improvement",
        action="store_true",
        help="Also generate TD-error improvement heatmaps (default: off)"
    )
    parser.add_argument(
        "--plot-td-extremes",
        action="store_true",
        help="Also generate maps for bottom/top TD-error percentiles"
    )
    parser.add_argument(
        "--td-extreme-percent",
        type=float,
        default=2.0,
        help="Bottom/top percentile for TD-extremes maps (default: 2.0)"
    )
    parser.add_argument(
        "--td-only",
        action="store_true",
        help="Run only TD-error plotting from stat_logs (skip base/critic plots)"
    )
    parser.add_argument(
        "--td-extremes-only",
        action="store_true",
        help="Generate only TD-extremes maps (implies --td-only and --plot-td-extremes)"
    )
    parser.add_argument(
        "--td-sample-stride",
        type=int,
        default=5,
        help="Only plot every Nth transition for TD-related maps (default: 5)"
    )
    parser.add_argument(
        "--stat-log-start-fraction",
        type=float,
        default=0.0,
        help=(
            "Keep only stat_log rows from this fraction to the end. "
            "Examples: 0.5 => second half, 0.75 => last quarter"
        ),
    )
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--interactive-plots",
        action="store_true",
        help="Use an interactive matplotlib backend (TkAgg)."
    )
    backend_group.add_argument(
        "--non-interactive-plots",
        action="store_true",
        help="Force non-interactive backend (Agg, default)."
    )
    
    args = parser.parse_args()

    if not (0.0 <= args.stat_log_start_fraction < 1.0):
        parser.error("--stat-log-start-fraction must be in [0.0, 1.0)")

    selected_backend = matplotlib.get_backend()
    print(f"Matplotlib backend: {selected_backend}")

    plot_base_metrics = not args.td_only
    plot_critic_output = args.plot_critic_output and not args.td_only
    plot_td_error = args.plot_td_error or args.td_only
    if args.td_extremes_only:
        plot_base_metrics = False
        plot_critic_output = False
        plot_td_error = True
    
    plotter = BatchPlotter(
        prefix=args.prefix,
        map_name=args.map_name,
        output_base_dir=args.output_dir,
        plot_base_metrics=plot_base_metrics,
        plot_critic_output=plot_critic_output,
        plot_td_error=plot_td_error,
        plot_td_improvement=args.plot_td_improvement,
        td_sample_stride=args.td_sample_stride,
        plot_td_extremes=(args.plot_td_extremes or args.td_extremes_only),
        td_extreme_percent=args.td_extreme_percent,
        td_extremes_only=args.td_extremes_only,
        stat_log_start_fraction=args.stat_log_start_fraction,
    )
    plotter.process_all()



if __name__ == '__main__':
    main()
