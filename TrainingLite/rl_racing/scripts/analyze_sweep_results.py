#!/usr/bin/env python3
"""
Analyze results from sweep experiments and generate plots/statistics.

Usage:
    python TrainingLite/rl_racing/scripts/analyze_sweep_results.py --prefix "Sweep_rank_Ex1_A0.0_B0.4_R0.0"
    python TrainingLite/rl_racing/scripts/analyze_sweep_results.py --results-file "sweep_experiment_results/sweep_Sweep_rank_*.csv"
"""

import os
import sys
import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add root dir to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_dir)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/pandas not available. Install with: pip install pandas matplotlib")


class SweepResultsAnalyzer:
    """Analyze and visualize sweep experiment results."""
    
    def __init__(self, results_dir: str = "sweep_experiment_results"):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing CSV results
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            self.results_dir.mkdir(exist_ok=True)
    
    def find_results_file(self, prefix: str) -> Path:
        """Find the results CSV file for a given prefix."""
        pattern = f"sweep_{prefix}_results.csv"
        candidates = list(self.results_dir.glob(pattern))
        if candidates:
            return candidates[0]
        
        # Try wildcard search
        pattern = f"sweep_{prefix}*_results.csv"
        candidates = list(self.results_dir.glob(pattern))
        if candidates:
            return candidates[0]
        
        raise FileNotFoundError(f"No results file found for prefix: {prefix}")
    
    def load_results(self, results_file: Path) -> pd.DataFrame:
        """Load results from CSV file."""
        if not HAS_MATPLOTLIB:
            raise ImportError("pandas is required for analysis")
        
        df = pd.read_csv(results_file)
        
        # Convert numeric columns
        numeric_cols = [
            'num_laps_completed', 'num_laps_attempted',
            'avg_lap_time', 'min_lap_time', 'max_lap_time', 'std_lap_time', 'lap_time_range', 'lap_consistency',
            'avg_speed', 'max_speed', 'min_speed', 'std_speed',
            'avg_distance_to_raceline', 'max_distance_to_raceline', 'std_distance_to_raceline',
            'avg_steering_angle', 'max_steering_angle', 'steering_smoothness',
            'avg_acceleration', 'max_acceleration', 'acceleration_smoothness',
            'total_sim_time', 'avg_time_per_lap',
            'num_crashes', 'num_car_states_recorded', 'num_control_inputs'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'crash_occurred' in df.columns:
            df['crash_occurred'] = df['crash_occurred'].astype(str).str.lower().isin(['true', '1', 'yes'])
        
        return df

    def _get_group_key(self, model_name: str) -> str:
        """Group key for repeated runs by stripping trailing _RunN."""
        return re.sub(r"_Run\d+$", "", str(model_name), flags=re.IGNORECASE)
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """Print comprehensive summary statistics."""
        print("\n" + "="*80)
        print("SWEEP RESULTS ANALYSIS")
        print("="*80)
        
        total_models = len(df)
        successful = (df['status'] == 'completed').sum()
        failed = (df['status'] != 'completed').sum()
        
        print(f"\nTotal models: {total_models}")
        print(f"Successful: {successful} ({100*successful/total_models:.1f}%)")
        print(f"Failed: {failed} ({100*failed/total_models:.1f}%)")
        
        # Completed lap analysis
        completed_df = df[df['status'] == 'completed']
        if not completed_df.empty:
            print(f"\n{'='*80}")
            print("LAP TIME STATISTICS (completed models only)")
            print(f"{'─'*80}")
            print(f"  Mean:            {completed_df['avg_lap_time'].mean():.4f}s")
            print(f"  Median:          {completed_df['avg_lap_time'].median():.4f}s")
            print(f"  Std deviation:   {completed_df['avg_lap_time'].std():.4f}s")
            print(f"  Best:            {completed_df['avg_lap_time'].min():.4f}s")
            print(f"  Worst:           {completed_df['avg_lap_time'].max():.4f}s")
            
            # Speed statistics
            if 'avg_speed' in completed_df.columns:
                print(f"\n{'='*80}")
                print("SPEED STATISTICS")
                print(f"{'─'*80}")
                speed_data = completed_df['avg_speed'].dropna()
                if len(speed_data) > 0:
                    print(f"  Mean Speed:      {speed_data.mean():.4f} m/s")
                    print(f"  Max Speed:       {completed_df['max_speed'].max():.4f} m/s")
                    print(f"  Min Speed:       {completed_df['min_speed'].min():.4f} m/s")
            
            # Raceline tracking
            if 'avg_distance_to_raceline' in completed_df.columns:
                print(f"\n{'='*80}")
                print("RACELINE TRACKING")
                print(f"{'─'*80}")
                distance_data = completed_df['avg_distance_to_raceline'].dropna()
                if len(distance_data) > 0:
                    print(f"  Mean Distance to Line:  {distance_data.mean():.4f} m")
                    print(f"  Max Deviation:          {completed_df['max_distance_to_raceline'].max():.4f} m")
            
            # Control behavior
            if 'avg_steering_angle' in completed_df.columns:
                print(f"\n{'='*80}")
                print("CONTROL BEHAVIOR")
                print(f"{'─'*80}")
                steering = completed_df['avg_steering_angle'].dropna()
                accel = completed_df['avg_acceleration'].dropna()
                if len(steering) > 0:
                    print(f"  Avg Steering:    {steering.mean():.4f}")
                    print(f"  Max Steering:    {completed_df['max_steering_angle'].max():.4f}")
                if len(accel) > 0:
                    print(f"  Avg Acceleration: {accel.mean():.4f}")
                    print(f"  Max Acceleration: {completed_df['max_acceleration'].max():.4f}")
            
            # Find best and worst models
            best_idx = completed_df['avg_lap_time'].idxmin()
            worst_idx = completed_df['avg_lap_time'].idxmax()
            print(f"\n{'='*80}")
            print("BEST & WORST MODELS")
            print(f"{'─'*80}")
            print(f"Best:  {completed_df.loc[best_idx, 'model_name']}")
            print(f"       Avg Lap Time: {completed_df.loc[best_idx, 'avg_lap_time']:.4f}s")
            print(f"\nWorst: {completed_df.loc[worst_idx, 'model_name']}")
            print(f"       Avg Lap Time: {completed_df.loc[worst_idx, 'avg_lap_time']:.4f}s")
            
            # Safety
            crashes = (completed_df['crash_occurred'] == 'True').sum() + (completed_df['crash_occurred'] == True).sum()
            print(f"\nModels with crashes: {crashes}")
        
        print("\n" + "="*80)
    
    def plot_results(self, df: pd.DataFrame, prefix: str) -> None:
        """Create comprehensive visualization of results with multiple metrics."""
        if not HAS_MATPLOTLIB:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        completed_df = df[df['status'] == 'completed'].copy()
        if completed_df.empty:
            print("No completed experiments to plot")
            return
        
        # Sort by lap time for better visualization
        completed_df = completed_df.sort_values('avg_lap_time')
        x_pos = np.arange(len(completed_df))
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Comprehensive Sweep Analysis: {prefix}', fontsize=16, fontweight='bold')
        
        # Plot 1: Average lap times (primary metric)
        ax = axes[0, 0]
        colors = ['red' if crash else 'blue' for crash in completed_df['crash_occurred']]
        ax.barh(x_pos, completed_df['avg_lap_time'], color=colors, alpha=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels([m[:25] + '...' if len(m) > 25 else m 
                            for m in completed_df['model_name']], fontsize=7)
        ax.set_xlabel('Average Lap Time (s)')
        ax.set_title('Average Lap Times by Model')
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 2: Speed analysis (if available)
        ax = axes[0, 1]
        if 'avg_speed' in completed_df.columns and not completed_df['avg_speed'].isna().all():
            speed_available = completed_df[completed_df['avg_speed'].notna()].copy()
            speed_x = np.arange(len(speed_available))
            ax.bar(speed_x, speed_available['avg_speed'], alpha=0.6, label='Avg', color='blue')
            if 'max_speed' in speed_available.columns:
                ax.scatter(speed_x, speed_available['max_speed'], color='red', s=30, label='Max', alpha=0.7)
            ax.set_xticks(speed_x)
            ax.set_xticklabels([m[:15] + '...' if len(m) > 15 else m 
                                for m in speed_available['model_name']], rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Speed (m/s)')
            ax.set_title('Speed Statistics')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Speed data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Speed Statistics (N/A)')
        
        # Plot 3: Raceline tracking (if available)
        ax = axes[1, 0]
        if 'avg_distance_to_raceline' in completed_df.columns and not completed_df['avg_distance_to_raceline'].isna().all():
            dist_available = completed_df[completed_df['avg_distance_to_raceline'].notna()].copy()
            dist_x = np.arange(len(dist_available))
            ax.bar(dist_x, dist_available['avg_distance_to_raceline'], alpha=0.6, color='green')
            ax.set_xticks(dist_x)
            ax.set_xticklabels([m[:15] + '...' if len(m) > 15 else m 
                                for m in dist_available['model_name']], rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Distance to Raceline (m)')
            ax.set_title('Raceline Tracking (lower is better)')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Raceline data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Raceline Tracking (N/A)')
        
        # Plot 4: Lap consistency (if available)
        ax = axes[1, 1]
        if 'lap_consistency' in completed_df.columns and not completed_df['lap_consistency'].isna().all():
            consistency_available = completed_df[completed_df['lap_consistency'].notna()].copy()
            cons_x = np.arange(len(consistency_available))
            colors_cons = ['green' if c > 0.8 else 'orange' if c > 0.6 else 'red' 
                          for c in consistency_available['lap_consistency']]
            ax.bar(cons_x, consistency_available['lap_consistency'], color=colors_cons, alpha=0.7)
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High (>0.8)')
            ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Medium (0.6-0.8)')
            ax.set_xticks(cons_x)
            ax.set_xticklabels([m[:15] + '...' if len(m) > 15 else m 
                                for m in consistency_available['model_name']], rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Consistency Score (0-1)')
            ax.set_ylim([0, 1.1])
            ax.set_title('Lap Consistency (higher is better)')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Consistency data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Lap Consistency (N/A)')
        
        # Plot 5: Lap time distribution (min/avg/max)
        ax = axes[2, 0]
        if not completed_df['min_lap_time'].isna().all():
            indices = np.arange(len(completed_df))
            ax.scatter(completed_df['min_lap_time'], indices, label='Min', alpha=0.6, s=30, color='green')
            ax.scatter(completed_df['avg_lap_time'], indices, label='Avg', alpha=0.8, s=50, color='blue')
            ax.scatter(completed_df['max_lap_time'], indices, label='Max', alpha=0.6, s=30, color='red')
            ax.set_yticks(indices)
            ax.set_yticklabels([m[:20] + '...' if len(m) > 20 else m 
                                for m in completed_df['model_name']], fontsize=7)
            ax.set_xlabel('Lap Time (s)')
            ax.set_title('Lap Time Range (Min/Avg/Max)')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 6: Summary statistics text
        ax = axes[2, 1]
        ax.axis('off')
        
        # Build summary text with available metrics
        summary_lines = [
            "SUMMARY STATISTICS",
            "─" * 40,
            f"Total models: {len(df)}",
            f"Completed: {len(completed_df)}",
            f"",
            "LAP TIMES:",
            f"  Mean: {completed_df['avg_lap_time'].mean():.4f}s",
            f"  Median: {completed_df['avg_lap_time'].median():.4f}s",
            f"  Std Dev: {completed_df['avg_lap_time'].std():.4f}s",
            f"  Range: {completed_df['avg_lap_time'].min():.4f}s - {completed_df['avg_lap_time'].max():.4f}s",
        ]
        
        if 'avg_speed' in completed_df.columns:
            speed_data = completed_df['avg_speed'].dropna()
            if len(speed_data) > 0:
                summary_lines.extend([
                    f"",
                    "SPEED:",
                    f"  Mean: {speed_data.mean():.4f} m/s",
                    f"  Peak: {completed_df['max_speed'].max():.4f} m/s",
                ])
        
        if 'lap_consistency' in completed_df.columns:
            cons_data = completed_df['lap_consistency'].dropna()
            if len(cons_data) > 0:
                summary_lines.extend([
                    f"",
                    "CONSISTENCY:",
                    f"  Mean: {cons_data.mean():.4f}",
                    f"  Best: {cons_data.max():.4f}",
                ])
        
        crashes = (completed_df['crash_occurred'].astype(str).str.lower() == 'true').sum() + \
                  (completed_df['crash_occurred'] == True).sum()
        summary_lines.extend([
            f"",
            f"Crashes: {crashes}",
            f"Failed: {len(df) - len(completed_df)}",
        ])
        
        stats_text = "\n".join(summary_lines)
        ax.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.results_dir / f"sweep_{prefix}_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        
        # Try to show (may fail in headless environments)
        try:
            plt.show()
        except:
            pass
    
    def export_detailed_stats(self, df: pd.DataFrame, prefix: str) -> None:
        """Export comprehensive statistics to a text file."""
        output_file = self.results_dir / f"sweep_{prefix}_analysis.txt"
        
        with open(output_file, 'w') as f:
            f.write("DETAILED SWEEP RESULTS ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            total = len(df)
            successful = (df['status'] == 'completed').sum()
            f.write(f"Total Models: {total}\n")
            f.write(f"Completed: {successful}\n")
            f.write(f"Failed: {total - successful}\n\n")
            
            # All statistics
            completed_df = df[df['status'] == 'completed']
            if not completed_df.empty:
                # Lap time stats
                f.write("LAP TIME STATISTICS\n")
                f.write("-"*80 + "\n")
                f.write(f"Mean:        {completed_df['avg_lap_time'].mean():.4f}s\n")
                f.write(f"Median:      {completed_df['avg_lap_time'].median():.4f}s\n")
                f.write(f"Std Dev:     {completed_df['avg_lap_time'].std():.4f}s\n")
                f.write(f"Min:         {completed_df['avg_lap_time'].min():.4f}s\n")
                f.write(f"Max:         {completed_df['avg_lap_time'].max():.4f}s\n")
                f.write(f"Range:       {completed_df['avg_lap_time'].max() - completed_df['avg_lap_time'].min():.4f}s\n\n")
                
                # Speed stats
                if 'avg_speed' in completed_df.columns:
                    speed_data = completed_df['avg_speed'].dropna()
                    if len(speed_data) > 0:
                        f.write("SPEED STATISTICS\n")
                        f.write("-"*80 + "\n")
                        f.write(f"Mean Speed:           {speed_data.mean():.4f} m/s\n")
                        f.write(f"Median Speed:         {speed_data.median():.4f} m/s\n")
                        f.write(f"Max Speed (peak):     {completed_df['max_speed'].max():.4f} m/s\n")
                        f.write(f"Min Speed (lowest):   {completed_df['min_speed'].min():.4f} m/s\n\n")
                
                # Raceline tracking
                if 'avg_distance_to_raceline' in completed_df.columns:
                    distance_data = completed_df['avg_distance_to_raceline'].dropna()
                    if len(distance_data) > 0:
                        f.write("RACELINE TRACKING\n")
                        f.write("-"*80 + "\n")
                        f.write(f"Mean Distance to Line:      {distance_data.mean():.4f} m\n")
                        f.write(f"Median Distance to Line:    {distance_data.median():.4f} m\n")
                        f.write(f"Worst Deviation (max):      {completed_df['max_distance_to_raceline'].max():.4f} m\n")
                        f.write(f"Best Line Tracking (min):   {completed_df['max_distance_to_raceline'].min():.4f} m\n\n")
                
                # Control behavior
                if 'avg_steering_angle' in completed_df.columns:
                    steering_data = completed_df['avg_steering_angle'].dropna()
                    accel_data = completed_df['avg_acceleration'].dropna()
                    if len(steering_data) > 0 or len(accel_data) > 0:
                        f.write("CONTROL BEHAVIOR\n")
                        f.write("-"*80 + "\n")
                        if len(steering_data) > 0:
                            f.write(f"Avg Steering Angle:     {steering_data.mean():.4f}\n")
                            f.write(f"Max Steering (peak):    {completed_df['max_steering_angle'].max():.4f}\n")
                        if len(accel_data) > 0:
                            f.write(f"Avg Acceleration:       {accel_data.mean():.4f}\n")
                            f.write(f"Max Acceleration:       {completed_df['max_acceleration'].max():.4f}\n")
                        f.write("\n")
                
                # Lap consistency
                if 'lap_consistency' in completed_df.columns:
                    consistency_data = completed_df['lap_consistency'].dropna()
                    if len(consistency_data) > 0:
                        f.write("LAP CONSISTENCY (0-1, higher is better)\n")
                        f.write("-"*80 + "\n")
                        f.write(f"Mean Consistency:       {consistency_data.mean():.4f}\n")
                        f.write(f"Best Consistency:       {consistency_data.max():.4f}\n")
                        f.write(f"Worst Consistency:      {consistency_data.min():.4f}\n\n")
                
                # Top 10 models
                f.write("TOP 10 MODELS (by average lap time)\n")
                f.write("-"*80 + "\n")
                top_10 = completed_df.nsmallest(10, 'avg_lap_time')
                name_width = int(max(top_10['model_name'].astype(str).str.len().max(), 20))
                for i, (_, row) in enumerate(top_10.iterrows(), 1):
                    speed_str = ""
                    if pd.notna(row.get('avg_speed')):
                        speed_str = f"  Speed: {row['avg_speed']:.2f} m/s"
                    consistency_str = ""
                    if pd.notna(row.get('lap_consistency')):
                        consistency_str = f"  Consistency: {row['lap_consistency']:.3f}"
                    model_name = str(row['model_name'])
                    f.write(f"{i:2d}. {model_name:<{name_width}}  {row['avg_lap_time']:8.4f}s{speed_str}{consistency_str}\n")
                
                # Bottom 10 models
                f.write("\nBOTTOM 10 MODELS (worst performance)\n")
                f.write("-"*80 + "\n")
                bottom_10 = completed_df.nlargest(10, 'avg_lap_time')
                bottom_name_width = int(max(bottom_10['model_name'].astype(str).str.len().max(), 20))
                for i, (_, row) in enumerate(bottom_10.iterrows(), 1):
                    model_name = str(row['model_name'])
                    f.write(f"{i:2d}. {model_name:<{bottom_name_width}}  {row['avg_lap_time']:8.4f}s\n")

            # Grouped averages across repeated runs (e.g., Run1/Run2/Run3)
            # Keep all runs in groups and explicitly track failed-to-complete-laps counts.
            grouped_source = df.copy()
            grouped_source['model_group'] = grouped_source['model_name'].apply(self._get_group_key)
            grouped_source['num_laps_completed_numeric'] = pd.to_numeric(
                grouped_source.get('num_laps_completed', pd.Series(dtype=float)), errors='coerce'
            )
            grouped_source['failed_to_complete_laps'] = (
                (grouped_source['status'] != 'completed') |
                (grouped_source['num_laps_completed_numeric'].fillna(0) <= 0)
            )

            agg_columns = [
                'num_laps_completed',
                'avg_lap_time', 'min_lap_time', 'max_lap_time', 'std_lap_time', 'lap_consistency',
                'avg_speed', 'max_speed', 'min_speed', 'std_speed',
                'avg_distance_to_raceline', 'max_distance_to_raceline', 'std_distance_to_raceline',
                'avg_steering_angle', 'max_steering_angle', 'steering_smoothness',
                'avg_acceleration', 'max_acceleration', 'acceleration_smoothness',
                'total_sim_time', 'avg_time_per_lap'
            ]
            agg_columns = [c for c in agg_columns if c in grouped_source.columns]

            group_stats = grouped_source.groupby('model_group').agg(
                run_count=('model_name', 'count'),
                failed_run_count=('failed_to_complete_laps', 'sum'),
                **{col: (col, 'mean') for col in agg_columns}
            ).reset_index()

            grouped_min_runs = 3
            group_stats = group_stats[group_stats['run_count'] >= grouped_min_runs]
            if 'avg_lap_time' in group_stats.columns:
                group_stats = group_stats.sort_values('avg_lap_time')
            else:
                group_stats = group_stats.sort_values('model_group')

            f.write(f"\nAVERAGED SETTINGS GROUPS (Run repeats, n >= {grouped_min_runs})\n")
            f.write("-"*80 + "\n")
            if group_stats.empty:
                f.write("None\n")
            else:
                for _, row in group_stats.iterrows():
                    f.write(f"{row['model_group']}\n")
                    f.write(f"  Runs Averaged: {int(row['run_count'])}\n")
                    f.write(f"  Runs Failed to Complete Laps: {int(row['failed_run_count'])}\n")

                    if pd.notna(row.get('num_laps_completed')):
                        f.write(f"  Avg Laps Completed: {row['num_laps_completed']:.2f}\n")

                    if pd.notna(row.get('avg_lap_time')):
                        f.write(f"  Avg Lap Time: {row['avg_lap_time']:.4f}s\n")
                    if pd.notna(row.get('min_lap_time')) and pd.notna(row.get('max_lap_time')):
                        f.write(f"  Avg Min/Max Lap Time: {row['min_lap_time']:.4f}s / {row['max_lap_time']:.4f}s\n")
                    if pd.notna(row.get('std_lap_time')):
                        f.write(f"  Avg Lap Std Dev: {row['std_lap_time']:.4f}s\n")
                    if pd.notna(row.get('lap_consistency')):
                        f.write(f"  Avg Consistency: {row['lap_consistency']:.4f}\n")

                    if pd.notna(row.get('avg_speed')):
                        f.write(f"  Avg Speed: {row['avg_speed']:.4f} m/s")
                        if pd.notna(row.get('max_speed')):
                            f.write(f", Avg Peak Speed: {row['max_speed']:.4f} m/s")
                        f.write("\n")

                    if pd.notna(row.get('avg_distance_to_raceline')):
                        f.write(f"  Avg Distance to Line: {row['avg_distance_to_raceline']:.4f} m")
                        if pd.notna(row.get('max_distance_to_raceline')):
                            f.write(f", Avg Max Deviation: {row['max_distance_to_raceline']:.4f} m")
                        f.write("\n")

                    if pd.notna(row.get('avg_steering_angle')):
                        f.write(f"  Avg Steering: {row['avg_steering_angle']:.4f}")
                        if pd.notna(row.get('max_steering_angle')):
                            f.write(f", Avg Max Steering: {row['max_steering_angle']:.4f}")
                        if pd.notna(row.get('steering_smoothness')):
                            f.write(f", Avg Steering Smoothness: {row['steering_smoothness']:.4f}")
                        f.write("\n")

                    if pd.notna(row.get('avg_acceleration')):
                        f.write(f"  Avg Acceleration: {row['avg_acceleration']:.4f}")
                        if pd.notna(row.get('max_acceleration')):
                            f.write(f", Avg Max Acceleration: {row['max_acceleration']:.4f}")
                        if pd.notna(row.get('acceleration_smoothness')):
                            f.write(f", Avg Acceleration Smoothness: {row['acceleration_smoothness']:.4f}")
                        f.write("\n")

                    if pd.notna(row.get('total_sim_time')):
                        f.write(f"  Avg Total Sim Time: {row['total_sim_time']:.4f}s\n")
                    if pd.notna(row.get('avg_time_per_lap')):
                        f.write(f"  Avg Time Per Lap: {row['avg_time_per_lap']:.4f}s\n")

                    f.write("\n")

            # Models that failed to complete laps (failed status OR 0/NaN completed laps)
            f.write("\nMODELS THAT FAILED TO COMPLETE LAPS\n")
            f.write("-"*80 + "\n")
            laps_completed = pd.to_numeric(df.get('num_laps_completed', pd.Series(dtype=float)), errors='coerce')
            failed_lap_df = df[(df['status'] != 'completed') | (laps_completed.fillna(0) <= 0)]
            if failed_lap_df.empty:
                f.write("None\n")
            else:
                for i, (_, row) in enumerate(failed_lap_df.iterrows(), 1):
                    model_name = str(row.get('model_name', 'UNKNOWN_MODEL'))
                    status = str(row.get('status', 'unknown'))
                    laps = row.get('num_laps_completed', 'N/A')
                    error_message = str(row.get('error_message', '')).strip()
                    f.write(f"{i:2d}. {model_name}\n")
                    f.write(f"    Status: {status}, Laps Completed: {laps}\n")
                    if error_message and error_message.lower() != 'nan':
                        f.write(f"    Error: {error_message}\n")
        
        print(f"Detailed analysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sweep experiment results"
    )
    parser.add_argument(
        "--prefix",
        help="Model name prefix (will find corresponding results file)"
    )
    parser.add_argument(
        "--results-file",
        help="Direct path to results CSV file"
    )
    parser.add_argument(
        "--results-dir",
        default="sweep_experiment_results",
        help="Directory containing results (default: sweep_experiment_results)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib)"
    )
    
    args = parser.parse_args()
    
    analyzer = SweepResultsAnalyzer(results_dir=args.results_dir)
    
    # Find results file
    if args.results_file:
        results_file = Path(args.results_file)
    elif args.prefix:
        results_file = analyzer.find_results_file(args.prefix)
    else:
        print("Error: Must provide either --prefix or --results-file")
        parser.print_help()
        sys.exit(1)
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"Loading results from: {results_file}")
    
    # Load and analyze
    if HAS_MATPLOTLIB:
        df = analyzer.load_results(results_file)
        prefix = args.prefix or results_file.stem.replace('sweep_', '').replace('_results', '')
        
        analyzer.print_summary(df)
        analyzer.export_detailed_stats(df, prefix)
        
        if args.plot:
            analyzer.plot_results(df, prefix)
    else:
        print("\nNote: pandas/matplotlib required for full analysis.")
        print("Install with: pip install pandas matplotlib")


if __name__ == "__main__":
    main()
