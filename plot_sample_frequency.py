"""
Plot sample frequency statistics from StatTracker CSV:
1. Sample frequency distribution (how often each transition was sampled)
2. Spatial heatmap on race track showing sample frequency by position
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm, Normalize
import pandas as pd
import numpy as np
import os
import yaml
from argparse import ArgumentParser
from PIL import Image

# ============================================================================
# CONFIGURATION - Edit these paths directly
# ============================================================================
# DEFAULT_CSV_PATH = 'TrainingLite/rl_racing/models/1502_pc_custom_uniform_10s/stat_logs/stats_log.csv'
DEFAULT_CSV_PATH = 'TrainingLite/rl_racing/models/Nachtrainiert/1502_from_Ex1_A0.4_Rank_True_Run2/stat_logs/stats_log.csv'
DEFAULT_MAP_NAME = 'RCA2'
DEFAULT_TOTAL_SAMPLE_CALLS = None  # Total sample calls if not in CSV
DEFAULT_MIN_POSSIBLE_SAMPLES_PERCENTILE = 5  # Drop bottom percentile of possible_samples
# ============================================================================

def load_map_image(map_name='RCA1'):
    """Load the race track map image and configuration"""
    map_path = os.path.join('utilities', 'maps', map_name)
    yaml_path = os.path.join(map_path, f'{map_name}.yaml')
    img_path = os.path.join(map_path, f'{map_name}.png')
    
    # Load yaml config for origin and resolution
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load map image
    img = Image.open(img_path)
    img_array = np.array(img)
    
    return img_array, config

def world_to_pixel(x, y, origin, resolution):
    """Convert world coordinates to pixel coordinates"""
    pixel_x = int((x - origin[0]) / resolution)
    pixel_y = int((y - origin[1]) / resolution)
    # Flip y-axis (image origin is top-left, world origin is bottom-left)
    return pixel_x, pixel_y

def compute_total_sample_calls_from_learning_metrics(csv_path):
    """
    Compute total sample calls from learning_metrics.csv.
    Takes the final total_weight_updates and adds batch size (256 or 512).
    
    Returns the computed total or None if file not found.
    """
    try:
        # Get parent directory of stats_log.csv (go up one level from stat_logs folder)
        parent_dir = os.path.dirname(os.path.dirname(csv_path))
        learning_metrics_path = os.path.join(parent_dir, 'learning_metrics.csv')
        
        if not os.path.exists(learning_metrics_path):
            return None
        
        # Read learning metrics CSV
        lm_df = pd.read_csv(learning_metrics_path)
        
        # Get the final total_weight_updates value
        if 'total_weight_updates' not in lm_df.columns:
            return None
        
        final_weight_updates = lm_df['total_weight_updates'].iloc[-1]
        
        # Get batch size from the last row (typically 256)
        batch_size = lm_df['batch_size'].iloc[-1] if 'batch_size' in lm_df.columns else 256
        
        # Add batch size to weight updates to get total sample calls
        total_sample_calls = int(final_weight_updates + batch_size)
        
        print(f"Computed total_sample_calls from learning_metrics.csv: {total_sample_calls}")
        print(f"  (final_weight_updates={final_weight_updates} + batch_size={batch_size})")
        
        return total_sample_calls
    except Exception as e:
        print(f"Warning: Could not compute total_sample_calls from learning_metrics.csv: {e}")
        return None

def compute_sample_frequency(
    df,
    csv_path=None,
    total_sample_calls=DEFAULT_TOTAL_SAMPLE_CALLS,
    min_possible_samples_percentile=DEFAULT_MIN_POSSIBLE_SAMPLES_PERCENTILE,
):
    """
    Compute samples per batch for each transition.
    
    samples_per_batch = sample_count / possible_samples
    where possible_samples = sample_calls_at_death - sample_calls_at_birth (if transition died)
                          = total_sample_calls - sample_calls_at_birth (if transition stayed until end)
    
    Note: Can be >1 when sampling with replacement (transition appears multiple times per batch)
    """
    # Try to compute from learning_metrics.csv first
    if csv_path:
        computed_total = compute_total_sample_calls_from_learning_metrics(csv_path)
        if computed_total is not None:
            total_sample_calls = computed_total
    
    # Get total sample calls from the CSV if available, otherwise use provided value
    if 'sample_calls_at_death' in df.columns and not df['sample_calls_at_death'].isna().all():
        # Use max of sample_calls_at_death (ignoring NaN) as the total
        csv_total = df['sample_calls_at_death'].max()
        if not pd.isna(csv_total):
            total_sample_calls = csv_total

    # For transitions that died: use their death time
    # For transitions still alive (NaN death): use total_sample_calls
    df['sample_calls_end'] = df['sample_calls_at_death'].fillna(total_sample_calls)
    df['possible_samples'] = df['sample_calls_end'] - df['sample_calls_at_birth']
    
    # Avoid division by zero
    df['possible_samples'] = df['possible_samples'].clip(lower=1)
    
    # Compute samples per batch (can be >1 if sampling with replacement)
    df['samples_per_batch'] = df['sample_count'] / df['possible_samples']

    # Optional: filter out bottom percentile of possible_samples
    if min_possible_samples_percentile is not None:
        percentile = min_possible_samples_percentile / 100.0
        if 0 < percentile < 1:
            cutoff = df['possible_samples'].quantile(percentile)
            df = df[df['possible_samples'] >= cutoff].copy()
            print(
                f"Filtered transitions with possible_samples < {cutoff:.2f} "
                f"(bottom {min_possible_samples_percentile:.1f}%)"
            )
    
    return df

def plot_sample_frequency_distribution(df, save_path=None):
    """Plot 1: Distribution of sample frequencies"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of samples per batch
    ax = axes[0, 0]
    ax.hist(df['samples_per_batch'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(df['samples_per_batch'].mean(), color='r', linestyle='--', 
               label=f'Mean: {df["samples_per_batch"].mean():.3f}')
    ax.axvline(df['samples_per_batch'].median(), color='g', linestyle='--',
               label=f'Median: {df["samples_per_batch"].median():.3f}')
    ax.set_xlabel('Samples per Batch (sample_count / possible_samples)')
    ax.set_ylabel('Number of Transitions')
    ax.set_title('Distribution of Samples per Batch')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Sample count vs samples per batch
    ax = axes[0, 1]
    scatter = ax.scatter(df['sample_count'], df['samples_per_batch'], 
                        alpha=0.5, s=1, c=df['possible_samples'], 
                        cmap='viridis')
    ax.set_xlabel('Absolute Sample Count')
    ax.set_ylabel('Samples per Batch')
    ax.set_title('Sample Count vs Samples per Batch')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Possible Samples')
    
    # 3. Samples per batch over transition lifetime (time in buffer)
    ax = axes[1, 0]
    ax.scatter(df['possible_samples'], df['samples_per_batch'], alpha=0.5, s=1)
    ax.set_xlabel('Transition Lifetime (possible samples)')
    ax.set_ylabel('Samples per Batch')
    ax.set_title('Samples per Batch vs Lifetime in Buffer')
    ax.grid(alpha=0.3)
    
    # 4. Top vs bottom samples per batch
    ax = axes[1, 1]
    top_percentile = df['samples_per_batch'].quantile(0.9)
    bottom_percentile = df['samples_per_batch'].quantile(0.1)
    
    categories = ['Bottom 10%', 'Middle 80%', 'Top 10%']
    counts = [
        (df['samples_per_batch'] <= bottom_percentile).sum(),
        ((df['samples_per_batch'] > bottom_percentile) & 
         (df['samples_per_batch'] < top_percentile)).sum(),
        (df['samples_per_batch'] >= top_percentile).sum()
    ]
    colors = ['red', 'gray', 'green']
    
    ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Transitions')
    ax.set_title('Samples per Batch Distribution')
    ax.grid(alpha=0.3, axis='y')
    
    # Add percentages
    total = len(df)
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, count, f'{count}\n({100*count/total:.1f}%)', 
               ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample frequency distribution to {save_path}")
    
    plt.show()

def plot_spatial_heatmap(df, map_name='RCA1', save_path=None):
    """Plot 2: Spatial heatmap of sample frequencies on race track"""
    # Load map
    img_array, config = load_map_image(map_name)
    origin = config['origin']
    resolution = config['resolution']
    
    # Filter out transitions without position data
    df_with_pos = df.dropna(subset=['pose_x', 'pose_y'])
    
    if len(df_with_pos) == 0:
        print("Warning: No position data found in CSV. Skipping spatial heatmap.")
        return
    
    # Convert world coordinates to pixel coordinates
    pixel_coords = []
    for _, row in df_with_pos.iterrows():
        px, py = world_to_pixel(row['pose_x'], row['pose_y'], origin, resolution)
        pixel_coords.append((px, py))
    
    df_with_pos['pixel_x'] = [c[0] for c in pixel_coords]
    df_with_pos['pixel_y'] = [c[1] for c in pixel_coords]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Track overlay with sample frequency coloring
    ax = axes[0]
    ax.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)  # Fade map to reduce border visibility
    
    # Sort by samples_per_batch so higher values are drawn on top
    df_sorted = df_with_pos.sort_values('samples_per_batch')
    
    # Flip y-coordinates for proper display
    img_height = img_array.shape[0]
    scatter = ax.scatter(
        df_sorted['pixel_x'],
        img_height - df_sorted['pixel_y'],
        c=df_sorted['samples_per_batch'],
        cmap='viridis',  # Dark blue (low) -> cyan -> green -> yellow (high)
        s=0.5,  # Much smaller dots to see individual points
        alpha=0.6,
        vmin=0,
        vmax=df_with_pos['samples_per_batch'].max()  # Scale to actual max
    )
    
    ax.set_title(f'Samples per Batch Heatmap on {map_name}')
    ax.axis('off')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Samples per Batch')
    
    # Right: 2D histogram heatmap
    ax = axes[1]
    ax.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)
    
    # Create 2D histogram
    x_bins = np.linspace(df_with_pos['pixel_x'].min(), 
                         df_with_pos['pixel_x'].max(), 50)
    y_bins = np.linspace(df_with_pos['pixel_y'].min(), 
                         df_with_pos['pixel_y'].max(), 50)
    
    # Compute weighted histogram (weighted by samples per batch)
    H, xedges, yedges = np.histogram2d(
        df_with_pos['pixel_x'],
        df_with_pos['pixel_y'],
        bins=[x_bins, y_bins],
        weights=df_with_pos['samples_per_batch']
    )
    
    # Count histogram for normalization
    H_count, _, _ = np.histogram2d(
        df_with_pos['pixel_x'],
        df_with_pos['pixel_y'],
        bins=[x_bins, y_bins]
    )
    
    # Normalize by count to get average samples per batch per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = H / H_count
        H_avg[~np.isfinite(H_avg)] = 0
    
    # Flip y-axis for display
    H_avg = np.flipud(H_avg.T)
    
    im = ax.imshow(
        H_avg,
        extent=[xedges[0], xedges[-1], 
                img_height - yedges[-1], img_height - yedges[0]],
        cmap='viridis',  # Dark blue (low) -> cyan -> green -> yellow (high)
        alpha=0.7,
        origin='upper'
    )
    
    ax.set_title('Aggregated Samples per Batch Heatmap')
    ax.axis('off')
    cbar2 = plt.colorbar(im, ax=ax)
    cbar2.set_label('Avg Samples per Batch')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial heatmap to {save_path}")
    
    plt.show()

    # Optional: Reward-based heatmaps
    if 'reward' not in df_with_pos.columns:
        print("Warning: No 'reward' column found. Skipping reward heatmap.")
        return

    df_reward = df_with_pos.dropna(subset=['reward'])
    if len(df_reward) == 0:
        print("Warning: No reward data found. Skipping reward heatmap.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Track overlay with reward coloring
    ax = axes[0]
    ax.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)

    # Sort by reward so higher values are drawn on top
    df_reward_sorted = df_reward.sort_values('reward')

    img_height = img_array.shape[0]
    reward_min = df_reward['reward'].min()
    reward_max = df_reward['reward'].max()
    if reward_min < 0 < reward_max:
        reward_norm = TwoSlopeNorm(vmin=reward_min, vcenter=0.0, vmax=reward_max)
    else:
        reward_norm = Normalize(vmin=reward_min, vmax=reward_max)
    scatter = ax.scatter(
        df_reward_sorted['pixel_x'],
        img_height - df_reward_sorted['pixel_y'],
        c=df_reward_sorted['reward'],
        cmap='RdBu_r',
        s=0.5,
        alpha=0.6,
        norm=reward_norm
    )

    ax.set_title(f'Reward Heatmap on {map_name}')
    ax.axis('off')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reward')

    # Right: 2D histogram heatmap (average reward per bin)
    ax = axes[1]
    ax.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)

    x_bins = np.linspace(df_reward['pixel_x'].min(),
                         df_reward['pixel_x'].max(), 50)
    y_bins = np.linspace(df_reward['pixel_y'].min(),
                         df_reward['pixel_y'].max(), 50)

    H, xedges, yedges = np.histogram2d(
        df_reward['pixel_x'],
        df_reward['pixel_y'],
        bins=[x_bins, y_bins],
        weights=df_reward['reward']
    )

    H_count, _, _ = np.histogram2d(
        df_reward['pixel_x'],
        df_reward['pixel_y'],
        bins=[x_bins, y_bins]
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = H / H_count
        H_avg[~np.isfinite(H_avg)] = 0

    H_avg = np.flipud(H_avg.T)

    im = ax.imshow(
        H_avg,
        extent=[xedges[0], xedges[-1],
                img_height - yedges[-1], img_height - yedges[0]],
        cmap='RdBu_r',
        alpha=0.7,
        origin='upper',
        norm=reward_norm
    )

    ax.set_title('Aggregated Reward Heatmap')
    ax.axis('off')
    cbar2 = plt.colorbar(im, ax=ax)
    cbar2.set_label('Avg Reward')

    plt.tight_layout()
    plt.show()

def print_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SAMPLES PER BATCH STATISTICS")
    print("="*80)
    print(f"Total transitions: {len(df)}")
    print(f"Transitions with position data: {df[['pose_x', 'pose_y']].notna().all(axis=1).sum()}")
    print(f"\nSamples per Batch (can be >1 with replacement sampling):")
    print(f"  Mean:   {df['samples_per_batch'].mean():.4f}")
    print(f"  Median: {df['samples_per_batch'].median():.4f}")
    print(f"  Std:    {df['samples_per_batch'].std():.4f}")
    print(f"  Min:    {df['samples_per_batch'].min():.4f}")
    print(f"  Max:    {df['samples_per_batch'].max():.4f}")
    print(f"\nPercentiles:")
    print(f"  10th:   {df['samples_per_batch'].quantile(0.10):.4f}")
    print(f"  25th:   {df['samples_per_batch'].quantile(0.25):.4f}")
    print(f"  50th:   {df['samples_per_batch'].quantile(0.50):.4f}")
    print(f"  75th:   {df['samples_per_batch'].quantile(0.75):.4f}")
    print(f"  90th:   {df['samples_per_batch'].quantile(0.90):.4f}")
    print(f"  95th:   {df['samples_per_batch'].quantile(0.95):.4f}")
    print(f"  99th:   {df['samples_per_batch'].quantile(0.99):.4f}")
    print("="*80 + "\n")

def main():
    parser = ArgumentParser(description='Plot sample frequency statistics from StatTracker CSV')
    parser.add_argument('csv_path', type=str, nargs='?', default=DEFAULT_CSV_PATH,
                       help='Path to stats_log.csv file (default: set in script)')
    parser.add_argument('--map-name', type=str, default=DEFAULT_MAP_NAME, 
                       help='Map name for spatial heatmap (default: RCA1)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save plots (default: same as CSV)')
    parser.add_argument('--total-samples', type=int, default=DEFAULT_TOTAL_SAMPLE_CALLS,
                       help=f'Total sample calls if not in CSV (default: {DEFAULT_TOTAL_SAMPLE_CALLS})')
    parser.add_argument('--min-possible-samples-percentile', type=float,
                       default=DEFAULT_MIN_POSSIBLE_SAMPLES_PERCENTILE,
                       help='Drop bottom percentile of possible_samples (e.g. 5 for bottom 5%)')
    args = parser.parse_args()
    
    # Load CSV
    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} transitions")
    
    # Compute sample frequency (will try to read from learning_metrics.csv)
    df = compute_sample_frequency(
        df,
        csv_path=args.csv_path,
        total_sample_calls=args.total_samples,
        min_possible_samples_percentile=args.min_possible_samples_percentile,
    )
    
    # Print statistics
    print_statistics(df)
    
    # Determine save paths
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.csv_path)
    
    dist_save_path = os.path.join(args.save_dir, 'sample_frequency_distribution.png')
    heatmap_save_path = os.path.join(args.save_dir, 'sample_frequency_heatmap.png')
    
    # Plot 1: Sample frequency distribution
    print("\nPlotting sample frequency distribution...")
    plot_sample_frequency_distribution(df, save_path=dist_save_path)
    
    # Plot 2: Spatial heatmap
    print("\nPlotting spatial heatmap...")
    plot_spatial_heatmap(df, map_name=args.map_name, save_path=heatmap_save_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
