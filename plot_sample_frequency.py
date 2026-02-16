"""
Plot sample frequency statistics from StatTracker CSV:
1. Sample frequency distribution (how often each transition was sampled)
2. Spatial heatmap on race track showing sample frequency by position
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import os
import yaml
from argparse import ArgumentParser
from PIL import Image

# ============================================================================
# CONFIGURATION - Edit these paths directly
# ============================================================================
DEFAULT_CSV_PATH = 'TrainingLite/rl_racing/models/laptop_pc_custom_TD_error_only_5s/stat_logs/stats_log.csv'
DEFAULT_MAP_NAME = 'RCA1'
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

def compute_sample_frequency(df):
    """
    Compute samples per batch for each transition.
    
    samples_per_batch = sample_count / possible_samples
    where possible_samples = sample_calls_at_death - sample_calls_at_birth (if transition died)
                          = total_sample_calls - sample_calls_at_birth (if transition stayed until end)
    
    Note: Can be >1 when sampling with replacement (transition appears multiple times per batch)
    """
    # Get total sample calls from the maximum sample_calls_at_death value
    # This represents the final sample call count when training ended
    if 'sample_calls_at_death' in df.columns:
        # Use max of sample_calls_at_death (ignoring NaN) as the total
        total_sample_calls = df['sample_calls_at_death'].max()
        if pd.isna(total_sample_calls):
            # All transitions are still alive, use max birth time as fallback
            total_sample_calls = df['sample_calls_at_birth'].max()
    else:
        # Fallback if column doesn't exist
        total_sample_calls = df['sample_calls_at_birth'].max()
    
    # For transitions that died: use their death time
    # For transitions still alive (NaN death): use total_sample_calls
    df['sample_calls_end'] = df['sample_calls_at_death'].fillna(total_sample_calls)
    df['possible_samples'] = df['sample_calls_end'] - df['sample_calls_at_birth']
    
    # Avoid division by zero
    df['possible_samples'] = df['possible_samples'].clip(lower=1)
    
    # Compute samples per batch (can be >1 if sampling with replacement)
    df['samples_per_batch'] = df['sample_count'] / df['possible_samples']
    
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
                        alpha=0.5, s=10, c=df['possible_samples'], 
                        cmap='viridis')
    ax.set_xlabel('Absolute Sample Count')
    ax.set_ylabel('Samples per Batch')
    ax.set_title('Sample Count vs Samples per Batch')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Possible Samples')
    
    # 3. Samples per batch over transition lifetime (time in buffer)
    ax = axes[1, 0]
    ax.scatter(df['possible_samples'], df['samples_per_batch'], alpha=0.5, s=10)
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
    
    # Flip y-coordinates for proper display
    img_height = img_array.shape[0]
    scatter = ax.scatter(
        df_with_pos['pixel_x'],
        img_height - df_with_pos['pixel_y'],
        c=df_with_pos['samples_per_batch'],
        cmap='viridis',  # Dark blue (low) -> cyan -> green -> yellow (high)
        s=3,  # Much smaller dots to see individual points
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
    args = parser.parse_args()
    
    # Load CSV
    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} transitions")
    
    # Compute sample frequency
    df = compute_sample_frequency(df)
    
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
