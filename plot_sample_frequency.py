"""
Plot sample frequency statistics from StatTracker CSV:
1. Sample frequency distribution (how often each transition was sampled)
2. Spatial heatmap on race track showing sample frequency by position
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import os
import yaml
from argparse import ArgumentParser
from PIL import Image

# ============================================================================
# CONFIGURATION - Edit these paths directly
# ============================================================================
DEFAULT_CSV_PATH = 'TrainingLite/rl_racing/models/2802_widsth_norank_04state/stat_logs/stats_log.csv'
# DEFAULT_CSV_PATH = 'TrainingLite/rl_racing/models/Nachtrainiert/2002_with_custom_07state_norank_slowdown_test/stat_logs/stats_log.csv'
DEFAULT_MAP_NAME = 'RCA1'
DEFAULT_TOTAL_SAMPLE_CALLS = None  # Total sample calls if not in CSV
DEFAULT_MIN_POSSIBLE_SAMPLES_PERCENTILE = 5  # Drop bottom percentile of possible_samples
ENABLE_TIME_SLIDER = True
TIME_SLIDER_COLUMN = 'timestamp'
TIME_SLIDER_FALLBACK_COLUMN = 'sample_calls_at_birth'
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

def prepare_time_data(df, enable_slider=ENABLE_TIME_SLIDER):
    """Prepare time data for slider if enabled. Returns (df, time_col) or (df, None)"""
    if not enable_slider:
        return df, None
    
    time_col = None
    if TIME_SLIDER_COLUMN in df.columns:
        time_col = TIME_SLIDER_COLUMN
        print(f"Using '{time_col}' for time slider")
        # Convert timestamp to numeric if it's a string
        if df[time_col].dtype == 'object':
            try:
                df = df.copy()
                df[time_col] = pd.to_datetime(df[time_col])
                df['time_numeric'] = (df[time_col] - df[time_col].min()).dt.total_seconds()
                time_col = 'time_numeric'
                print(f"Converted timestamp to numeric (seconds from start)")
            except Exception as e:
                print(f"Warning: Could not convert timestamp to datetime: {e}")
                print(f"Falling back to '{TIME_SLIDER_FALLBACK_COLUMN}'")
                time_col = TIME_SLIDER_FALLBACK_COLUMN if TIME_SLIDER_FALLBACK_COLUMN in df.columns else None
    elif TIME_SLIDER_FALLBACK_COLUMN in df.columns:
        time_col = TIME_SLIDER_FALLBACK_COLUMN
        print(f"'{TIME_SLIDER_COLUMN}' not found, using fallback '{time_col}' for time slider")
    
    if time_col is None or time_col not in df.columns:
        print(f"Warning: No valid time column found. Disabling slider.")
        return df, None
    
    return df, time_col

def plot_sample_frequency_distribution(df, save_path=None, enable_slider=ENABLE_TIME_SLIDER):
    """Plot 1: Distribution of sample frequencies"""
    # Prepare time data if slider enabled
    df, time_col = prepare_time_data(df, enable_slider)
    
    if enable_slider and time_col:
        plot_sample_frequency_distribution_interactive(df, time_col, save_path)
    else:
        plot_sample_frequency_distribution_static(df, save_path)

def plot_sample_frequency_distribution_static(df, save_path=None, global_stats=None):
    """Static version of sample frequency distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Use global stats if provided, otherwise use local data
    samples_max = global_stats['samples_per_batch_max'] if global_stats else df['samples_per_batch'].max()
    possible_samples_max = global_stats['possible_samples_max'] if global_stats else df['possible_samples'].max()
    
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
    ax.set_xlim(0, samples_max)  # Fixed x-axis scale
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Sample count vs samples per batch
    ax = axes[0, 1]
    scatter = ax.scatter(df['sample_count'], df['samples_per_batch'], 
                        alpha=0.5, s=1, c=df['possible_samples'], 
                        cmap='viridis', vmin=global_stats.get('possible_samples_min', 0) if global_stats else 0,
                        vmax=possible_samples_max)
    max_idx = df['sample_count'].idxmax()
    max_x = float(df.loc[max_idx, 'sample_count'])
    max_y = float(df.loc[max_idx, 'samples_per_batch'])
    ax.scatter([max_x], [max_y], marker='x', s=60, color='red', linewidths=1.5, zorder=5)
    ax.annotate(
        f"max x={max_x:.0f}",
        (max_x, max_y),
        textcoords="offset points",
        xytext=(-6, 8),
        ha='right',
        fontsize=8,
        color='red'
    )
    ax.set_xlabel('Absolute Sample Count')
    ax.set_ylabel('Samples per Batch')
    ax.set_ylim(0, samples_max)  # Fixed y-axis scale
    if global_stats and 'sample_count_max' in global_stats:
        ax.set_xlim(0, global_stats['sample_count_max'])  # Fixed x-axis scale
    ax.set_title('Sample Count vs Samples per Batch')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Possible Samples')
    
    # 3. Samples per batch over transition lifetime (time in buffer)
    ax = axes[1, 0]
    ax.scatter(df['possible_samples'], df['samples_per_batch'], alpha=0.5, s=1)
    ax.set_xlabel('Transition Lifetime (possible samples)')
    ax.set_ylabel('Samples per Batch')
    ax.set_ylim(0, samples_max)  # Fixed y-axis scale
    if global_stats:
        ax.set_xlim(global_stats['possible_samples_min'], possible_samples_max)  # Fixed x-axis scale
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
    
    # Use global max for y-axis if provided
    if global_stats:
        ax.set_ylim(0, global_stats['num_transitions_max'])
    
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

def plot_sample_frequency_distribution_interactive(df, time_col, save_path=None):
    """Interactive version of sample frequency distribution with time range slider"""
    # Get time range
    time_min = df[time_col].min()
    time_max = df[time_col].max()
    
    # Create figure with space for sliders (2 separate sliders)
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.05, 0.05, 1, 1, 0.05], hspace=0.25, wspace=0.25)
    
    # Create axes
    ax_slider_start = fig.add_subplot(gs[0, :])
    ax_slider_end = fig.add_subplot(gs[1, :])
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_scatter = fig.add_subplot(gs[2, 1])
    ax_lifetime = fig.add_subplot(gs[3, 0])
    ax_categories = fig.add_subplot(gs[3, 1])
    ax_text = fig.add_subplot(gs[4, :])
    ax_text.axis('off')
    
    # Track current time range
    current_range = [time_min, time_max]
    
    def filter_data():
        """Filter data to show only transitions in current time range"""
        return df[(df[time_col] >= current_range[0]) & (df[time_col] <= current_range[1])].copy()
    
    def format_time_text():
        """Format time range for display"""
        if time_col == 'time_numeric':
            def format_seconds(s):
                h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"Time range: {format_seconds(current_range[0])} - {format_seconds(current_range[1])}"
        else:
            return f"Time range: {current_range[0]:.2f} - {current_range[1]:.2f}"
    
    def update_plots(val=None):
        """Update all plots based on current time range"""
        # Clear all axes
        for ax in [ax_hist, ax_scatter, ax_lifetime, ax_categories]:
            ax.clear()
        
        # Filter data
        df_filtered = filter_data()
        
        if len(df_filtered) == 0:
            ax_hist.text(0.5, 0.5, 'No data in this time range', ha='center', va='center', transform=ax_hist.transAxes)
            return
        
        # 1. Histogram of samples per batch
        ax_hist.hist(df_filtered['samples_per_batch'], bins=50, edgecolor='black', alpha=0.7)
        ax_hist.axvline(df_filtered['samples_per_batch'].mean(), color='r', linestyle='--',
                       label=f'Mean: {df_filtered["samples_per_batch"].mean():.3f}')
        ax_hist.axvline(df_filtered['samples_per_batch'].median(), color='g', linestyle='--',
                       label=f'Median: {df_filtered["samples_per_batch"].median():.3f}')
        ax_hist.set_xlabel('Samples per Batch')
        ax_hist.set_ylabel('Number of Transitions')
        ax_hist.set_title(f'Distribution of Samples per Batch\n({len(df_filtered)}/{len(df)} transitions)')
        ax_hist.legend()
        ax_hist.grid(alpha=0.3)
        
        # 2. Sample count vs samples per batch
        scatter = ax_scatter.scatter(df_filtered['sample_count'], df_filtered['samples_per_batch'],
                                    alpha=0.5, s=1, c=df_filtered['possible_samples'],
                                    cmap='viridis')
        ax_scatter.set_xlabel('Absolute Sample Count')
        ax_scatter.set_ylabel('Samples per Batch')
        ax_scatter.set_title('Sample Count vs Samples per Batch')
        ax_scatter.grid(alpha=0.3)
        
        # 3. Samples per batch over transition lifetime
        ax_lifetime.scatter(df_filtered['possible_samples'], df_filtered['samples_per_batch'], alpha=0.5, s=1)
        ax_lifetime.set_xlabel('Transition Lifetime (possible samples)')
        ax_lifetime.set_ylabel('Samples per Batch')
        ax_lifetime.set_title('Samples per Batch vs Lifetime in Buffer')
        ax_lifetime.grid(alpha=0.3)
        
        # 4. Top vs bottom samples per batch
        top_percentile = df_filtered['samples_per_batch'].quantile(0.9)
        bottom_percentile = df_filtered['samples_per_batch'].quantile(0.1)
        
        categories = ['Bottom 10%', 'Middle 80%', 'Top 10%']
        counts = [
            (df_filtered['samples_per_batch'] <= bottom_percentile).sum(),
            ((df_filtered['samples_per_batch'] > bottom_percentile) &
             (df_filtered['samples_per_batch'] < top_percentile)).sum(),
            (df_filtered['samples_per_batch'] >= top_percentile).sum()
        ]
        colors = ['red', 'gray', 'green']
        
        ax_categories.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax_categories.set_ylabel('Number of Transitions')
        ax_categories.set_title('Samples per Batch Distribution')
        ax_categories.grid(alpha=0.3, axis='y')
        
        # Add percentages
        total = len(df_filtered)
        for i, (cat, count) in enumerate(zip(categories, counts)):
            if count > 0:
                ax_categories.text(i, count, f'{count}\n({100*count/total:.1f}%)',
                                 ha='center', va='bottom')
        
        # Update info text
        ax_text.clear()
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, format_time_text(), ha='center', va='center',
                    transform=ax_text.transAxes, fontsize=10)
        
        fig.canvas.draw_idle()
    
    # Create separate sliders for start and end time
    slider_label = 'Training Time (seconds)' if time_col == 'time_numeric' else time_col
    
    slider_start = Slider(
        ax=ax_slider_start,
        label=f'Start {slider_label}',
        valmin=time_min,
        valmax=time_max,
        valinit=time_min,
        valstep=(time_max - time_min) / 200
    )
    
    slider_end = Slider(
        ax=ax_slider_end,
        label=f'End {slider_label}',
        valmin=time_min,
        valmax=time_max,
        valinit=time_max,
        valstep=(time_max - time_min) / 200
    )
    
    def update_start(val):
        current_range[0] = min(val, current_range[1])  # Ensure start <= end
        update_plots()
    
    def update_end(val):
        current_range[1] = max(val, current_range[0])  # Ensure end >= start
        update_plots()
    
    # Connect sliders to update functions
    slider_start.on_changed(update_start)
    slider_end.on_changed(update_end)
    
    # Initial plot
    update_plots()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample frequency distribution to {save_path}")
    
    plt.show()

def plot_spatial_heatmap(df, map_name='RCA1', save_path=None, enable_slider=ENABLE_TIME_SLIDER):
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
    
    # Prepare time data if slider enabled
    df_with_pos, time_col = prepare_time_data(df_with_pos, enable_slider)
    
    if enable_slider and time_col:
        # Use interactive version
        plot_spatial_heatmap_interactive(df_with_pos, img_array, config, map_name, time_col, save_path)
    else:
        # Use static version
        plot_spatial_heatmap_static(df_with_pos, img_array, config, map_name, save_path)
    
    # Plot reward heatmap (always static, shown separately)
    plot_reward_heatmap(df_with_pos, img_array, map_name)

def plot_spatial_heatmap_static(df_with_pos, img_array, config, map_name, save_path=None, global_stats=None):
    """Static version of spatial heatmap (no slider)"""
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Use global max for color scaling if provided
    samples_max = global_stats['samples_per_batch_max'] if global_stats else df_with_pos['samples_per_batch'].max()
    
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
        vmax=samples_max  # Use global max for consistent color scaling
    )
    
    ax.set_title(f'Samples per Batch Heatmap on {map_name}')
    ax.axis('off')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Samples per Batch')
    
    # Right: 2D histogram heatmap
    ax = axes[1]
    ax.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)
    
    # Create 2D histogram using the same y-axis convention as the scatter plot
    y_flipped = img_height - df_with_pos['pixel_y']
    
    # Use global ranges if provided for consistent binning
    if global_stats and 'pose_x_min' in global_stats:
        # Convert world coords to pixel coords for consistent bins
        origin = config['origin']
        resolution = config['resolution']
        x_min_px = (global_stats['pose_x_min'] - origin[0]) / resolution
        x_max_px = (global_stats['pose_x_max'] - origin[0]) / resolution
        y_min_flipped = img_height - (global_stats['pose_y_max'] - origin[1]) / resolution
        y_max_flipped = img_height - (global_stats['pose_y_min'] - origin[1]) / resolution
        
        x_bins = np.linspace(x_min_px, x_max_px, 50)
        y_bins = np.linspace(y_min_flipped, y_max_flipped, 50)
    else:
        x_bins = np.linspace(df_with_pos['pixel_x'].min(), 
                             df_with_pos['pixel_x'].max(), 50)
        y_bins = np.linspace(y_flipped.min(), 
                             y_flipped.max(), 50)
    
    # Compute weighted histogram (weighted by samples per batch)
    H, xedges, yedges = np.histogram2d(
        df_with_pos['pixel_x'],
        y_flipped,
        bins=[x_bins, y_bins],
        weights=df_with_pos['samples_per_batch']
    )
    
    # Count histogram for normalization
    H_count, _, _ = np.histogram2d(
        df_with_pos['pixel_x'],
        y_flipped,
        bins=[x_bins, y_bins]
    )
    
    # Normalize by count to get average samples per batch per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = H / H_count
        H_avg[~np.isfinite(H_avg)] = 0
    
    im = ax.imshow(
        H_avg.T,
        extent=[xedges[0], xedges[-1], 
            yedges[0], yedges[-1]],
        cmap='viridis',  # Dark blue (low) -> cyan -> green -> yellow (high)
        alpha=0.7,
        origin='upper',
        vmin=0,
        vmax=samples_max  # Use global max for consistent color scaling
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

def plot_spatial_heatmap_interactive(df_with_pos, img_array, config, map_name, time_col, save_path=None):
    """Interactive version of spatial heatmap with time slider"""
    img_height = img_array.shape[0]
    
    # Get time range
    time_min = df_with_pos[time_col].min()
    time_max = df_with_pos[time_col].max()
    
    # Create figure with space for sliders (2 separate sliders)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.05, 0.05, 1, 0.05], hspace=0.15, wspace=0.15)
    
    # Create axes
    ax_slider_start = fig.add_subplot(gs[0, :])
    ax_slider_end = fig.add_subplot(gs[1, :])
    ax_left = fig.add_subplot(gs[2, 0])
    ax_right = fig.add_subplot(gs[2, 1])
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis('off')
    
    # Track current time range
    current_range = [time_min, time_max]
    
    # Filter data by time range
    def filter_data():
        """Filter data to show only transitions in current time range"""
        return df_with_pos[(df_with_pos[time_col] >= current_range[0]) & 
                           (df_with_pos[time_col] <= current_range[1])].copy()
    
    # Setup bins for histogram (computed once)
    y_flipped_all = img_height - df_with_pos['pixel_y']
    x_bins = np.linspace(df_with_pos['pixel_x'].min(), 
                         df_with_pos['pixel_x'].max(), 50)
    y_bins = np.linspace(y_flipped_all.min(), 
                         y_flipped_all.max(), 50)
    
    def format_time_text():
        """Format time range for display"""
        if time_col == 'time_numeric':
            def format_seconds(s):
                h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"Time range: {format_seconds(current_range[0])} - {format_seconds(current_range[1])}"
        else:
            return f"Time range: {current_range[0]:.2f} - {current_range[1]:.2f}"
    
    # Plot function
    def update_plots(val=None):
        """Update both plots based on current time range"""
        ax_left.clear()
        ax_right.clear()
        
        # Filter data
        df_filtered = filter_data()
        
        if len(df_filtered) == 0:
            ax_left.text(0.5, 0.5, 'No data in this time range', ha='center', va='center', transform=ax_left.transAxes)
            ax_right.text(0.5, 0.5, 'No data in this time range', ha='center', va='center', transform=ax_right.transAxes)
            ax_left.set_title(f'Samples per Batch Heatmap on {map_name}')
            ax_right.set_title('Aggregated Samples per Batch Heatmap')
            return
        
        # Left: Track overlay with sample frequency coloring
        ax_left.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)
        
        df_sorted = df_filtered.sort_values('samples_per_batch')
        scatter = ax_left.scatter(
            df_sorted['pixel_x'],
            img_height - df_sorted['pixel_y'],
            c=df_sorted['samples_per_batch'],
            cmap='viridis',
            s=0.5,
            alpha=0.6,
            vmin=0,
            vmax=df_with_pos['samples_per_batch'].max()  # Use full range for consistency
        )
        
        ax_left.set_title(f'Samples per Batch Heatmap on {map_name}\n({len(df_filtered)}/{len(df_with_pos)} transitions)')
        ax_left.axis('off')
        
        # Right: 2D histogram heatmap
        ax_right.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)
        
        if len(df_filtered) > 0:
            y_flipped = img_height - df_filtered['pixel_y']
            
            # Compute weighted histogram
            H, xedges, yedges = np.histogram2d(
                df_filtered['pixel_x'],
                y_flipped,
                bins=[x_bins, y_bins],
                weights=df_filtered['samples_per_batch']
            )
            
            H_count, _, _ = np.histogram2d(
                df_filtered['pixel_x'],
                y_flipped,
                bins=[x_bins, y_bins]
            )
            
            with np.errstate(divide='ignore', invalid='ignore'):
                H_avg = H / H_count
                H_avg[~np.isfinite(H_avg)] = 0
            
            im = ax_right.imshow(
                H_avg.T,
                extent=[xedges[0], xedges[-1], 
                        yedges[0], yedges[-1]],
                cmap='viridis',
                alpha=0.7,
                origin='upper'
            )
        
        ax_right.set_title('Aggregated Samples per Batch Heatmap')
        ax_right.axis('off')
        
        # Update info text
        ax_text.clear()
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, format_time_text(), ha='center', va='center',
                    transform=ax_text.transAxes, fontsize=10)
        
        fig.canvas.draw_idle()
    
    # Create separate sliders for start and end time
    slider_label = 'Training Time (seconds)' if time_col == 'time_numeric' else time_col
    
    slider_start = Slider(
        ax=ax_slider_start,
        label=f'Start {slider_label}',
        valmin=time_min,
        valmax=time_max,
        valinit=time_min,
        valstep=(time_max - time_min) / 200
    )
    
    slider_end = Slider(
        ax=ax_slider_end,
        label=f'End {slider_label}',
        valmin=time_min,
        valmax=time_max,
        valinit=time_max,
        valstep=(time_max - time_min) / 200
    )
    
    def update_start(val):
        current_range[0] = min(val, current_range[1])  # Ensure start <= end
        update_plots()
    
    def update_end(val):
        current_range[1] = max(val, current_range[0])  # Ensure end >= start
        update_plots()
    
    # Connect sliders to update functions
    slider_start.on_changed(update_start)
    slider_end.on_changed(update_end)
    
    # Initial plot
    update_plots()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial heatmap to {save_path}")
    
    plt.show()

def plot_value_heatmap(
    df_with_pos,
    img_array,
    map_name,
    value_column,
    value_label,
    save_path=None,
    value_min=None,
    value_max=None,
):
    """
    Generic heatmap plotter for any spatial value column (reward, critic output, etc.)
    
    Args:
        df_with_pos: DataFrame with pixel_x, pixel_y, and value column
        img_array: Map image array
        map_name: Name of the map
        value_column: Column name to plot (e.g., 'reward', 'critic_q_min')
        value_label: Label for the colorbar (e.g., 'Reward', 'Q-value')
        save_path: Optional path to save the figure
        value_min: Optional fixed minimum for color normalization
        value_max: Optional fixed maximum for color normalization
    """
    # Check for required columns
    if value_column not in df_with_pos.columns:
        print(f"Warning: No '{value_column}' column found. Skipping {value_label.lower()} heatmap.")
        return

    df_values = df_with_pos.dropna(subset=[value_column])
    if len(df_values) == 0:
        print(f"Warning: No {value_label.lower()} data found. Skipping heatmap.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Track overlay with value coloring
    ax = axes[0]
    ax.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)

    # Sort by value so higher values are drawn on top
    df_values_sorted = df_values.sort_values(value_column)

    img_height = img_array.shape[0]
    data_min = float(df_values[value_column].min())
    data_max = float(df_values[value_column].max())
    value_min = data_min if value_min is None else float(value_min)
    value_max = data_max if value_max is None else float(value_max)

    # Guard against degenerate ranges to avoid singular normalization.
    if np.isclose(value_min, value_max):
        pad = max(1e-6, abs(value_max) * 1e-6)
        value_min -= pad
        value_max += pad
    
    # Use TwoSlopeNorm if values cross zero, otherwise use standard Normalize
    if value_min < 0 < value_max:
        value_norm = TwoSlopeNorm(vmin=value_min, vcenter=0.0, vmax=value_max)
    else:
        value_norm = Normalize(vmin=value_min, vmax=value_max)
    
    scatter = ax.scatter(
        df_values_sorted['pixel_x'],
        img_height - df_values_sorted['pixel_y'],
        c=df_values_sorted[value_column],
        cmap='viridis',
        s=0.5,
        alpha=0.6,
        norm=value_norm
    )

    ax.set_title(f'{value_label} Heatmap on {map_name}')
    ax.axis('off')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(value_label)
    cbar.locator = MaxNLocator(nbins=7)
    cbar.update_ticks()

    # Ensure important anchors are always visible on the colorbar.
    anchor_ticks = [value_min, value_max]
    if value_min < 0 < value_max:
        anchor_ticks.append(0.0)
    ticks = np.array(sorted(set(float(t) for t in np.concatenate([cbar.get_ticks(), anchor_ticks]))))
    ticks = ticks[(ticks >= value_min) & (ticks <= value_max)]
    cbar.set_ticks(ticks)

    # Right: 2D histogram heatmap (average value per bin)
    ax = axes[1]
    ax.imshow(img_array, cmap='gray', origin='upper', alpha=0.3)

    y_flipped = img_height - df_values['pixel_y']
    x_bins = np.linspace(df_values['pixel_x'].min(),
                         df_values['pixel_x'].max(), 50)
    y_bins = np.linspace(y_flipped.min(),
                         y_flipped.max(), 50)

    H, xedges, yedges = np.histogram2d(
        df_values['pixel_x'],
        y_flipped,
        bins=[x_bins, y_bins],
        weights=df_values[value_column]
    )

    H_count, _, _ = np.histogram2d(
        df_values['pixel_x'],
        y_flipped,
        bins=[x_bins, y_bins]
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = H / H_count
        H_avg[~np.isfinite(H_avg)] = 0

    im = ax.imshow(
        H_avg.T,
        extent=[xedges[0], xedges[-1],
            yedges[0], yedges[-1]],
        cmap='viridis',
        alpha=0.7,
        origin='upper',
        norm=value_norm
    )

    ax.set_title(f'Aggregated {value_label} Heatmap')
    ax.axis('off')
    cbar2 = plt.colorbar(im, ax=ax)
    cbar2.set_label(f'Avg {value_label}')
    cbar2.locator = MaxNLocator(nbins=7)
    cbar2.update_ticks()
    cbar2_ticks = np.array(sorted(set(float(t) for t in np.concatenate([cbar2.get_ticks(), anchor_ticks]))))
    cbar2_ticks = cbar2_ticks[(cbar2_ticks >= value_min) & (cbar2_ticks <= value_max)]
    cbar2.set_ticks(cbar2_ticks)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {value_label.lower()} heatmap to {save_path}")
    
    plt.show()

def plot_reward_heatmap(df_with_pos, img_array, map_name, save_path=None, value_min=None, value_max=None):
    """Plot reward-based heatmaps (convenience wrapper)"""
    plot_value_heatmap(
        df_with_pos,
        img_array,
        map_name,
        'reward',
        'Reward',
        save_path,
        value_min=value_min,
        value_max=value_max,
    )

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
