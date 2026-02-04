import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from math import ceil

# ================= CONFIGURATION =================
ROOT_DIR = os.path.join("TrainingLite", "rl_racing", "30JanModels")
OUTPUT_DIR = "summary_plots_curriculum"
METRIC_FILENAME = "training_metrics.png"

# Split Settings
SPLIT_RATIO = 0.46 
PLOT_HEADER_HEIGHT = 0.076 

# Spacing Control
# Gap between Part 1 and Part 2 (Very small number = closer together)
PAIR_GAP = 0.02  
# Gap between different runs within a group
RUN_GAP = 0.05
# Gap between different Model groups (Larger number = distinct separation)
GROUP_GAP = 0.2  
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_folder_name(folder_name):
    """
    Parse folder name with pattern: 30Jan_Sweep_S_<value>_W_<value>_N_<value>_Run<number>
    Returns: (s_value, w_value, n_value, run_num) or None
    """
    pattern = r"30Jan_Sweep_S_(\w+)_W_(\w+)_N_(\w+)_Run(\d+)"
    match = re.search(pattern, folder_name)
    if match:
        s_value = match.group(1)      # "None", "speed_cap", "vel_factor", etc.
        w_value = match.group(2)      # "True" or "False"
        n_value = match.group(3)      # "True" or "False"
        run_num = int(match.group(4)) # 1, 2, 3, 4, 5
        return (s_value, w_value, n_value, run_num)
    return None

def create_curriculum_summary_plots():
    """
    Groups all runs with the same S_W_N settings and plots them together.
    Each group shows all 5 runs side-by-side.
    Also creates Part 1 only versions and a master stacked comparison.
    """
    # --- Step 1: Collect Data ---
    data = {}  # Key: (s_value, w_value, n_value), Value: list of {run_num, path}
    part1_images = {}  # Store Part 1 images for master plot: Key: (s, w, n), Value: cropped image array
    
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return

    subdirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    
    # --- Step 1.5: Detect Image Size for Scaling ---
    sample_img_path = None
    for folder in subdirs:
        possible_path = os.path.join(ROOT_DIR, folder, METRIC_FILENAME)
        if os.path.exists(possible_path):
            sample_img_path = possible_path
            break
            
    if not sample_img_path:
        print("No images found to process.")
        return

    # Read dimensions
    sample_img = mpimg.imread(sample_img_path)
    img_h, img_w, _ = sample_img.shape
    
    # Calculate crop height
    split_px = int(img_h * SPLIT_RATIO)
    header_px = int(img_h * PLOT_HEADER_HEIGHT)
    
    # Effective height of one "Part" (approx half height - header)
    part_h = split_px - header_px
    
    # DPI (Dots Per Inch) - Standard screen res
    dpi = 100 
    
    print(f"Detected Image Size: {img_w}x{img_h}")

    # --- Step 2: Build Dictionary ---
    for folder in subdirs:
        params = parse_folder_name(folder)
        if not params:
            continue
            
        s_value, w_value, n_value, run_num = params
        img_path = os.path.join(ROOT_DIR, folder, METRIC_FILENAME)
        
        if os.path.exists(img_path):
            key = (s_value, w_value, n_value)
            if key not in data:
                data[key] = []
            data[key].append({
                'run_num': run_num,
                'path': img_path
            })
            print(f"Found: S={s_value}, W={w_value}, N={n_value}, Run={run_num}")

    if not data:
        print("No matching folders found.")
        return

    # --- Step 3: Generate Plots for Each Setting Group ---
    sorted_keys = sorted(data.keys())
    
    # Store Part 1 only figures for final stacked plot
    part1_group_data = []  # List of (key, runs, part1_images) tuples

    for key in sorted_keys:
        s_value, w_value, n_value, = key
        runs = data[key]
        runs.sort(key=lambda x: x['run_num'])
        
        n_runs = len(runs)
        print(f"\nProcessing S={s_value}, W={w_value}, N={n_value} with {n_runs} runs")
        
        # Each run gets 2 columns (Part 1 and Part 2)
        n_cols = n_runs * 2
        
        # Calculate figure size
        # Width: accommodate all runs side-by-side (2 images per run)
        fig_w_inches = ((img_w * n_cols) / dpi) * 1.15
        fig_h_inches = (part_h / dpi) * 1.3
        
        fig = plt.figure(figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
        fig.suptitle(f"S={s_value} | W={w_value} | N={n_value} - All Runs", 
                     fontsize=24, weight='bold', y=0.98)

        # Create grid: 1 row, n_runs groups, each group has 2 columns (Part1, Part2)
        # We'll use nested gridspec for fine control
        outer_grid = gridspec.GridSpec(1, n_runs, figure=fig,
                                       wspace=RUN_GAP,
                                       left=0.02, right=0.98, top=0.92, bottom=0.02)

        # Store Part 1 images for this group
        group_part1_images = []

        for i, run_entry in enumerate(runs):
            try:
                # Load Image
                full_img = mpimg.imread(run_entry['path'])
                
                # Crop Logic (same as original)
                p1_start = header_px
                p1_end = split_px
                p2_start = split_px + header_px
                
                img_part1 = full_img[p1_start:p1_end, :, :]
                img_part2 = full_img[p2_start:, :, :]
                
                # Store Part 1 for later use
                group_part1_images.append(img_part1)

                # Inner grid for Part 1 & Part 2
                inner_grid = outer_grid[i].subgridspec(1, 2, wspace=PAIR_GAP)
                
                ax1 = fig.add_subplot(inner_grid[0])
                ax2 = fig.add_subplot(inner_grid[1])

                ax1.imshow(img_part1, aspect='auto')
                ax1.axis('off')
                
                ax2.imshow(img_part2, aspect='auto')
                ax2.axis('off')

                # Title for this run
                ax1.set_title(f"Run {run_entry['run_num']}", 
                             loc='center', fontsize=16, fontweight='bold')

            except Exception as e:
                print(f"Error processing {run_entry['path']}: {e}")

        # Save with descriptive filename
        safe_s = s_value.replace('_', '-')
        save_path = os.path.join(OUTPUT_DIR, f"S_{safe_s}_W_{w_value}_N_{n_value}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")
        
        # --- Create Part 1 Only Version ---
        if group_part1_images:
            fig_part1 = plt.figure(figsize=(fig_w_inches / 2, fig_h_inches), dpi=dpi)
            fig_part1.suptitle(f"S={s_value} | W={w_value} | N={n_value} - All Runs (Part 1 Only)", 
                               fontsize=20, weight='bold', y=0.98)
            
            outer_grid_p1 = gridspec.GridSpec(1, n_runs, figure=fig_part1,
                                              wspace=RUN_GAP * 0.5,
                                              left=0.02, right=0.98, top=0.92, bottom=0.02)
            
            for i, img_part1 in enumerate(group_part1_images):
                ax = fig_part1.add_subplot(outer_grid_p1[i])
                ax.imshow(img_part1, aspect='auto')
                ax.axis('off')
                ax.set_title(f"Run {runs[i]['run_num']}", 
                            loc='center', fontsize=14, fontweight='bold')
            
            save_path_p1 = os.path.join(OUTPUT_DIR, f"S_{safe_s}_W_{w_value}_N_{n_value}_part1.png")
            plt.savefig(save_path_p1, dpi=dpi, bbox_inches='tight')
            plt.close(fig_part1)
            print(f"Saved Part 1: {save_path_p1}")
            
            # Store for master stacked plot
            part1_group_data.append((key, runs, group_part1_images))

    # --- Step 4: Create Master Stacked Comparison (Part 1 Only) ---
    if part1_group_data:
        print("\n--- Creating master stacked comparison ---")
        n_groups = len(part1_group_data)
        
        # Determine max runs per group (usually 5)
        max_runs = max(len(runs) for _, runs, _ in part1_group_data)
        
        # Calculate figure dimensions
        # Width: based on max runs
        master_w_inches = ((img_w * max_runs) / dpi) * 1.15
        # Height: stack all groups vertically
        master_h_inches = (part_h / dpi) * n_groups * 1.1
        
        fig_master = plt.figure(figsize=(master_w_inches, master_h_inches), dpi=dpi)
        fig_master.suptitle("All Curriculum Settings Comparison (Part 1)", 
                           fontsize=26, weight='bold', y=0.995)
        
        # Outer grid: n_groups rows, 1 column
        outer_master = gridspec.GridSpec(n_groups, 1, figure=fig_master,
                                        hspace=0.15,
                                        left=0.02, right=0.98, top=0.98, bottom=0.02)
        
        for group_idx, (key, runs, part1_images) in enumerate(part1_group_data):
            s_value, w_value, n_value = key
            n_runs = len(runs)
            
            # Inner grid for this group's runs
            inner_master = outer_master[group_idx].subgridspec(1, n_runs, wspace=RUN_GAP * 0.3)
            
            for run_idx, img_part1 in enumerate(part1_images):
                ax = fig_master.add_subplot(inner_master[run_idx])
                ax.imshow(img_part1, aspect='auto')
                ax.axis('off')
                
                # Title only on first run of each group
                if run_idx == 0:
                    ax.text(-0.02, 0.5, f"S={s_value}\nW={w_value}\nN={n_value}", 
                           transform=ax.transAxes, fontsize=11, weight='bold',
                           verticalalignment='center', horizontalalignment='right')
                
                # Run number at top
                ax.set_title(f"Run {runs[run_idx]['run_num']}", 
                           loc='center', fontsize=10, fontweight='bold')
        
        master_path = os.path.join(OUTPUT_DIR, "all_groups_comparison.png")
        plt.savefig(master_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_master)
        print(f"\n✓ Master stacked comparison saved: {master_path}")

    print(f"\n✓ All curriculum summary plots saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    create_curriculum_summary_plots()
