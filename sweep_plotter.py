import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from math import ceil

# ================= CONFIGURATION =================
ROOT_DIR = os.path.join("TrainingLite", "rl_racing", "models")
OUTPUT_DIR = "summary_plots"
METRIC_FILENAME = "training_metrics.png"

# Split Settings
SPLIT_RATIO = 0.46 
PLOT_HEADER_HEIGHT = 0.076 

# Spacing Control
# Gap between Part 1 and Part 2 (Very small number = closer together)
PAIR_GAP = 0.02  
# Gap between different Model groups (Larger number = distinct separation)
GROUP_GAP = 0.2  
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_folder_name(folder_name):
    pattern = r"Sweep_Clip_1Anneal_Example-1_A([0-9\.]+)_B([0-9\.]+)_R([0-9\.]+)"
    match = re.search(pattern, folder_name)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None

def create_summary_plots():
    # --- Step 1: Collect Data ---
    data = {}
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return

    subdirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)) and ("Sweep_" in d or "sweep_" in d)]
    
    # --- Step 1.5: Detect Image Size for Scaling ---
    # We load the first available image to determine the correct figure dimensions
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
    
    # Calculate Figure Size in Inches
    # We want 3 groups per row. Each group has 2 images side-by-side.
    # Total Width = (Image Width * 2 * 3) / DPI
    # Total Height per Row = (Part Height) / DPI
    
    # We add 10% buffer for margins/titles
    fig_w_inches = ((img_w * 6) / dpi) * 1.1 
    row_h_inches = (part_h / dpi) * 1.2 

    print(f"Detected Image Size: {img_w}x{img_h}")
    print(f"Calculated Figure Width: {fig_w_inches:.1f} inches")

    print(subdirs)

    # --- Step 2: Build Dictionary ---
    for folder in subdirs:
        params = parse_folder_name(folder)

        if not params: continue
        print("found params:", params)
        
        alpha, beta, ratio = params
        img_path = os.path.join(ROOT_DIR, folder, METRIC_FILENAME)
        
        if os.path.exists(img_path):
            if alpha not in data: data[alpha] = []
            data[alpha].append({
                'beta': beta,
                'ratio': ratio,
                'path': img_path
            })

    # --- Step 3: Generate Plots ---
    sorted_alphas = sorted(data.keys())

    for alpha in sorted_alphas:
        entries = data[alpha]
        entries.sort(key=lambda x: (x['beta'], x['ratio']))
        
        n_entries = len(entries)
        cols = 3 
        rows = ceil(n_entries / cols)
        
        # Set the figure size based on how many rows we have
        total_fig_height = row_h_inches * rows
        
        fig = plt.figure(figsize=(fig_w_inches, total_fig_height), dpi=dpi)
        fig.suptitle(f"Metrics for Alpha = {alpha}", fontsize=30, weight='bold', y=0.99)

        # Outer Grid: Handles separation between Models
        outer_grid = gridspec.GridSpec(rows, cols, figure=fig, 
                                     wspace=GROUP_GAP, 
                                     hspace=0.15,
                                     left=0.02, right=0.98, top=0.95, bottom=0.02)

        for i, entry in enumerate(entries):
            try:
                # Load Image
                full_img = mpimg.imread(entry['path'])
                
                # Crop Logic
                p1_start = header_px
                p1_end = split_px
                p2_start = split_px + header_px
                
                img_part1 = full_img[p1_start:p1_end, :, :]
                img_part2 = full_img[p2_start:, :, :]

                # Inner Grid: Handles separation between Part 1 & Part 2
                row_idx = i // cols
                col_idx = i % cols
                
                inner_grid = outer_grid[row_idx, col_idx].subgridspec(1, 2, wspace=PAIR_GAP)
                
                ax1 = fig.add_subplot(inner_grid[0])
                ax2 = fig.add_subplot(inner_grid[1])

                ax1.imshow(img_part1, aspect='auto') # aspect='auto' helps fill the box
                ax1.axis('off')
                
                ax2.imshow(img_part2, aspect='auto')
                ax2.axis('off')

                # Title
                ax1.set_title(f"Beta: {entry['beta']} | Ratio: {entry['ratio']}", 
                             loc='left', fontsize=14, fontweight='bold')

            except Exception as e:
                print(f"Error processing {entry['path']}: {e}")

        save_path = os.path.join(OUTPUT_DIR, f"grouped_alpha_{alpha}.png")
        plt.savefig(save_path, dpi=dpi) # Save at same DPI we calculated with
        plt.close(fig)
        print(f"Saved High-Res summary for Alpha {alpha} -> {save_path}")

if __name__ == "__main__":
    create_summary_plots()