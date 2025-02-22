import os
import sys
import glob
import platform
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.colors as mcolors
from tqdm import tqdm

# Select the matplotlib backend based on the operating system.
if platform.system() == "Darwin":
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# Import the decoding function from our separate module.
from mu_vs_mu_control_helpers import decode_mu_from_filename

# Define the data folder.
data_folder = "./MPC_mu_vs_mu_control_08"

# Check if the requested folder exists.
if not os.path.isdir(data_folder):
    sys.exit(f"Error: The requested folder '{data_folder}' does not exist.")

# Search for CSV files in the specified folder.
file_list = glob.glob(os.path.join(data_folder, "*.csv"))
if not file_list:
    sys.exit(f"Error: The folder '{data_folder}' is empty or contains no CSV files.")

# Options:
REMOVE_OFFSET = True         # Remove offset so that the global minimum becomes 0.
NORMALIZE_COLUMNWISE = True  # Normalize each column to the 0-1 scale.
SAVE_FIGURE = False           # Set to True to save the figure.
FIGURE_FILENAME = data_folder + '_heatmap.png'
ANNOT_FORMAT = "{:.2f}"       # Option to control the number format in annotations.

# Dictionary to store a list of cost values for each (mu, mu_control) pair.
# Using lists allows us to average the cost when multiple files exist for the same parameters.
cost_dict = {}

def calculate_laps(df):
    """
    Calculate lap numbers based on the 'nearest_wpt_idx' column.

    A new lap is detected when the 'nearest_wpt_idx' value decreases compared to the previous row.
    The function adds a 'lap' column that labels the lap number for each row.
    """
    indices = df['nearest_wpt_idx']
    lap_indices = np.where(indices < indices.shift(1))[0]
    df['lap'] = 0
    for i in range(len(lap_indices)):
        if i == 0:
            df.loc[:lap_indices[i], 'lap'] = 0
        else:
            df.loc[lap_indices[i - 1]:lap_indices[i], 'lap'] = i
    return df

# Process files with a progress bar.
for filename in tqdm(file_list, desc="Processing files"):
    # Decode mu and mu_control from the filename.
    result = decode_mu_from_filename(filename)
    if result is None:
        continue
    mu_val, mu_control_val = result

    try:
        # Read the CSV file, skipping commented lines.
        df = pd.read_csv(filename, comment='#')
    except Exception as e:
        print(f"WARNING: Could not read {filename}: {e}")
        continue

    # Verify the required column exists.
    if 'total_stage_cost' not in df.columns:
        print(f"WARNING: 'total_stage_cost' column not found in {filename}")
        continue

    # Calculate lap numbers.
    df = calculate_laps(df)
    unique_laps = sorted(df['lap'].unique())

    # If there are at least 3 laps, exclude the first and last lap.
    if len(unique_laps) >= 3:
        min_lap = unique_laps[0]
        max_lap = unique_laps[-1]
        # Filter out the first and last lap.
        df_filtered = df[(df['lap'] != min_lap) & (df['lap'] != max_lap)]
        # Group by lap and sum the cost for each lap.
        lap_costs = df_filtered.groupby('lap')['total_stage_cost'].sum()
        # Compute the average total cost per lap from the remaining laps.
        cost_value = lap_costs.mean()
    else:
        # If there are fewer than 3 laps, use the overall cost sum.
        cost_value = df['total_stage_cost'].sum()

    key = (mu_val, mu_control_val)
    # Append the cost_value for averaging later if multiple files exist.
    if key not in cost_dict:
        cost_dict[key] = []
    cost_dict[key].append(cost_value)

# ---------------------------------------------------------------------
# Build the heatmap grid based on the available data.
# Create sorted lists of unique mu and mu_control values.
unique_mus = sorted({mu for (mu, _) in cost_dict.keys()})
unique_mu_controls = sorted({mu_control for (_, mu_control) in cost_dict.keys()})

# Create a DataFrame where rows correspond to mu_control values and columns to mu.
heatmap_data = pd.DataFrame(index=unique_mu_controls, columns=unique_mus)
annot_data   = pd.DataFrame(index=unique_mu_controls, columns=unique_mus)
status_data  = pd.DataFrame(index=unique_mu_controls, columns=unique_mus)  # "valid", "incomplete", "missing"

# Populate the DataFrames.
for mu in unique_mus:
    for mu_control in unique_mu_controls:
        key = (mu, mu_control)
        if key in cost_dict:
            # Only use the value if there are exactly two files.
            if len(cost_dict[key]) == 2:
                avg_val = np.mean(cost_dict[key])
                heatmap_data.at[mu_control, mu] = avg_val
                annot_data.at[mu_control, mu] = ANNOT_FORMAT.format(avg_val)
                status_data.at[mu_control, mu] = "valid"
            else:
                # If only one file (or more than two), mark as invalid.
                heatmap_data.at[mu_control, mu] = np.nan
                annot_data.at[mu_control, mu] = ""
                status_data.at[mu_control, mu] = "incomplete"
        else:
            heatmap_data.at[mu_control, mu] = np.nan
            annot_data.at[mu_control, mu] = "miss.\ndata"
            status_data.at[mu_control, mu] = "missing"

# Ensure all values are float (NaN values remain unchanged).
heatmap_data = heatmap_data.astype(float)

# ---------------------------------------------------------------------
# Optionally remove offset: subtract the global minimum from every cell so that the scale starts at 0.
if REMOVE_OFFSET:
    # Compute the global minimum of the cost values in the DataFrame.
    offset = heatmap_data.min().min()
    # Subtract the offset from all entries.
    heatmap_data = heatmap_data - offset

# ---------------------------------------------------------------------
# Reorder the DataFrame so that the lowest mu_control values appear at the bottom.
# This flips the order of the rows.
heatmap_data = heatmap_data.iloc[::-1]
annot_data   = annot_data.iloc[::-1]
status_data  = status_data.iloc[::-1]

# ---------------------------------------------------------------------
# Optionally normalize each column to a 0-1 scale.
if NORMALIZE_COLUMNWISE:
    def normalize_column(col):
        col_min = col.min(skipna=True)
        col_max = col.max(skipna=True)
        # Avoid division by zero when all values are the same.
        if col_max == col_min:
            return col
        return (col - col_min) / (col_max - col_min)
    heatmap_data = heatmap_data.apply(normalize_column, axis=0)

# ---------------------------------------------------------------------
# Create a custom colormap for valid cells.
# We'll let "bad" (NaN) cells appear in the missing-data color.
base_cmap = plt.cm.get_cmap('viridis', 256)
cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))
cmap.set_bad(color='grey')  # "missing" cells will be grey.

# ---------------------------------------------------------------------
# Plot the heatmap using seaborn.
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    heatmap_data,
    cmap=cmap,
    annot=annot_data,
    fmt="",
    annot_kws={"size": 8},
    linewidths=0.5,
    square=True,
    cbar_kws={
        'label': 'Average Total Stage Cost per Lap (Normalized)' if NORMALIZE_COLUMNWISE
                 else 'Average Total Stage Cost per Lap'
    },
    mask=heatmap_data.isnull()  # This will color "missing" cells with cmap.set_bad.
)

# Now overlay a lightgray rectangle for cells marked as "incomplete".
# Iterate over the data grid coordinates.
for i, mu_control in enumerate(heatmap_data.index):
    for j, mu in enumerate(heatmap_data.columns):
        if status_data.loc[mu_control, mu] == "incomplete":
            # (j, i) is the bottom-left corner of the cell in heatmap coordinates.
            rect = plt.Rectangle((j, i), 1, 1, facecolor="lightgray", edgecolor="none", zorder=2)
            ax.add_patch(rect)

# Increase the axis label font size.
ax.set_xlabel("mu", fontsize=14)
ax.set_ylabel("mu_control", fontsize=14)
plt.title(f"Heatmap of Average Total Stage Cost per Lap for mu vs. mu_control\nDataset: {os.path.basename(data_folder)}", fontsize=16)
plt.tight_layout()

# Optionally save the figure.
if SAVE_FIGURE:
    plt.savefig(FIGURE_FILENAME)

plt.show()
