import os
import sys
import glob
import platform
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.colors as mcolors

# Select the matplotlib backend based on the operating system.
if platform.system() == "Darwin":
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# Import the decoding function from our separate module.
from mu_vs_mu_control_helpers import decode_mu_from_filename

# Define the data folder.
data_folder = "./MPC_mu_vs_mu_control"

# Check if the requested folder exists.
if not os.path.isdir(data_folder):
    sys.exit(f"Error: The requested folder '{data_folder}' does not exist.")

# Search for CSV files in the specified folder.
file_list = glob.glob(os.path.join(data_folder, "*.csv"))
if not file_list:
    sys.exit(f"Error: The folder '{data_folder}' is empty or contains no CSV files.")

# Option to remove offset from all cost values.
REMOVE_OFFSET = True

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

for filename in file_list:
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

# Populate the DataFrame with the average cost values.
for mu in unique_mus:
    for mu_control in unique_mu_controls:
        key = (mu, mu_control)
        if key in cost_dict:
            heatmap_data.at[mu_control, mu] = np.mean(cost_dict[key])
        else:
            heatmap_data.at[mu_control, mu] = np.nan

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

# ---------------------------------------------------------------------
# Create a custom colormap based on viridis, with NaN values shown in grey.
base_cmap = plt.cm.get_cmap('viridis', 256)
cmap = mcolors.ListedColormap(base_cmap(np.linspace(0, 1, 256)))
cmap.set_bad(color='grey')

# ---------------------------------------------------------------------
# Plot the heatmap using seaborn.
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    heatmap_data,
    cmap=cmap,
    annot=True,  # Annotate each cell with the average cost value.
    fmt=".2f",   # Format the annotation to two decimal places.
    linewidths=0.5,  # Draw gridlines for clarity.
    square=True,     # Ensure each cell is square.
    cbar_kws={'label': 'Average Total Stage Cost per Lap'},
    mask=heatmap_data.isnull()  # Use the custom grey for missing data.
)

# Label the axes and set a title.
ax.set_xlabel("mu")
ax.set_ylabel("mu_control")
plt.title("Heatmap of Average Total Stage Cost per Lap for mu vs. mu_control")
plt.tight_layout()
plt.show()
