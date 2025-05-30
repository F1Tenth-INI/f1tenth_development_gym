import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

load_data = True

# Folders containing CSVs
FOLDER_1 = "F1tenth_data/hardware_data/sysid/auto_control"
FOLDER_2 = "F1tenth_data/hardware_data/sysid/manual_control"

# Common state columns you want to load
state_columns = [
    'angular_vel_z',      # yaw rate
    'linear_vel_x',       # velocity in x
    'linear_vel_y',       # velocity in y
    'pose_theta',         # yaw angle
    'pose_theta_cos',
    'pose_theta_sin',
    'pose_x',             # global x
    'pose_y',             # global y
    'slip_angle',         # deprecated
    'steering_angle'      # front wheel angle
]

# Dictionary for control columns based on folder
control_column_map = {
    "folder1": ["angular_control", "translational_control"],
    "folder2": ["manual_steering_angle", "manual_acceleration"]
}

# Function to load and standardize data
def load_folder(folder, source_name, control_cols):
    all_data = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            df = pd.read_csv(path, comment="#")
            
            # Filter required columns if they exist
            needed_columns = state_columns + control_cols
            present_cols = [col for col in needed_columns if col in df.columns]
            df = df[present_cols].copy()
            
            # Rename control columns to common names
            rename_dict = {}
            if control_cols[0] in df.columns:
                rename_dict[control_cols[0]] = "angular_control"
            if control_cols[1] in df.columns:
                rename_dict[control_cols[1]] = "translational_control"
            df.rename(columns=rename_dict, inplace=True)

            df["source"] = source_name  # tag the origin
            df["source_file"] = file
            all_data.append(df)
    if not all_data:
        raise ValueError(f"No CSV files found in {folder}.")
    return pd.concat(all_data, ignore_index=True)

if load_data:
    df_auto = load_folder(FOLDER_1, "auto", control_column_map["folder1"])
    df_manual = load_folder(FOLDER_2, "manual", control_column_map["folder2"])

    # Combine both datasets
    df_all = pd.concat([df_auto, df_manual],axis=0 ,ignore_index=True)
    df_all.to_csv("state_coverage.csv", index=False)
else:
    # Load the combined dataset
    df_all = pd.read_csv("state_coverage.csv")


# Pairplot of state variables
vars_to_plot = [col for col in ['angular_vel_z', 'linear_vel_x', 'linear_vel_y','pose_theta', 'steering_angle','angular_control','translational_control'] if col in df_all.columns]
g=sns.pairplot(df_all[vars_to_plot + ['source']], hue='source',corner= True)
g.figure.suptitle("State Variable Pairwise Comparison", y=1.02)
plt.tight_layout()
g.figure.savefig("pairplot_states.png", dpi=300)

