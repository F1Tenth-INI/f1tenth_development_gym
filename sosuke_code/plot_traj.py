import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Folder where your CSVs are stored
folder_path = "F1tenth_data/hardware_data/sysid/manual_control"

# Folder where you want to save the plots
save_folder = os.path.join(folder_path, "plots")
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

for file in csv_files:
    # Load CSV
    data = pd.read_csv(file, comment = "#")
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Position over time
    ax1.plot(data["time"], data["pose_x"], label="X Position")
    ax1.plot(data["time"], data["pose_y"], label="Y Position")
    ax1.set_title("Position Over Time")
    ax1.legend()
    ax1.grid(True)

    # Control inputs
    # ax2.plot(data["time"], data["angular_control"], label="Steering")
    # ax2.plot(data["time"], data["translational_control"], label="Acceleration")
    ax2.plot(data["time"], data["manual_steering_angle"], label="Steering")
    ax2.plot(data["time"], data["manual_acceleration"], label="Acceleration")
    ax2.set_title("Control Inputs")
    ax2.legend()
    ax2.grid(True)

    # Trajectory (x vs y)
    ax3.plot(data["pose_x"], data["pose_y"], label="Trajectory")
    ax3.set_title("Trajectory")
    ax3.set_xlabel("X Position")
    ax3.set_ylabel("Y Position")
    ax3.legend()
    ax3.grid(True)

    # Use filename for the plot title and output image
    base_filename = os.path.splitext(os.path.basename(file))[0]
    fig.suptitle(f"File: {base_filename}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    # Save the figure
    save_path = os.path.join(save_folder, f"{base_filename}.png")
    plt.savefig(save_path)
    plt.close()
