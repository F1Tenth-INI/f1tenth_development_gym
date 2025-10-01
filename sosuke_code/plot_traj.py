import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Folder where your CSVs are stored
folder_path = "F1tenth_data/hardware_data/sysid/auto_control/bad_data"

# Folder where you want to save the plots
save_folder = os.path.join(folder_path, "plots")
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

for file in csv_files:
    # Load CSV
    data = pd.read_csv(file, comment = "#")
    
    # Create plot
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 10))
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 20))

    plt.rcParams.update({
    "font.size": 20,        # default text size
    "axes.titlesize": 24,   # title
    "axes.labelsize": 20,   # x and y labels
    "xtick.labelsize": 18,  # x tick labels
    "ytick.labelsize": 18,  # y tick labels
    "legend.fontsize": 18,  # legend
    })

    # Position over time
    ax1.plot(data["time"], data["pose_x"], label="X Position")
    ax1.plot(data["time"], data["pose_y"], label="Y Position")
    ax1.set_title("Position Over Time", fontsize=24)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.legend()
    ax1.grid(True)

    # Control inputs
    ax2.plot(data["time"], data["angular_control"], label="Steering")
    ax2.plot(data["time"], data["translational_control"], label="Acceleration")
    # ax2.plot(data["time"], data["manual_steering_angle"], label="Steering")
    # ax2.plot(data["time"], data["manual_acceleration"], label="Acceleration")
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

    ax4.plot(data["time"], data["linear_vel_x"], label="vel_x")
    ax4.plot(data["time"], data["linear_vel_y"], label="vel_y")
    ax4.set_title("velocities")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("velocity (m/s)")
    ax4.legend()
    ax4.grid(True)

    ax5.plot(data["time"], data["steering_angle"], label="steering_angle")
    ax5.plot(data["time"], data["pose_theta"], label="pose_theta")
    ax5.set_title("Angles")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Angles (rad)")
    ax5.legend()
    ax5.grid(True)


    # Use filename for the plot title and output image
    base_filename = os.path.splitext(os.path.basename(file))[0]
    # fig.suptitle(f"File: {base_filename}")
    fig.suptitle("Control input and state plots after processing", fontsize=24)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    # Save the figure
    save_path = os.path.join(save_folder, f"{base_filename}.png")
    plt.savefig(save_path)
    plt.close()
