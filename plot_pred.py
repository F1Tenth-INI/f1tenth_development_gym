import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#Recorded data from physical car
title="Weighted NN vs MPC prediction error on 30 steps of real data"
file_path = "sosuke_code/F1tenth_data/hardware_data/sysid/auto_control/mpc_loop4.csv"
data = pd.read_csv(file_path, comment = "#")
horizon = 30
start=100 # Start index for the trajectory to study

real_x= data['pose_x'].values[start:horizon+start +1]
real_y= data['pose_y'].values[start:horizon+start+1]
real_vel_x = data['linear_vel_x'].values[start:horizon+start+1]
real_vel_y = data['linear_vel_y'].values[start:horizon+start+1]
real_angular_vel_z = data['angular_vel_z'].values[start:horizon+start+1]
real_steering_angle = data['steering_angle'].values[start:horizon+start+1]
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 15), constrained_layout=True)


"""Load sim data of mpc and NN"""
mpc_data = np.load("MPC_data.npz")
NN_data = np.load("NN_data.npz")

time = mpc_data['time']


mpc_posx = mpc_data['poses_x']
mpc_posy = mpc_data['poses_y']
mpc_vel_x = mpc_data['linear_vel_x']
mpc_vel_y = mpc_data['linear_vel_y']
mpc_angular_vel_z = mpc_data['angular_vel_z']
mpc_steering_angle = mpc_data['steering_angle']
mpc_rmse_xy = mpc_data['rmse_xy']
mpc_rmse_vx = mpc_data['rmse_vx']
mpc_rmse_vy = mpc_data['rmse_vy']
mpc_rmse_yaw = mpc_data['rmse_yaw']
mpc_rmse_steer = mpc_data['rmse_steer']

nn_posx=NN_data['poses_x']
nn_posy=NN_data['poses_y']
nn_vel_x=NN_data['linear_vel_x']
nn_vel_y=NN_data['linear_vel_y']
nn_angular_vel_z=NN_data['angular_vel_z']
nn_steering_angle=NN_data['steering_angle']
nn_rmse_xy = NN_data['rmse_xy']
nn_rmse_vx = NN_data['rmse_vx']
nn_rmse_vy = NN_data['rmse_vy']
nn_rmse_yaw = NN_data['rmse_yaw']
nn_rmse_steer = NN_data['rmse_steer']

plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

fig.suptitle(title, fontsize=24)  # make the overall figure title even larger


# Trajectory
ax1.plot(real_x, real_y, label="Real Trajectory")
ax1.plot(mpc_posx, mpc_posy, label="MPC Trajectory")
ax1.plot(nn_posx, nn_posy, label="NN Trajectory")
ax1.plot(real_x[0], real_y[0], marker='o', color='red', markersize=10, label='Start Point')
ax1.set_xlabel("displacement (m)", fontsize=21)
ax1.set_ylabel("displacement (m)", fontsize=21)
ax1.set_title(f"Position MPC (RMSE ≈ {mpc_rmse_xy:.3f}) vs NN (RMSE ≈ {nn_rmse_xy:.3f})")
# ax1.legend()
ax1.grid(True)

# v_x
ax2.plot(time, real_vel_x, label="Real Trajectory")
ax2.plot(time, mpc_vel_x, label="MPC Trajectory")
ax2.plot(time, nn_vel_x, label="NN Trajectory")
ax2.set_xlabel("Time (s)", fontsize=21)
ax2.set_ylabel("velocity (m/s)", fontsize=21)
ax2.set_title(f"Longitudinal velocity MPC (RMSE ≈ {mpc_rmse_vx:.3f}) vs NN (RMSE ≈ {nn_rmse_vx:.3f})")
# ax2.legend()
ax2.grid(True)

# v_y
ax3.plot(time, real_vel_y, label="Real Trajectory")
ax3.plot(time, mpc_vel_y, label="MPC Trajectory")
ax3.plot(time, nn_vel_y, label="NN Trajectory")
ax3.set_xlabel("Time (s)", fontsize=21)
ax3.set_ylabel("velocity (m/s)", fontsize=21)
ax3.set_title(f"Lateral velocity MPC (RMSE ≈ {mpc_rmse_vy:.3f}) vs NN (RMSE ≈ {nn_rmse_vy:.3f})")
# ax3.legend()
ax3.grid(True)

# yaw rate
ax4.plot(time, real_angular_vel_z, label="Real Trajectory")
ax4.plot(time, mpc_angular_vel_z, label="MPC Trajectory")
ax4.plot(time, nn_angular_vel_z, label="NN Trajectory")
ax4.set_xlabel("Time (s)", fontsize=21)
ax4.set_ylabel("angle (rad/s)", fontsize=21)
ax4.set_title(f"Yaw angular velocity MPC (RMSE ≈ {mpc_rmse_yaw:.3f}) vs NN (RMSE ≈ {nn_rmse_yaw:.3f})")
# ax4.legend()
ax4.grid(True)

# steering angle
ax5.plot(time, real_steering_angle, label="Real Trajectory")
ax5.plot(time, mpc_steering_angle, label="MPC Trajectory")
ax5.plot(time, nn_steering_angle, label="NN Trajectory")
ax5.set_xlabel("Time (s)", fontsize=21)
ax5.set_ylabel("angle (rad)", fontsize=21)
ax5.set_title(f"Steering angle MPC (RMSE ≈ {mpc_rmse_steer:.3f}) vs NN (RMSE ≈ {nn_rmse_steer:.3f})")
# ax5.legend()
ax5.grid(True)

fig.suptitle(title, fontsize=16)

# --- GLOBAL LEGEND ---
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=16)

# adjust layout so legend doesn’t overlap
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.suptitle(title, fontsize=20)

plt.savefig(title + ".png", dpi=300)

plt.show()