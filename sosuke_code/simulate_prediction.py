from utilities.Settings import Settings
from utilities.state_utilities import *

from SI_Toolkit.computation_library import  NumpyLibrary, TensorFlowLibrary
from SI_Toolkit.Predictors.predictor_wrapper import PredictorWrapper
from SI_Toolkit_ASF.car_model import car_model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

horizon = 30
start=100 # Start index for the trajectory to study
title="NN with history, weighted, performance on 30 steps of real data"
file_path = "sosuke_code/F1tenth_data/hardware_data/sysid/auto_control/mpc_loop4.csv"
lib = TensorFlowLibrary()
CarModel = car_model(
            model_of_car_dynamics = Settings.ODE_MODEL_OF_CAR_DYNAMICS,
            batch_size = 1, 
            car_parameter_file = Settings.ENV_CAR_PARAMETER_FILE, 
            dt = 0.02, 
            intermediate_steps=1,
            computation_lib=lib
            )


predoctor = PredictorWrapper()
predoctor.configure(
    batch_size=1,
    horizon=horizon,
    dt=0.02,
    computation_library=lib,
    predictor_specification=None,
    compile_standalone=False,
    mode='simple evaluation',
    hls=False
)


data = pd.read_csv(file_path, comment = "#")

states= data[['angular_vel_z', 'linear_vel_x', 'linear_vel_y', 'pose_theta',
               'pose_theta_cos', 'pose_theta_sin', 'pose_x', 'pose_y', 'slip_angle','steering_angle']]

# control=data[['manual_steering_angle', 'manual_acceleration']]
control=data[['angular_control', 'translational_control']]

s=states.loc[[start], :].values
u=control.loc[start:horizon+start-1, :].values

# Convert to tensors
s = lib.to_tensor(s, dtype=lib.float32)
u = lib.to_tensor(u, dtype=lib.float32)

# Predict the trajectory
s = predoctor.predict(s, u)[0]


# Plot
poses_x = s[:, POSE_X_IDX]
poses_y = s[:, POSE_Y_IDX]
print("poses_x", poses_x)
print("poses_y", poses_y)

pose_theta= s[:, POSE_THETA_IDX]
pose_theta_cos = s[:, POSE_THETA_COS_IDX]
pose_theta_sin = s[:, POSE_THETA_SIN_IDX]
linear_vel_x = s[:, LINEAR_VEL_X_IDX]
linear_vel_y = s[:, LINEAR_VEL_Y_IDX]

print("linear_vel_x", linear_vel_x)
print("linear_vel_y", linear_vel_y)

angular_vel_z = s[:, ANGULAR_VEL_Z_IDX]
steering_angle = s[:, STEERING_ANGLE_IDX]
angular_control = u[:, ANGULAR_CONTROL_IDX]
translational_control = u[:, TRANSLATIONAL_CONTROL_IDX]
time= np.arange(0, (horizon+1) * 0.02, 0.02)
time_control = np.arange(0, horizon * 0.02, 0.02)


real_x= data['pose_x'].values[start:horizon+start +1]
real_y= data['pose_y'].values[start:horizon+start+1]
real_vel_x = data['linear_vel_x'].values[start:horizon+start+1]
real_vel_y = data['linear_vel_y'].values[start:horizon+start+1]
real_angular_vel_z = data['angular_vel_z'].values[start:horizon+start+1]
real_steering_angle = data['steering_angle'].values[start:horizon+start+1]


def compute_rmse(true, pred):
    return np.sqrt(np.mean((np.array(true) - np.array(pred))**2))

rmse_xy = compute_rmse(real_x, poses_x) + compute_rmse(real_y, poses_y)
rmse_vx = compute_rmse(real_vel_x, linear_vel_x)
rmse_vy = compute_rmse(real_vel_y, linear_vel_y)
rmse_yaw = compute_rmse(real_angular_vel_z, angular_vel_z)
rmse_steer = compute_rmse(real_steering_angle, steering_angle)


#saving data
np.savez("MPC_data.npz",
         poses_x=poses_x,
         poses_y=poses_y,
         pose_theta=pose_theta,
         pose_theta_cos=pose_theta_cos,
         pose_theta_sin=pose_theta_sin,
         linear_vel_x=linear_vel_x,
         linear_vel_y=linear_vel_y,
         angular_vel_z=angular_vel_z,
         steering_angle=steering_angle,
         rmse_xy=rmse_xy,
         rmse_vx=rmse_vx,
         rmse_vy=rmse_vy,
         rmse_yaw=rmse_yaw,
         rmse_steer=rmse_steer,
         time=time,
         time_control=time_control,)

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 15), constrained_layout=True)
plt.rcParams.update({
    "font.size": 20,        # default text size
    "axes.titlesize": 24,   # title
    "axes.labelsize": 20,   # x and y labels
    "xtick.labelsize": 18,  # x tick labels
    "ytick.labelsize": 18,  # y tick labels
    "legend.fontsize": 18,  # legend
})

# Trajectory
ax1.plot(real_x, real_y, label="Real Trajectory")
ax1.plot(poses_x, poses_y, label="Simulated Trajectory")
ax1.plot(real_x[0], real_y[0], marker='o', color='red', markersize=10, label='Start Point')
ax1.set_xlabel("displacement (m)", fontsize=21)
ax1.set_ylabel("displacement (m)", fontsize=21)
ax1.set_title(f"Real Trajectory vs Simulated Trajectory (RMSE ≈ {rmse_xy:.3f} m)", fontsize=23)
ax1.legend()
ax1.grid(True)

# v_x
ax2.plot(time, real_vel_x, label="Real Trajectory")
ax2.plot(time, linear_vel_x, label="Simulated Trajectory")
ax2.set_xlabel("Time (s)", fontsize=21)
ax2.set_ylabel("velocity (m/s)", fontsize=21)
ax2.set_title(f"v_x real vs simulated (RMSE ≈ {rmse_vx:.3f})", fontsize=23)
ax2.legend()
ax2.grid(True)

# v_y
ax3.plot(time, real_vel_y, label="Real Trajectory")
ax3.plot(time, linear_vel_y, label="Simulated Trajectory")
ax3.set_xlabel("Time (s)", fontsize=21)
ax3.set_ylabel("velocity (m/s)", fontsize=21)
ax3.set_title(f"v_y real vs simulated (RMSE ≈ {rmse_vy:.3f})", fontsize=23)
ax3.legend()
ax3.grid(True)

# yaw rate
ax4.plot(time, real_angular_vel_z, label="Real Trajectory")
ax4.plot(time, angular_vel_z, label="Simulated Trajectory")
ax4.set_xlabel("Time (s)", fontsize=21)
ax4.set_ylabel("angle (rad/s)", fontsize=21)
ax4.set_title(f"Yaw angular velocity real vs simulated (RMSE ≈ {rmse_yaw:.3f})", fontsize=23)
ax4.legend()
ax4.grid(True)

# steering angle
ax5.plot(time, real_steering_angle, label="Real Trajectory")
ax5.plot(time, steering_angle, label="Simulated Trajectory")
ax5.set_xlabel("Time (s)", fontsize=21)
ax5.set_ylabel("angle (rad)", fontsize=21)
ax5.set_title(f"Steering angle real vs simulated (RMSE ≈ {rmse_steer:.3f})", fontsize=23)
ax5.legend()
ax5.grid(True)

ax6.plot(time_control, angular_control, label="Angular Control")
ax6.plot(time_control, translational_control, label="Translational Control")
ax6.set_xlabel("Time (s)", fontsize=21)
ax6.set_ylabel("Control", fontsize=21)
ax6.set_title("Control Signals", fontsize=23)
ax6.legend()

fig.suptitle(title, fontsize=16)

plt.savefig(title + ".png", dpi=300)

plt.show()