import numpy as np
import pandas as pd


A = np.array(['angular_vel_z', 'linear_vel_x', 'pose_theta_cos', 'pose_theta_sin', 'pose_x', 'pose_y', 'slip_angle', 'steering_angle'])
B = np.array([9,8,7,6,5])

print(np.sort(A))
#df = pd.DataFrame({0:A, 1:B})
#print(df)


'''
# Coordinate transformation to describe waypoint position relative to car position, x-axis points through windshield, y-axis to the left of the driver
if np.sum(waypoints_x_to_save_absolute) != 0 and np.sum(waypoints_y_to_save_absolute) != 0:
    # Translation:
    waypoints_x_to_save_after_translation = waypoints_x_to_save_absolute - self.odometry_dict['pose_x']
    waypoints_y_to_save_after_translation = waypoints_y_to_save_absolute - self.odometry_dict['pose_y']
    sin_theta = self.odometry_dict['pose_theta_sin']
    cos_theta = self.odometry_dict['pose_theta_cos']
    # Rotation (counterclockwise):
    waypoints_x_to_save_relative = np.round(
        waypoints_x_to_save_after_translation * cos_theta + waypoints_y_to_save_after_translation * sin_theta, 4)
    waypoints_y_to_save_relative = np.round(
        waypoints_x_to_save_after_translation * -sin_theta + waypoints_y_to_save_after_translation * cos_theta, 4)
else:
    waypoints_x_to_save_relative = waypoints_x_to_save_absolute
    waypoints_y_to_save_relative = waypoints_y_to_save_absolute
'''

"""
w_x = 2
w_y = 5

pose_x = 9
pose_y = -1
theta = np.pi * 0.25
#translation

w_x2 = w_x - pose_x
w_y2 = w_y - pose_y

print(w_x2, "and", w_y2)


w_x3 = round(w_x2 * np.cos(theta) + w_y2 * np.sin(theta),4)
w_y3 = round(w_x2 * -np.sin(theta) + w_y2 * np.cos(theta),4)

print(w_x3, "and", w_y3)

x = round((w_x-pose_x) * np.cos(theta) + (w_y-pose_y) * np.sin(theta),4)
y = round((w_x-pose_x) * -np.sin(theta) + (w_y-pose_y) * np.cos(theta),4)

print(x, "and", y)

"""