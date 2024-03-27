import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import yaml
from matplotlib.colors import LinearSegmentedColormap



from scipy.ndimage import rotate

# NN
current_dir = os.path.dirname(os.path.abspath(__file__))

# map_name = "RCA1"
# offset = -150

map_name = "RCA2"
offset = -620

map_path = "/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/utilities/maps/" + map_name
wp_file = os.path.join(map_path ,map_name + '_wp.csv') 
print(wp_file)   
wp_data = pd.read_csv(wp_file)
    
wp_x = wp_data['x_m'].to_numpy()
wp_y = wp_data['y_m'].to_numpy()
wp_kappa = wp_data['kappa_radpm'].to_numpy()
wp_kappa = np.roll(wp_kappa, offset)

window_size = 10
window = np.ones(window_size) / window_size
wp_kappa = np.convolve(wp_kappa, window, mode='same')

wp_kappa_derivative = np.gradient(wp_kappa)

fig, ax1 = plt.subplots(figsize=(10, 6))  # Create a new figure and a set of subplots

color = 'tab:blue'
ax1.set_xlabel('Waypoint Index')  # Set the x-axis label
ax1.set_ylabel('Curvature (kappa) [rad/m]', color=color)  # Set the y-axis label
ax1.plot(wp_kappa, color=color, linewidth=2)  # Plot wp_kappa
ax1.tick_params(axis='y', labelcolor=color)  # Set the color of the y-axis labels

ax2 = ax1.twinx()  # Create a second y-axis that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Derivative of Curvature (dkappa) [rad/m^2]', color=color)  # Set the y-axis label for the second axis
ax2.plot(wp_kappa_derivative, color=color, linewidth=2, linestyle='--')  # Plot the derivative of wp_kappa on the second y-axis
ax2.tick_params(axis='y', labelcolor=color)  # Set the color of the y-axis labels for the second axis

# Get the limits of the y-axes
kappa_min, kappa_max = ax1.get_ylim()
derivative_min, derivative_max = ax2.get_ylim()

# Calculate the absolute maximum values for both axes
abs_max = max(abs(kappa_min), abs(kappa_max), abs(derivative_min), abs(derivative_max))

# Set the limits of the y-axes so that 0 is aligned
ax1.set_ylim(-abs_max, abs_max)
ax2.set_ylim(-0.25 * abs_max, 0.25 * abs_max)

fig.tight_layout()  # Adjust the layout so everything fits
plt.title('Curvature and its Derivative over Waypoints')  # Set the title of the plot
plt.savefig('curvature_plot'+map_name+'.png')  # Save the plot as a .png file