import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
from matplotlib.image import imread
from matplotlib.widgets import Slider
from copy import deepcopy

import datetime  # For timestamping

import matplotlib
import yaml

# path_to_maps = "../../racecar/racecar/maps/"
# map_name = "IPZ15"
path_to_maps = "utilities/maps/"
map_name = "RCA1"

waypoints_new_file_name = None  # If None, overwrites
scale_initial = 20.0  # Smoothness parameter for Gaussian weight

if sys.platform == 'darwin':
    matplotlib.use('MacOSX')

plt.rcParams.update({'font.size': 15})

path_to_map_png = os.path.join(path_to_maps, map_name, map_name + ".png")
path_to_map_config = os.path.join(path_to_maps, map_name, map_name + ".yaml")
path_to_waypoints = os.path.join(path_to_maps, map_name, map_name + "_wp.csv")

# Global variables for modularity
x, y = None, None  # Current waypoints
initial_x, initial_y = None, None  # Initial waypoints
cs_x, cs_y = None, None  # Cubic splines
dense_t, dense_x, dense_y = None, None, None  # Dense points for plotting
dragging, drag_index = False, None  # Interaction state
image_loaded = False
scale = scale_initial  # Initial scale value for Gaussian weight
text_box = None
original_data = None



def load_waypoints(csv_path):
    """Load waypoints from a CSV file."""
    global x, y, t, cs_x, cs_y, dense_t, dense_x, dense_y, initial_x, initial_y, original_data
    original_data = pd.read_csv(csv_path, comment="#")
    original_data.columns = original_data.columns.str.strip()
    
    
    x = original_data['x_m'].to_numpy()
    y = original_data['y_m'].to_numpy()
    # Save the initial waypoints
    initial_x = x.copy()
    initial_y = y.copy()
    t = np.arange(len(x))  # Parameterize waypoints
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    dense_t = np.linspace(t[0], t[-1], 500)
    dense_x = cs_x(dense_t)
    dense_y = cs_y(dense_t)

def redraw_plot():
    """Redraw the plot with updated waypoints and spline."""
    ax.clear()
    if image_loaded:
        load_image_background(path_to_map_png, path_to_map_config)
    # Plot the initial blue line connecting the original waypoints
    ax.plot(initial_x, initial_y, color="blue", label="Initial Waypoints Line", linestyle="--")
    # Plot the dynamic green line connecting the updated waypoints
    ax.plot(x, y, color="green", label="Adjusted Waypoints Line", linestyle="-")
    # Plot the waypoints
    ax.scatter(x, y, color="red", label="Waypoints")
    ax.legend()
    plt.draw()

    # Display text in the text box (example message)
    update_text_box("Waypoints updated. Current scale: {:.1f}".format(scale))


waypoint_history = []

def save_waypoint_state():
    """
    Save the current state of waypoints to the history stack.
    If the stack exceeds 10 states, remove the oldest state.
    Avoid saving duplicate consecutive states.
    """
    global waypoint_history
    if waypoint_history:
        last_state = waypoint_history[-1]
        if np.array_equal(last_state[0], x) and np.array_equal(last_state[1], y):
            return  # Do not save duplicate state
    # Save current state
    waypoint_history.append((x.copy(), y.copy()))
    # Limit history to last 10 states
    if len(waypoint_history) > 10:
        waypoint_history.pop(0)

def undo_changes(event):
    """
    Undo the last change to waypoints if 'CMD+Z' or 'Ctrl+Z' is pressed.

    Args:
        event: The key press event from matplotlib.
    """
    global x, y, waypoint_history

    # # Debug: Print the detected key
    # print(f"Key pressed: {event.key}")

    # Normalize the key to lowercase for consistency
    key = event.key.lower() if event.key else ''

    # Check for CMD+Z (Mac) or Ctrl+Z (Windows/Linux)
    if key in ["ctrl+z", "cmd+z"]:
        if len(waypoint_history) > 1:
            # Remove the current state
            waypoint_history.pop()
            # Restore the previous state
            x, y = deepcopy(waypoint_history[-1])
            recalculate_splines()
            redraw_plot()
            update_text_box("Undo successful. Reverted to previous waypoint state.")
        else:
            update_text_box("No more undo steps available.")
    else:
        pass
        # print("Undo key combination not detected.")

def load_image_background(image_path, yaml_path, grayscale=True):
    """
    Load and display an image as the background, using the resolution
    and origin from the corresponding YAML file. Optionally, display in grayscale.

    Args:
        image_path (str): Path to the background image file.
        yaml_path (str): Path to the YAML file containing origin and resolution.
        grayscale (bool): Whether to display the image in grayscale. Default is True.
    """
    global ax, image_loaded

    # Load the image
    img = imread(image_path)

    # Convert to grayscale if specified
    if grayscale and img.ndim == 3:  # Check if the image is RGB
        img = np.dot(img[..., :3], [0.2989, 0.587, 0.114])  # Grayscale conversion formula

    # Load the YAML file
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    resolution = yaml_data['resolution']
    origin = yaml_data['origin']  # Assuming [origin_x, origin_y]

    # Compute real-world extents based on resolution and origin
    extent = [
        origin[0],  # Left boundary (in meters)
        origin[0] + img.shape[1] * resolution,  # Right boundary
        origin[1],  # Bottom boundary
        origin[1] + img.shape[0] * resolution  # Top boundary
    ]

    # Display the image with correct scaling
    ax.imshow(img, extent=extent, aspect='auto', cmap='gray' if grayscale else None)
    image_loaded = True


def recalculate_splines():
    """Recompute splines with the updated waypoints."""
    global cs_x, cs_y, dense_x, dense_y
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    dense_x = cs_x(dense_t)
    dense_y = cs_y(dense_t)



def apply_weighted_adjustment(dx, dy):
    """Distribute the adjustment to neighbors using a Gaussian weight."""
    global x, y
    n = len(x)
    for i in range(len(x)):
        # Compute distance in parameter space (cyclic adjacency)
        d = min(abs(i - drag_index), abs(i - drag_index + n), abs(i - drag_index - n))
        weight = np.exp(-0.5 * (d / scale) ** 2)  # Gaussian weight
        x[i] += dx * weight
        y[i] += dy * weight


def on_press(event):
    """Handle mouse press for dragging."""
    global dragging, drag_index
    if event.inaxes != ax:
        return
    for i, (px, py) in enumerate(zip(x, y)):
        if np.hypot(event.xdata - px, event.ydata - py) < 0.3:  # Adjust sensitivity
            dragging = True
            drag_index = i
            break


def on_release(event):
    """Handle mouse release to stop dragging."""
    global dragging, drag_index
    if dragging:
        dragging = False
        drag_index = None
        # Recompute splines with updated waypoints
        recalculate_splines()
        save_waypoint_state()
        redraw_plot()
        print("Mouse released. Waypoint state saved.")

def on_motion(event):
    """Handle mouse motion for dragging."""
    global dragging, drag_index
    if dragging and drag_index is not None:
        # Compute adjustment based on mouse movement
        dx = event.xdata - x[drag_index]
        dy = event.ydata - y[drag_index]
        # Apply weighted adjustment
        apply_weighted_adjustment(dx, dy)
        # Redraw the plot
        redraw_plot()


# Global variable to keep track of the slider object
sigma_slider = None


def update_sigma(val):
    """
    Update the global scale value from the slider and redraw the plot.

    Args:
        val (float): The current value of the slider.
    """
    global scale
    scale = val
    update_text_box(f"Scale updated to: {scale}")
    redraw_plot()

def update_text_box(message):
    """
    Update the text box with a new message.

    Args:
        message (str): The text to display in the box.
    """
    if text_box:
        text_box.clear()  # Clear the previous message
        text_box.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        text_box.set_xticks([])
        text_box.set_yticks([])
        plt.draw()


def save_waypoints(event):
    """
    Save waypoints to a CSV file when CMD+S or Ctrl+S is pressed.

    Args:
        event: The key press event from matplotlib.
    """
    global waypoints_new_file_name

    key = event.key.lower() if event.key else ''

    # Check for CMD+S (Mac) or Ctrl+S (Windows/Linux)
    if key in ["ctrl+s", "cmd+s"]:
        # Determine file path
        file_path = waypoints_new_file_name if waypoints_new_file_name else path_to_waypoints

        # Create a DataFrame with updated x and y
        data = pd.DataFrame({
            "x_m": x,
            "y_m": y
        })

        # Read the original file to retain column order and additional columns

        for col in original_data.columns:
            if col not in data.columns:
                data[col] = original_data[col]

        # Reorder columns to match the original order
        data = data[original_data.columns]

        # Add a comment at the top of the file with the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_comment = f"# Updated waypoints saved on {timestamp}\n"

        # Write the data to the file with the timestamp comment
        with open(file_path, "w") as f:
            f.write(header_comment)
            f.write("# " + ", ".join(data.columns) + "\n")
            data.to_csv(f, index=False, float_format="%.6f", header=False)

        # Update text box with save confirmation
        update_text_box(f"Waypoints saved to {file_path} at {timestamp}")

        print(f"Waypoints saved to {file_path} at {timestamp}")


def key_press_handler(event):
    """
    General key press handler to route events.

    Args:
        event: The key press event from matplotlib.
    """
    save_waypoints(event)  # Handle save functionality
    undo_changes(event)  # Handle undo functionality


def create_backup_if_needed(original_path):
    """
    Create a backup of the original waypoints file if it does not already exist.
    The backup file will have the suffix '_backup' added to its name.

    Args:
        original_path (str): Path to the original waypoints file.
    """
    backup_path = original_path.replace(".csv", "_backup.csv")
    if not os.path.exists(backup_path):
        # Copy the original file to the backup location
        with open(original_path, 'r') as original_file:
            with open(backup_path, 'w') as backup_file:
                backup_file.write(original_file.read())
        print(f"Backup created: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")



# Update `main()` to connect the new save functionality
def main():
    global fig, ax, sigma_slider, text_box

    # Ensure backup of waypoints file
    create_backup_if_needed(path_to_waypoints)

    # Load data
    load_waypoints(path_to_waypoints)
    # Initially save the state before any modifications
    save_waypoint_state()

    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.canvas.manager.set_window_title("INIvincible Waypoints Editor")

    plt.subplots_adjust(bottom=0.2)  # Leave space for the slider
    load_image_background(path_to_map_png, path_to_map_config)
    redraw_plot()
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid()

    # Add a slider for modifying scale
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # Adjusted position for slider
    sigma_slider = Slider(ax_slider, "Scale", 1.0, 50.0, valinit=scale, valstep=0.1)
    sigma_slider.on_changed(update_sigma)

    # Add a text box below the plot
    text_box = plt.axes([0.1, 0.05, 0.8, 0.05])  # [left, bottom, width, height]
    update_text_box("Drag waypoints to change, CMD+S or Ctrl+S to save, CMD+Z or Ctrl+Z to undo.")

    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", key_press_handler)  # General key handler

    plt.show()


# Run the program
if __name__ == "__main__":
    main()
