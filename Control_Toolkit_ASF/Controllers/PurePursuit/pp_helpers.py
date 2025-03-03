
import sys
sys.path.insert(1, 'FollowtheGap')

import numpy as np
import math
from utilities.Settings import Settings
from utilities.render_utilities import RenderUtils

from utilities.waypoint_utils import WaypointUtils


from numba import jit, njit, prange
import numpy as np

def get_current_waypoint(waypoints, lookahead_distance, position, theta):
    """
    Optimized function to get the current waypoint using JIT compilation.
    """
    wpts = waypoints[:, 1:3]  # Extract x, y from waypoints

    # Find the nearest point on the trajectory
    nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)

    if nearest_dist < lookahead_distance:
        # Ensure i2 is initialized as a valid integer
        lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i + t, wrap=True)
        
        # Ensure i2 is within bounds
        if i2 is None or i2 < 0:
            i2 = len(wpts) - 1  # Use last waypoint if invalid

        i2 = min(i2, len(wpts) - 1)  # Ensure i2 never exceeds array bounds

        # Get the current waypoint: x, y, and speed
        current_waypoint = np.zeros(3, dtype=np.float32)
        current_waypoint[0] = wpts[i2, 0]
        current_waypoint[1] = wpts[i2, 1]
        current_waypoint[2] = waypoints[min(i, len(waypoints) - 1), 5]  # Ensure valid speed index

        return current_waypoint, i, max(i2, 8)

    elif nearest_dist < 20:  # max reacquire distance
        current_waypoint = np.zeros(3, dtype=np.float32)
        current_waypoint[0] = wpts[i, 0]
        current_waypoint[1] = wpts[i, 1]
        current_waypoint[2] = waypoints[min(i, len(waypoints) - 1), 5]  # Ensure valid index

        return current_waypoint, i, i
    else:
        return np.zeros(3, dtype=np.float32), -1, -1  # Return a valid empty array instead of None
    


@njit(fastmath=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation using JIT acceleration
    """
    waypoint_y = np.sin(-pose_theta) * (lookahead_point[0] - position[0]) + np.cos(-pose_theta) * (lookahead_point[1] - position[1])
    speed = lookahead_point[2]
    if abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


@njit(fastmath=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Finds the nearest point along the given trajectory using Numba with parallel execution.
    """
    num_segments = trajectory.shape[0] - 1

    # Compute segment vectors without ascontiguousarray
    diffs = trajectory[1:] - trajectory[:-1]
    l2s = diffs[:, 0]**2 + diffs[:, 1]**2

    # Initialize arrays correctly
    dots = np.zeros(num_segments, dtype=np.float32)
    t = np.zeros(num_segments, dtype=np.float32)
    projections = np.zeros((num_segments, 2), dtype=np.float32)
    dists = np.zeros(num_segments, dtype=np.float32)

    # Compute dot products and projection factors (parallelized)
    for i in prange(num_segments):
        dots[i] = np.dot(point - trajectory[i], diffs[i])
        if l2s[i] > 1e-6:
            t[i] = max(0.0, min(1.0, dots[i] / l2s[i]))

    # Compute projections manually
    for i in prange(num_segments):
        projections[i, 0] = trajectory[i, 0] + t[i] * diffs[i, 0]
        projections[i, 1] = trajectory[i, 1] + t[i] * diffs[i, 1]

    # Compute distances manually
    for i in prange(num_segments):
        dx = projections[i, 0] - point[0]
        dy = projections[i, 1] - point[1]
        dists[i] = np.sqrt(dx**2 + dy**2)

    # Find the closest projection
    min_idx = np.argmin(dists)

    return projections[min_idx], dists[min_idx], t[min_idx], min_idx


@njit(fastmath=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    Finds the first point where a trajectory intersects with a circle using fast JIT execution.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t, first_i, first_p = None, None, None

    for i in range(start_i, trajectory.shape[0] - 1):
        start, end = trajectory[i, :], trajectory[i + 1, :]
        V = end - start

        # Ensure arrays are contiguous
        V = np.ascontiguousarray(V)
        start = np.ascontiguousarray(start)
        point = np.ascontiguousarray(point)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        if i == start_i:
            if 0.0 <= t1 <= 1.0 and t1 >= start_t:
                first_t, first_i, first_p = t1, i, start + t1 * V
                break
            if 0.0 <= t2 <= 1.0 and t2 >= start_t:
                first_t, first_i, first_p = t2, i, start + t2 * V
                break
        elif 0.0 <= t1 <= 1.0:
            first_t, first_i, first_p = t1, i, start + t1 * V
            break
        elif 0.0 <= t2 <= 1.0:
            first_t, first_i, first_p = t2, i, start + t2 * V
            break

    return first_p, first_i, first_t


