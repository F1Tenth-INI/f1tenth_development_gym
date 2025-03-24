import numpy as np
import pytest
from numba import njit

# Import the function (assuming it's in the same module)
from utilities.waypoint_utils import get_nearest_waypoint  

# Define indices for car state
POSE_X_IDX = 6
POSE_Y_IDX = 7

# Define indices for waypoints
WP_X_IDX = 1
WP_Y_IDX = 2

def test_get_nearest_waypoint_basic():
    """Test basic nearest waypoint detection."""
    car_state = np.array([0, 0, 0, 0, 0, 0, 10.0, 10.0, 0, 0])
    waypoints = np.array([
        [0, 0.0, 0.0],
        [1, 5.0, 5.0],
        [2, 10.1, 10.1],  # Closest
        [3, 20.0, 20.0],
    ])
    nearest_idx, nearest_dist = get_nearest_waypoint(car_state, waypoints)
    assert nearest_idx == 2, f"Expected 2, got {nearest_idx}"
    assert np.isclose(nearest_dist, 0.02, atol=1e-2)

def test_get_nearest_waypoint_wrap_around():
    """Test wrap-around indexing when searching within limits."""
    car_state = np.array([0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, 0])
    waypoints = np.array([
        [0, 10.0, 10.0],
        [1, 20.0, 20.0],
        [2, 0.5, 0.5],  # Closest
        [3, 30.0, 30.0],
    ])
    nearest_idx, _ = get_nearest_waypoint(car_state, waypoints, last_nearest_waypoint_index=1, lower_search_limit=-2, upper_search_limit=2)
    assert nearest_idx == 2, f"Expected 2, got {nearest_idx}"

def test_get_nearest_waypoint_large_waypoints():
    """Test performance with a large number of waypoints."""
    car_state = np.array([0, 0, 0, 0, 0, 0, 50.0, 50.0, 0, 0])
    waypoints = np.array([[i, i * 1.0, i * 1.0] for i in range(1000)])
    nearest_idx, nearest_dist = get_nearest_waypoint(car_state, waypoints)
    assert nearest_idx == 50, f"Expected 50, got {nearest_idx}"
    assert np.isclose(nearest_dist, 0.0, atol=1e-2)

if __name__ == "__main__":
    pytest.main()
