import numpy as np


class WaypointHistoryManager:
    """Handles the storage and retrieval of waypoint states to support undo operations."""
    def __init__(self):
        # Holds a list of tuples representing historical waypoint states.
        # Each tuple: (x_array, y_array) or (x_array, y_array, vx_array) if vx exists.
        self.history = []
        self.max_history_length = 10

    def save_waypoint_state(self, x, y, vx=None):
        """
        Saves a copy of the current waypoints state.
        Ensures we don't store the same state consecutively.
        """
        # If we have at least one state, check if the current one is identical
        if self.history:
            last_state = self.history[-1]
            # Compare arrays from the last state with the current arrays
            # If vx present, it should also match for a no-save scenario
            if vx is not None:
                if (np.array_equal(last_state[0], x) and
                    np.array_equal(last_state[1], y) and
                    np.array_equal(last_state[2], vx)):
                    return
            else:
                if (np.array_equal(last_state[0], x) and
                    np.array_equal(last_state[1], y)):
                    return

        # Append new state to history
        if vx is not None:
            self.history.append((x.copy(), y.copy(), vx.copy()))
        else:
            self.history.append((x.copy(), y.copy()))

        # Limit history size to avoid excessive memory use
        if len(self.history) > self.max_history_length:
            self.history.pop(0)

    def undo(self):
        """
        Reverts to the previous waypoint state if available.
        Returns the restored state as a tuple, or None if no undo is possible.
        """
        if len(self.history) > 1:
            # Remove the current state
            self.history.pop()
            # Return the last saved state
            return self.history[-1]
        return None
