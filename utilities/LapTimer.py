import numpy as np


class LapTimer:
    def __init__(self, total_waypoints, single_measurement_point=True, lap_finished_callback=None):
        """
        Initializes the LapTimer with the total number of waypoints and optional settings.

        Args:
            total_waypoints (int): Total number of waypoints in the racetrack.
            single_measurement_point (bool): If True, measures the lap time at a single point.
            lap_finished_callback (callable): Function to call when a lap is finished.
        """
        self.current_lap_time = None  # Stores the currently registered lap time
        self.waypoint_log = []        # Logs the indices of waypoints passed
        self.time_log = []            # Logs the times when waypoints were passed
        self.ready_for_readout = []   # Flags indicating if a waypoint should trigger a lap, when revisited (because car was registered at the opposite side of the track, since last visiting this waypoint)
        self.total_waypoints = total_waypoints  # Total number of waypoints on the track

        self.single_measurement_point = single_measurement_point
        self.single_measurement_point_index = None  # Index of the measurement point if measurement at single point
        self.single_measurement_point_time = None   # Time when the measurement point was last passed

        self.lap_finished_callback = lap_finished_callback  # Callback function to call when lap is finished (e.g. print the lap time to terminal)

    def update(self, nearest_waypoint_index, time_now):
        """
        Updates the lap timer with the latest waypoint index and current time.
        Calculate the lap time and call the callback function if full lap completed
        Args:
            nearest_waypoint_index (int): The index of the nearest waypoint passed.
            time_now (float): The current time when the waypoint was passed.
        """

        if not self.waypoint_log:
            # If no waypoints have been logged yet, initialize the logs
            self.waypoint_log.append(nearest_waypoint_index)
            self.time_log.append(time_now)
            self.ready_for_readout.append(False)
            self.single_measurement_point_index = nearest_waypoint_index
            self.single_measurement_point_time = time_now
            return  # Need at least two waypoints to start calculating lap times

        # Handle potential reverse movement by removing waypoints that indicate going backward
        while self.waypoint_log:
            last_waypoint = self.waypoint_log[-1]
            mod_diff = nearest_waypoint_index - last_waypoint
            if mod_diff < 0:
                mod_diff += self.total_waypoints  # Adjust for circular track

            if mod_diff > 3 * self.total_waypoints // 4:
                # If the difference is too large, it's likely reverse movement; remove the last waypoint
                self.waypoint_log.pop()
                self.time_log.pop()
                self.ready_for_readout.pop()
            else:
                break  # Exit if movement is forward

        if not self.waypoint_log:
            # After removing reverse waypoints, if log is empty, re-initialize
            self.waypoint_log.append(nearest_waypoint_index)
            self.time_log.append(time_now)
            self.ready_for_readout.append(False)
            self.single_measurement_point_index = nearest_waypoint_index
            self.single_measurement_point_time = time_now
            return  # Need at least two waypoints to start calculating lap times

        # Convert waypoint log to a NumPy array for efficient calculations
        waypoint_log_array = np.array(self.waypoint_log)

        # Calculate distance (in units of waypoints indices) from each logged waypoint to the current waypoint, considering track wrap-around
        direct_distance = nearest_waypoint_index - waypoint_log_array
        direct_distance = np.where(direct_distance < 0, direct_distance + self.total_waypoints, direct_distance)

        # If the latest waypoint is the same as the current and not ready for readout, skip processing
        if direct_distance[-1] == 0 and not self.ready_for_readout[-1]:
            return

        # Mark waypoints as ready for readout if sufficiently far behind the current waypoint
        for i in range(len(direct_distance)):
            if direct_distance[i] > self.total_waypoints // 2:
                self.ready_for_readout[i] = True

        # Identify indices of waypoints that mark a lap completion
        indices_to_check = [
            i for i, (distance, ready) in enumerate(zip(direct_distance, self.ready_for_readout))
            if distance < self.total_waypoints // 4 and ready
        ]

        if indices_to_check:
            # If there are eligible waypoints, determine the latest one to consider for lap completion
            max_index = max(indices_to_check)
            if self.single_measurement_point:
                if self.time_log[max_index] >= self.single_measurement_point_time:
                    # Calculate lap time based on the single measurement point
                    self.current_lap_time = time_now - self.time_log[max_index]
                    # Update the single measurement point to the current waypoint and time
                    self.single_measurement_point_index = nearest_waypoint_index
                    self.single_measurement_point_time = time_now
                    if self.lap_finished_callback is not None:
                        # Trigger the callback with the lap time
                        self.lap_finished_callback(self.current_lap_time)
            else:
                # If not using a single measurement point, calculate lap time based on the identified waypoint
                self.current_lap_time = time_now - self.time_log[max_index]
                if self.lap_finished_callback is not None:
                    # Trigger the callback with the lap time
                    self.lap_finished_callback(self.current_lap_time)

            # Remove all waypoints up to and including the one that marked the lap completion
            self.waypoint_log = self.waypoint_log[max_index + 1:]
            self.time_log = self.time_log[max_index + 1:]
            self.ready_for_readout = self.ready_for_readout[max_index + 1:]

        # Append the current waypoint and time to the logs for future lap calculations
        self.waypoint_log.append(nearest_waypoint_index)
        self.time_log.append(time_now)
        self.ready_for_readout.append(False)
