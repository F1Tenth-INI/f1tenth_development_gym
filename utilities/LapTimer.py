import numpy as np


class LapTimer:
    def __init__(self, total_waypoints):
        self.current_lap_time = None
        self.waypoint_log = []  # Stores indices of waypoints passed
        self.time_log = []      # Stores times when waypoints were passed
        self.ready_for_readout = []
        self.total_waypoints = total_waypoints

    def update(self, nearest_waypoint_index, time_now):

        if not self.waypoint_log:
            # Append the waypoint and time to the logs
            self.waypoint_log.append(nearest_waypoint_index)
            self.time_log.append(time_now)
            self.ready_for_readout.append(False)
            return  # Not enough data to compare

        # Check for reverse movement
        while self.waypoint_log:
            last_waypoint = self.waypoint_log[-1]
            mod_diff = nearest_waypoint_index - last_waypoint
            if mod_diff < 0:
                mod_diff += self.total_waypoints

            if mod_diff > 3 * self.total_waypoints // 4:
                self.waypoint_log.pop()
                self.time_log.pop()
                self.ready_for_readout.pop()
            else:
                break

        if not self.waypoint_log:
            # Append the waypoint and time to the logs
            self.waypoint_log.append(nearest_waypoint_index)
            self.time_log.append(time_now)
            self.ready_for_readout.append(False)
            return  # Not enough data to compare


        # Calculate distance, considering wrapping
        waypoint_log_array = np.array(self.waypoint_log)

        # Calculate distance, considering wrapping
        direct_distance = nearest_waypoint_index - waypoint_log_array
        direct_distance = np.where(direct_distance < 0, direct_distance + self.total_waypoints, direct_distance)

        # If the last waypoint is the same as the current waypoint, return
        if direct_distance[-1] == 0:
            return

        # Append the waypoint and time to the logs
        self.waypoint_log.append(nearest_waypoint_index)
        self.time_log.append(time_now)
        self.ready_for_readout.append(False)


        # Checking which waypoints are already enough away to count their revisiting as a new lap
        for i in range(len(direct_distance)):
            if direct_distance[i] > self.total_waypoints // 2:
                self.ready_for_readout[i] = True  # Never flip True back to False

        # Check for entries to delete - finished laps
        indices_to_check = [
            i for i, (distance, ready) in enumerate(zip(direct_distance, self.ready_for_readout))
            if distance < self.total_waypoints//4 and ready
        ]
        if indices_to_check:
            # Find the largest index that satisfies the condition
            max_index = max(indices_to_check)
            self.current_lap_time = self.time_log[-1] - self.time_log[max_index]
            # Delete all entries with indices <= max_index
            self.waypoint_log = self.waypoint_log[max_index + 1:]
            self.time_log = self.time_log[max_index + 1:]
            self.ready_for_readout = self.ready_for_readout[max_index + 1:]

