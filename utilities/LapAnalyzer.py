import numpy as np
from utilities.waypoint_utils import *
from utilities.state_utilities import *

class LapTimer:
    def __init__(self, total_waypoints, single_measurement_point=True, lap_finished_callback=None):
        self.current_lap_time = None
        self.waypoint_log = []  # Stores indices of waypoints passed
        self.distance_log = []  # Stores distances to waypoints passed
        self.time_log = []      # Stores times when waypoints were passed
        self.ready_for_readout = []
        self.total_waypoints = total_waypoints

        self.single_measurement_point = single_measurement_point
        self.single_measurement_point_index = None
        self.single_measurement_point_time = None

        self.lap_finished_callback = lap_finished_callback
        
    def get_distance_stats(self):
        if not self.waypoint_log:
            return None, None, None
     
        mean_distance = np.mean(self.distance_log)
        std_distance = np.std(self.distance_log)
        max_distance = np.max(self.distance_log)
        
        return mean_distance, std_distance, max_distance



    def update(self, nearest_waypoint, time_now, distance):

        nearest_waypoint_index = nearest_waypoint[WP_GLOBID_IDX]    
        
        self.distance_log.append(distance)     
        
        self.current_lap_time = None

        if not self.waypoint_log:
            # Append the waypoint and time to the logs
            self.waypoint_log.append(nearest_waypoint_index)
            self.time_log.append(time_now)
            self.ready_for_readout.append(False)
            self.single_measurement_point_index = nearest_waypoint_index
            self.single_measurement_point_time = time_now
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
            self.single_measurement_point_index = nearest_waypoint_index
            self.single_measurement_point_time = time_now
            return  # Not enough data to compare


        # Calculate distance, considering wrapping
        waypoint_log_array = np.array(self.waypoint_log)

        # Calculate distance, considering wrapping
        direct_distance = nearest_waypoint_index - waypoint_log_array
        direct_distance = np.where(direct_distance < 0, direct_distance + self.total_waypoints, direct_distance)

        # If the last waypoint is the same as the current waypoint, return
        if direct_distance[-1] == 0 and not self.ready_for_readout[-1]:
            return

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
            if self.single_measurement_point:
                if self.time_log[max_index] >= self.single_measurement_point_time:
                    # lap finished
                    self.current_lap_time = time_now - self.time_log[max_index]
                    self.single_measurement_point_index = nearest_waypoint_index
                    self.single_measurement_point_time = time_now
                    if self.lap_finished_callback is not None:
                        mean_distance, std_distance, max_distance = self.get_distance_stats()
                        self.lap_finished_callback(self.current_lap_time, mean_distance, std_distance, max_distance)
            else:
                self.current_lap_time = time_now - self.time_log[max_index]
                if self.lap_finished_callback is not None:
                    mean_distance, std_distance, max_distance = self.get_distance_stats()
                    self.lap_finished_callback(self.current_lap_time, mean_distance, std_distance, max_distance)

            # Delete all entries with indices <= max_index
            self.waypoint_log = self.waypoint_log[max_index + 1:]
            self.time_log = self.time_log[max_index + 1:]
            self.ready_for_readout = self.ready_for_readout[max_index + 1:]

        # Append the waypoint and time to the logs
        self.waypoint_log.append(nearest_waypoint_index)
        self.time_log.append(time_now)
        self.ready_for_readout.append(False)
        
        
   