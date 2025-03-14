import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import *

class RewardCalculator:
    def __init__(self):
        
        self.print_info = False
        self.reset()
        
    def reset(self):
        self.last_progress = 0
        self.last_steering = 0
        self.spin_counter = 0
        self.stuck_counter = 0
        
        self.reward_history = []
        
    def _calculate_reward(self, driver: 'CarSystem') -> float:
        waypoint_utils = driver.waypoint_utils
        car_state = driver.car_state

        reward = 0

        # Reward Track Progress (Incentivize Moving Forward)
        progress = waypoint_utils.get_cumulative_progress()
        delta_progress = progress - self.last_progress
        progress_reward = delta_progress * 200.0 + progress * 0.25
        if progress_reward < 0:
            progress_reward *= 3.0  # Increase penalty for going backwards
        if delta_progress > 0.2:  # Progress can not be that high ( prevent car from actively skipping path)
            progress_reward = 0
            
        if delta_progress < -0.9: # Lap complete
            progress_reward = 5
            # print("Lap complete")
            
        
        reward += progress_reward
        # print(f"Progress: {progress}, Reward: {progress_reward}")
        

        # ✅ 2. Reward Maintaining Speed (Prevent Stops and Stalls)
        speed = car_state[LINEAR_VEL_X_IDX]
        speed_reward = speed * 0.1    
        reward += speed_reward
        

        # # ✅ 3. Penalize Sudden Steering Changes
        steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        reward -= steering_diff * 0.05  # Increased penalty to discourage aggressive corrections
        

        
        # Penalize slip
        vy = abs(car_state[LINEAR_VEL_Y_IDX])
        reward -= vy * 0.1
        
        # Penalize Steering Angle
        steering_penalty = abs(car_state[STEERING_ANGLE_IDX]) * 0.01  # Scale penalty
        reward -= steering_penalty

        # Penalize Collisions
        # if self.simulation.obs["collisions"][0] == 1:
        #     reward -= 100  # Keep high penalty for crashes

        # ✅ 5. Penalize Distance from Raceline
        # nearest_waypoint_index, nearest_waypoint_dist = get_nearest_waypoint(car_state, waypoint_utils.next_waypoints)
        # if nearest_waypoint_dist < 0.05:
        #     nearest_waypoint_dist = 0
        # wp_penality = nearest_waypoint_dist * 1
        # reward -= wp_penality
        # if print:
            # print(f"Nearest waypoint distance: {nearest_waypoint_dist}, penality: {-wp_penality}")

        # Penalize if lidar scans are < 0.5 and penalize more for even less ranges
        # Find all ranges < 0.5
        lidar_penality = 0
        for i, range in enumerate(driver.LIDAR.processed_ranges):
            angle = driver.LIDAR.processed_angles_rad[i]
            if range < 0.5 and range != 0:
                lidar_penality += min(100, np.cos(angle) * 1 / range)
                    
        reward -= lidar_penality * 0.1

        #  Penalize Spinning (Fixing Instability)
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 15.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                reward -= self.spin_counter * 0.5
            
            if self.spin_counter >= 200:
                reward -= 100

        else:
            self.spin_counter = 0
            
            
        # Penalize beeing stuck
        if abs(car_state[LINEAR_VEL_X_IDX]) < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 10:
                reward -= 5
            
            if self.stuck_counter >= 200:
                reward -= 5

       
        self.last_steering = car_state[STEERING_ANGLE_IDX]
        self.last_progress = progress
        
        if(self.print_info):
            print(f"Reward: {reward}")
            
        self.reward_history.append(reward)
        return reward
        