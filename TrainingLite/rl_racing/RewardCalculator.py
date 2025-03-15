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
        
        self.time = 0
        self.last_progress = 0
        self.last_progress_time = 0
        self.last_steering = 0
        self.spin_counter = 0
        self.stuck_counter = 0
        
        self.reward_history = []
        
    def _calculate_reward(self, driver: 'CarSystem') -> float:
        waypoint_utils: WaypointUtils = driver.waypoint_utils
        car_state = driver.car_state

        reward = 0

        # ✅ Reward Track Progress (Encourage Fast Forward Movement)
        progress = waypoint_utils.get_cumulative_lap_progress()
        delta_progress = progress - self.last_progress

        # 🛠 Fix: Ensure progress is within a reasonable range to avoid glitches
        if abs(delta_progress) > 0.2:
            print("BUG: Unexpected jump in progress!")
            delta_progress = 0  # Ignore unexpected jumps

        # ✅ Reward for Moving Forward (Scaled to Time)
        if delta_progress > 0:
            time_since_last_progress = self.time - self.last_progress_time
            if time_since_last_progress > 0.02:
                reward += (delta_progress / time_since_last_progress) * 500  # Higher for faster progress

            self.last_progress_time = self.time

        # ❌ Penalize Backward Movement
        if delta_progress < 0:
            reward += delta_progress * 1500  # Strong penalty for going backward

        # ✅ Reward Higher Speed (Faster is Better)
        speed = car_state[LINEAR_VEL_X_IDX]
        if speed > 2:  # Reward only when the car is moving at a decent pace
            reward += speed * 2  # Encourages the car to move fast

        # ❌ Penalize Stopping (Encourages Constant Motion)
        if speed < 0.5:
            reward -= 5  # Stronger penalty for being too slow

        # ✅ Penalize Sudden Steering (Encourage Stability)
        steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        reward -= steering_diff * 0.1  # Stronger penalty to discourage aggressive corrections

        # ✅ Penalize Large Steering Angle (Encourage Smooth Driving)
        steering_penalty = abs(car_state[STEERING_ANGLE_IDX]) * 0.02  # Scale penalty
        reward -= steering_penalty

        # ✅ Penalize Collisions (Avoid Crashes)
        lidar_penalty = sum(
            min(100, np.cos(angle) * 1 / range)
            for i, range in enumerate(driver.LIDAR.processed_ranges)
            if range < 0.5 and range != 0
        )
        reward -= lidar_penalty * 0.1

        # ✅ Penalize Spinning (Fixing Instability)
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 15.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                reward -= self.spin_counter * 0.5
            if self.spin_counter >= 200:
                reward -= 100
        else:
            self.spin_counter = 0

        # ✅ Penalize Being Stuck
        if speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 10:
                reward -= 10  # Stronger penalty for being stuck
            if self.stuck_counter >= 200:
                reward -= 100  # Even stronger penalty if it's really stuck
        else:
            self.stuck_counter = 0  # Reset if the car starts moving again

        # ✅ Update State
        self.last_steering = car_state[STEERING_ANGLE_IDX]
        self.last_progress = progress

        # ✅ Debug Info (Optional)
        if self.print_info and reward != 0:
            print(f"Reward: {reward}")

        # ✅ Save to History
        self.reward_history.append(reward)
        self.time += 0.01

        return reward
