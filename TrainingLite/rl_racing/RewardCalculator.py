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
        
        self.checkpoint_fraction = 1/20 # 1 / number of checkopoints that give reward. ATTENTION: must not be smaller than dist between waypoints
        
        
    def reset(self):
        
        self.time = 0
        self.last_progress : float = 0
        self.last_progress_time : float = 0
        self.last_steering = 0
        self.spin_counter = 0
        self.stuck_counter = 0
        self.last_wp_index = 0
        
        self.reward_components_history = []
        self.reward = 0
        
        self.reward_history = []
        
    def _calculate_reward(self, driver: 'CarSystem') -> float:

        waypoint_utils: WaypointUtils = driver.waypoint_utils
        car_state = driver.car_state
        reward = 0

        # Reward Track Progress (Encourage Fast Forward Movement)
        progress = waypoint_utils.get_cumulative_lap_progress() # 1 per complete lap in small steps (1/number of waypoints)
        delta_progress = progress - self.last_progress
    
        # Reward for Moving Forward (Scaled to Time)
        progress_reward = 0.0
        
        # Filter unrealistic progress jumps 
        if abs(delta_progress) > 1.5 * self.checkpoint_fraction + 0.02: 
            print("Unrealistic progress jump detected", delta_progress)
            delta_progress = 0
            self.last_progress_time = self.time
            self.last_progress = progress
            
        # Progresss reward
        if abs(delta_progress) > self.checkpoint_fraction:
            time_since_last_progress = self.time - self.last_progress_time
            if time_since_last_progress > 0.03:
                progress_reward += (delta_progress / time_since_last_progress)
                progress_reward *= self.checkpoint_fraction # Normalize by checkpoint size so the progress_reward is consistent
                progress_reward *= 20000 
            
            # print("Checkpoint reached, progress_reward: ", reward, "delta_progress: ", delta_progress, "time_since_last_progress: ", time_since_last_progress)
            self.last_progress_time = self.time
            self.last_progress = progress
            
            reward += progress_reward
            


        # if(reward != 0):
            # print(f"Progress: {progress}, Reward: {reward}")

        # Speed Reward (Encourage Fast Movement)
        speed = car_state[LINEAR_VEL_X_IDX]
        speed_reward = 0.0
        if speed > 2:  # Reward only when the car is moving at a decent pace
            speed_reward = speed * 0.005  # Scale the reward
        reward += speed_reward  # Encourages the car to move fast


        # delta Steering reward: Penalize Sudden Steering (Encourage Stability)
        steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        steering_diff_reward = steering_diff * - 0.1  # Scale the penalty
        reward += steering_diff_reward  # Stronger penalty to discourage aggressive corrections
        

        # Steering reward: Penalize Large Steering Angle (Encourage Smooth Driving)
        steering_penalty = - abs(car_state[STEERING_ANGLE_IDX]) * 0.02  # Scale penalty
        reward += steering_penalty

        # Lidar reward: Penalize beeing close to walls (Encourage Safe Driving)
        lidar_penalty = -sum(
            min(100, np.cos(driver.LIDAR.processed_angles_rad[i]) * 1 / range)
            for i, range in enumerate(driver.LIDAR.processed_ranges)
            if range < 0.5 and range != 0
        )
        reward += lidar_penalty * 0.1


        # Spinning reward Penalize Spinning (Fixing Instability)
        spin_reward = 0.0
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 15.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                spin_reward = -self.spin_counter * 0.5
            if self.spin_counter >= 200:
                spin_reward = -100
        else:
            self.spin_counter = 0
        reward += spin_reward

        # ✅ Penalize Being Stuck
        stuck_reward = 0.0
        if speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 20:
                stuck_reward = -10  # Stronger penalty for being stuck
            if self.stuck_counter >= 200:
                stuck_reward = -100  # Even stronger penalty if it's really stuck
        else:
            self.stuck_counter = 0  # Reset if the car starts moving again
        reward += stuck_reward

        # ✅ Update State
        self.last_steering = car_state[STEERING_ANGLE_IDX]

        # ✅ Debug Info (Optional)
        if self.print_info and reward != 0:
            print(f"Reward: {reward}")

        self.reward = reward
        
        
        
        if(Settings.SAVE_REWARDS):
            reward_components = {
                "progress": progress_reward,
                "speed": speed_reward,
                "steering_diff": steering_diff_reward,
                "steering_penalty": steering_penalty,
                "lidar_penalty": lidar_penalty,
                "stuck_reward": stuck_reward,
                "spin_reward": spin_reward,
                "total_reward": reward,
            }
            
            self.reward_components_history.append(reward_components)

        
        # ✅ Save to History
        self.reward_history.append(reward)
        self.time += 0.01
        return reward


    def plot_history(self, save_path: str):
        import matplotlib.pyplot as plt
        import numpy as np

        # Access reward components history
        reward_components_history = self.reward_components_history

        # Extract reward components
        steps = range(len(reward_components_history))
        reward_labels = ["progress", "speed", "steering_diff", "steering_penalty", "lidar_penalty", "stuck_reward","spin_reward", "total_reward"]
        reward_colors = ["blue", "green", "orange", "red", "purple","purple","purple", "black"]

        # Compute cumulative sums for each reward component
        cumulative_rewards = {
            label: np.cumsum([comp[label] for comp in reward_components_history])
            for label in reward_labels
        }

        # Create subplots
        fig, axes = plt.subplots(len(reward_labels), 1, figsize=(10, 15), sharex=True)
        fig.suptitle("Cumulative Reward Components Over Time", fontsize=16)

        # Plot each cumulative reward component in a loop
        for i, label in enumerate(reward_labels):
            axes[i].plot(steps, cumulative_rewards[label], label=f"Cumulative {label.capitalize()} Reward", color=reward_colors[i])
            axes[i].set_ylabel(f"{label.capitalize()} Reward")
            axes[i].legend()

        # Set x-axis label for the last subplot
        axes[-1].set_xlabel("Step")

        # Save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, "cumulative_reward_components.png"))
