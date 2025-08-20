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
        
        self.checkpoint_fraction = 1/400 # 1 / number of checkopoints that give reward. ATTENTION: must not be smaller than dist between waypoints
        
        
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
        self.last_position = None
        self.reward_history = []
        
    def _calculate_reward(self, driver: 'CarSystem') -> float:
        waypoint_utils: WaypointUtils = driver.waypoint_utils
        car_state = driver.car_state
        reward = 0

        # Get current progress
        if self.last_position is None:
            self.last_position = [car_state[POSE_X_IDX], car_state[POSE_Y_IDX]]

        current_position = [car_state[POSE_X_IDX], car_state[POSE_Y_IDX]]
        delta_position = np.array(current_position) - np.array(self.last_position)

        wpts = waypoint_utils.next_waypoints
        closest_wp_pos = [wpts[0][WP_X_IDX], wpts[0][WP_Y_IDX]]
        next_wp_pos = [wpts[1][WP_X_IDX], wpts[1][WP_Y_IDX]]
        wp_vector = np.array(next_wp_pos) - np.array(closest_wp_pos)

        distance = np.linalg.norm(delta_position)
        projection = np.dot(delta_position, wp_vector) / np.linalg.norm(wp_vector)

        reward += projection * 10

        # Debug prints for diagnosing reward calculation
        if self.print_info:
            print(f"[Reward Debug] delta_position: {delta_position}")
            print(f"[Reward Debug] wp_vector: {wp_vector}")
            print(f"[Reward Debug] projection: {projection}")
            print(f"[Reward Debug] reward after projection: {reward}")

        self.last_position = current_position
        # print(f"distance: {distance}, projection: {projection}")

        # Speed Reward (Encourage Fast Movement)
        speed = car_state[LINEAR_VEL_X_IDX]
        speed_reward = 0.0
        if speed > 2:
            speed_reward = speed * 0.005
        # reward += speed_reward

        # delta Steering reward: Penalize Sudden Steering (Encourage Stability)
        steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        steering_diff_reward = steering_diff * -0.1
        # reward += steering_diff_reward

        # Steering reward: Penalize Large Steering Angle (Encourage Smooth Driving)
        steering_penalty = -abs(car_state[STEERING_ANGLE_IDX]) * 0.02
        # reward += steering_penalty

        # Lidar reward: Penalize being close to walls (Encourage Safe Driving)
        lidar_penalty = -sum(
            min(100, np.cos(driver.LIDAR.processed_angles_rad[i]) * 1 / range)
            for i, range in enumerate(driver.LIDAR.processed_ranges)
            if range < 0.5 and range != 0
        )
        # reward += lidar_penalty * 0.1

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
        # reward += spin_reward

        # Penalize Being Stuck
        stuck_reward = 0.0
        if speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 20:
                stuck_reward = -1
            if self.stuck_counter >= 200:
                stuck_reward = -10
        else:
            self.stuck_counter = 0
        reward += stuck_reward

        # Update State
        self.last_steering = car_state[STEERING_ANGLE_IDX]

        # Debug Info (Optional)
        if self.print_info and reward != 0:
            print(f"Reward: {reward}")

        self.reward = reward

        if Settings.SAVE_REWARDS:
            reward_components = {
                "progress": projection,
                "speed": speed_reward,
                "steering_diff": steering_diff_reward,
                "steering_penalty": steering_penalty,
                "lidar_penalty": lidar_penalty,
                "stuck_reward": stuck_reward,
                "spin_reward": spin_reward,
                "total_reward": reward,
            }
            self.reward_components_history.append(reward_components)

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
