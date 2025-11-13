
import os
import sys

from pyparsing import deque
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import *

class RewardCalculator:
    def __init__(self):
        
        self.print_info = False


        self.action_history_queue = deque(maxlen=10)
        self.reward_components_history = []
        
        self.reset()

    def reset(self):
        
        self.time = 0
        self.last_progress : float = 0
        self.last_progress_time : float = 0
        self.action_history_queue.clear()
        self.spin_counter = 0
        self.stuck_counter = 0
        self.last_wp_index = 0
        self.truncated = False
        self.last_s = None
        self.last_action = None
        self.reward = 0
        self.reward_history = []

    def _calculate_reward(self, driver: 'CarSystem', obs: dict) -> float:
        waypoint_utils: WaypointUtils = driver.waypoint_utils
        car_state = driver.car_state
        reward = 0

        crash = obs.get('collision', False)
        s, d, e, k = waypoint_utils.frenet_coordinates

        # Terminate if track boundary is crossed
        wp_distances_l = driver.waypoint_utils.next_waypoints[0, WP_D_LEFT_IDX]
        wp_distances_r = driver.waypoint_utils.next_waypoints[0, WP_D_RIGHT_IDX]
        leave_bounds = d < -wp_distances_r or d > wp_distances_l

        # Crash penalty 
        crash_penalty = 0
        if leave_bounds or crash:
            crash_penalty = -15
            reward += crash_penalty
            self.truncated = True


        # Progress along the waypoints
        if(self.last_s is None):
            self.last_s = s
        delta_s = s - self.last_s
        self.last_s = s

        progress_reward = delta_s * 1.0
        reward += progress_reward


        # Distance to waypoints
        wp_distance_penalty = - abs(d) * 0.05
        reward += wp_distance_penalty


        # Penalize d_control for smooth control
        action = np.array([driver.angular_control, driver.translational_control])

        if(self.last_action is None):
            self.last_action = action
        

        d_action_penality = 0
        if len(self.action_history_queue) > 0:
            d_action = self.last_action - action
            d_action_penality = - (0.5 * abs(d_action[0]) + 0.05 * abs(d_action[1]))

        reward += d_action_penality
        
        self.action_history_queue.append(action)


        # Speed Reward (Encourage Fast Movement)
        speed = car_state[LINEAR_VEL_X_IDX]

        # Spinning reward Penalize Spinning (Fixing Instability)
        spin_reward = 0.0
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 15.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                spin_reward = -15
                self.truncated = True
        else:
            self.spin_counter = 0
        reward += spin_reward

        # Penalize Being Stuck
        stuck_reward = 0.0
        if speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 100:
                stuck_reward = -15
                self.truncated = True
        else:
            self.stuck_counter = 0
        reward += stuck_reward

        # Update State
        self.last_action = action

        # Debug Info (Optional)
        if self.print_info and reward != 0:
            print(f"Reward: {reward}")

        self.reward = reward

        if Settings.SAVE_REWARDS:
            reward_components = {
                "progress": progress_reward,
                "crash_reward": crash_penalty,
                "wp_distance_penalty": wp_distance_penalty,
                "d_action_penality": d_action_penality,
                "stuck_reward": stuck_reward,
                "spin_reward": spin_reward,
                "total_reward": reward,
            }
            self.reward_components_history.append(reward_components)

        self.reward_history.append(reward)
        self.time += Settings.TIMESTEP_CONTROL
        return reward


    def plot_history(self, save_path: str):
        import matplotlib.pyplot as plt
        import numpy as np

        # Access reward components history
        reward_components_history = self.reward_components_history

        # Extract reward components
        steps = range(len(reward_components_history))
        reward_labels = [
            "progress",
            "crash_reward",
            # "steering_penalty",
            # "acceleration_penalty",
            "wp_distance_penalty",
            "d_action_penality",
            "stuck_reward",
            "spin_reward",
            "total_reward"
        ]
        reward_colors = [
            "blue",
            "gray",
            "red",
            "orange",
            "purple",
            "green",
            "magenta",
            "black"
        ]

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
