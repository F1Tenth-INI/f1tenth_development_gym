
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


        self.reward_components_history = []


        # Weights
        self.w_crash = 15
        self.w_progress = 1.0   # per meter of along-track progress
        self.w_lateral_error = 0.05   # per meter cross-track error penalty
        self.w_d_steering = 0.0 # 2.5
        self.w_d_acceleration = 0.05
        
      
        
        self.increase_difficulty = False
        
        # Weight ranges for Curriculum Learning
        self.crash_penalty_range = [15, 15]
        self.w_d_steering_range = [0.0, 3.5]
        self.w_d_acceleration_range = [0.0, 0.2]
        
        
        self.simulation_step = 0
        self.difficulty = 0

        self.reset()

    def reset(self):
        
        self.time = 0
        self.last_progress : float = 0
        self.last_progress_time : float = 0
        self.spin_counter = 0
        self.stuck_counter = 0
        self.last_wp_index = 0
        self.truncated = False
        self.last_s = None
        self.last_action = None
        self.reward = 0
        self.reward_history = []
        self.accumulated_reward = 0

    def _calculate_reward(self, driver: 'CarSystem', obs: dict) -> float:
        waypoint_utils: WaypointUtils = driver.waypoint_utils
        car_state = driver.car_state
        reward = 0

        # Get variables
        crash = obs.get('collision', False)
        s, d, e, k = waypoint_utils.frenet_coordinates


        # Crash / Leave track penalty 
        crash_penalty = 0

        wp_distances_l = driver.waypoint_utils.next_waypoints[0, WP_D_LEFT_IDX]
        wp_distances_r = driver.waypoint_utils.next_waypoints[0, WP_D_RIGHT_IDX]
        leave_bounds = d < -wp_distances_r or d > wp_distances_l
        if leave_bounds or crash:
            crash_penalty = -self.w_crash
            reward += crash_penalty
            self.truncated = True


        # Progress along the raceline ( Frenet s coordinate ) [meters]
        progress_reward = 0.0

        if(self.last_s is None):
            self.last_s = s
        delta_s = s - self.last_s
        progress_reward = delta_s * self.w_progress
        reward += progress_reward


        # Lateral error do raceline
        wp_distance_penalty = 0.0
        wp_distance_penalty = -self.w_lateral_error * abs(d)
        reward += wp_distance_penalty


        # Penalize d_control for smooth control
        d_action_penality = 0.0

        action = np.array([driver.angular_control, driver.translational_control])
        if(self.last_action is None):
            self.last_action = action
        d_action = self.last_action - action

        d_action_penality = - (self.w_d_steering * abs(d_action[0]) + self.w_d_acceleration * abs(d_action[1]))
        reward += d_action_penality
        

        # Spinning reward Penalize Spinning (Fixing Instability)
        spin_reward = 0.0
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 15.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                spin_reward = -self.w_crash 
                self.truncated = True
        else:
            self.spin_counter = 0
        reward += spin_reward

        # Penalize Being Stuck
        speed = car_state[LINEAR_VEL_X_IDX]
        stuck_reward = 0.0
        if speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 100:
                stuck_reward = -self.w_crash 
                self.truncated = True
        else:
            self.stuck_counter = 0
        reward += stuck_reward

        # Update State
        self.last_s = s
        self.last_action = action
      
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
                "difficulty": self.difficulty
            }
            self.reward_components_history.append(reward_components)

        self.reward_history.append(reward)
        self.accumulated_reward += reward
        self.simulation_step += 1
        
        self.adjust_difficulty()

        return reward
    
    def adjust_difficulty(self):
        
        if not self.increase_difficulty:
            return
        
        progress = self.simulation_step / Settings.SIMULATION_LENGTH
        
        if progress <= 0.3:
            self.difficulty = 0.0
        elif progress >= 0.8:
            self.difficulty = 1.0
        else:
            self.difficulty = (progress - 0.3) / (0.8 - 0.3)

        self.w_crash = self.crash_penalty_range[0] + self.difficulty * (self.crash_penalty_range[1] - self.crash_penalty_range[0])
        self.w_d_steering = self.w_d_steering_range[0] + self.difficulty * (self.w_d_steering_range[1] - self.w_d_steering_range[0])
        self.w_d_acceleration = self.w_d_acceleration_range[0] + self.difficulty * (self.w_d_acceleration_range[1] - self.w_d_acceleration_range[0])
        
    


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
            "difficulty",
            "total_reward"
        ]
        reward_colors = ["blue"] * len(reward_labels)

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
