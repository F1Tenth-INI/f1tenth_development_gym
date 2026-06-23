
import os
import sys
import math

from collections import deque
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

import numpy as np
from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import *

class RewardCalculator:
    # Cap history to avoid unbounded growth → GC pauses and FPS drops after 100k+ steps
    REWARD_HISTORY_CAP = 10_000

    def __init__(self):
        
        self.print_info = False


        self.reward_components_history = deque(maxlen=self.REWARD_HISTORY_CAP)


        # Weights
        self.w_crash = 10
        self.w_progress = 1.0   # per meter of along-track progress
        self.w_lateral_error = 0.05   # per meter cross-track error penalty
        self.w_d_steering = 1.5
        self.w_d_acceleration = 0.1
        self.w_speed_cap = 0.0 # 0.3

        if Settings.RANDOM_WAYPOINT_VEL_FACTOR:
            self.w_speed_cap = 0.3
      
        
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
        self.last_reward_components = {}

    def _calculate_reward(self, controller_obs: dict) -> dict:
        car_state = np.asarray(controller_obs["car_state"])
        next_waypoints = np.asarray(controller_obs["next_waypoints"])
        frenet_coordinates = np.asarray(controller_obs["frenet_coordinates"])
        reward = 0

        crash = controller_obs.get('collision', False)
        interruption = controller_obs.get('interrupted', False)

        speed = math.sqrt(car_state[LINEAR_VEL_X_IDX]**2 + car_state[LINEAR_VEL_Y_IDX]**2)
        s, d, e, k = frenet_coordinates

        # Crash / Leave virtual track penalty 
        crash_penalty = 0

        wp_distances_l = next_waypoints[0, WP_D_LEFT_IDX]
        wp_distances_r = next_waypoints[0, WP_D_RIGHT_IDX]
        leave_bounds = d < -wp_distances_r or d > wp_distances_l
        if interruption:
            crash_penalty = -self.w_crash
            crash_penalty -= 1.5 * speed
            reward += crash_penalty
            self.truncated = True
        elif (leave_bounds or crash) and not self.truncated:
            crash_penalty = -self.w_crash
            crash_penalty -= 1.5 * speed
            reward += crash_penalty
            if Settings.TRUNCATE_ON_LEAVE_TRACK:
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

        control_history = np.asarray(controller_obs.get("control_history", []))
        if len(control_history) > 0:
            action = np.asarray(control_history[-1], dtype=np.float64)
        else:
            action = np.zeros(2, dtype=np.float64)
        if(self.last_action is None):
            self.last_action = action
        d_action = self.last_action - action

        d_action_penality = - (self.w_d_steering * abs(d_action[0]) + self.w_d_acceleration * abs(d_action[1]))
        reward += d_action_penality
        
        # Speed cap penalty
        speed_cap_penalty = 0.0
        
        suggested_speed = next_waypoints[0, WP_VX_IDX]
        if(speed > suggested_speed):
            speed_cap_penalty = - self.w_speed_cap * (speed - suggested_speed) ** 2
        reward += speed_cap_penalty
        
        

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
        stuck_reward = 0.0
        if speed < 1.0:
            self.stuck_counter += 1
            stuck_reward = -0.05
            if self.stuck_counter >= 50:
                stuck_reward = -self.w_crash 
                self.truncated = True
        else:
            self.stuck_counter = 0
        reward += stuck_reward

        # Update State
        self.last_s = s
        self.last_action = action
      
        self.reward = reward

        components = {
            "progress": float(progress_reward),
            "crash_reward": float(crash_penalty),
            "wp_distance_penalty": float(wp_distance_penalty),
            "d_action_penality": float(d_action_penality),
            "speed_cap_penalty": float(speed_cap_penalty),
            "stuck_reward": float(stuck_reward),
            "spin_reward": float(spin_reward),
        }
        self.last_reward_components = components

        if Settings.SAVE_REWARDS:
            self.reward_components_history.append({
                **components,
                "total_reward": float(reward),
                "difficulty": self.difficulty,
            })

        self.reward_history.append(reward)
        self.accumulated_reward += reward
        self.simulation_step += 1
        
        self.adjust_difficulty()

        return {"total_reward": float(reward), "components": components}
    
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
        output_path = os.path.join(save_path, "cumulative_reward_components.png")
        plt.savefig(output_path)
        plt.close(fig)
