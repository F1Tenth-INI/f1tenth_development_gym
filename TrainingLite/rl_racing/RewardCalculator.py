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

        self.control_penalty_factor = 0.05
        self.d_control_penalty_factor = 0.5

    def reset(self):
        
        self.time = 0
        self.last_progress : float = 0
        self.last_progress_time : float = 0
        self.action_history_queue.clear()
        self.spin_counter = 0
        self.stuck_counter = 0
        self.last_wp_index = 0
        self.truncated = False

        self.reward = 0
        self.last_position = None
        self.reward_history = []

    def _calculate_reward(self, driver: 'CarSystem', obs: dict) -> float:
        waypoint_utils: WaypointUtils = driver.waypoint_utils
        car_state = driver.car_state
        reward = 0

        # Crash penalty
        crash = obs.get('collision', False)
        crash_reward = -30 if crash else 0
        reward += crash_reward


        # Progress along the waypoints
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
        self.last_position = [car_state[POSE_X_IDX], car_state[POSE_Y_IDX]]
        progress_reward = projection * 2.0
        reward += progress_reward


        # Distance to waypoints
        next_wp_pos_relative = driver.waypoint_utils.next_waypoint_positions_relative[0]
        distance_to_next_wp = np.linalg.norm(next_wp_pos_relative)
        
        if(distance_to_next_wp < 0.15): 
            distance_to_next_wp = 0

        reward -= distance_to_next_wp * 0.1

        # Terminate if too far
        if(distance_to_next_wp > 5): 
            reward = -30
            self.truncated = True

        # Terminate if track boundary is crossed
        s, d, e, k = driver.waypoint_utils.get_frenet_coordinates(driver.car_state)
        wp_distances_l = driver.waypoint_utils.next_waypoints[0, WP_D_LEFT_IDX]
        wp_distances_r = driver.waypoint_utils.next_waypoints[0, WP_D_RIGHT_IDX]
        if d < -wp_distances_r or d > wp_distances_l:
            reward = -30
            self.truncated = True


        # Penalize action  
        steering_reward = - self.control_penalty_factor * np.linalg.norm(1 * driver.angular_control)
        acceleration_reward = - self.control_penalty_factor * np.linalg.norm(0.1 * driver.translational_control)
        reward += steering_reward
        reward += acceleration_reward

        # Penalize d_control for smooth control
        action = np.array([driver.angular_control, driver.translational_control])
        d_control_reward = 0
        if len(self.action_history_queue) > 0:
            last_action = self.action_history_queue[-1] 
            d_control = last_action - action
            d_angular_control = d_control[0]
            d_translational_control = d_control[1]

            d_control_reward = - self.d_control_penalty_factor * (np.linalg.norm(d_angular_control) * 1.0 + np.linalg.norm(d_translational_control) * 0.1)
        reward += d_control_reward
        
        self.action_history_queue.append(action)

        # Debug prints for diagnosing reward calculation
        if self.print_info:
            print(f"[Reward Debug] delta_position: {delta_position}")
            print(f"[Reward Debug] wp_vector: {wp_vector}")
            print(f"[Reward Debug] projection: {projection}")
            print(f"[Reward Debug] distance: {distance}")
            print(f"[Reward Debug] v_x: {car_state[LINEAR_VEL_X_IDX]}")
            print(f"[Reward Debug] reward after projection: {reward}")
            # print(f"[Reward Debug] d_control_reward: {d_control_reward}")
            print(f"[Reward Debug] total_reward: {reward + d_control_reward}")

        self.last_position = current_position
        # print(f"distance: {distance}, projection: {projection}")

        # Speed Reward (Encourage Fast Movement)
        speed = car_state[LINEAR_VEL_X_IDX]

        # Spinning reward Penalize Spinning (Fixing Instability)
        spin_reward = 0.0
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 15.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                spin_reward = -10
                self.truncated = True
        else:
            self.spin_counter = 0
        reward += spin_reward

        # Penalize Being Stuck
        stuck_reward = 0.0
        if speed < 1.0:
            self.stuck_counter += 1
            if self.stuck_counter >= 100:
                stuck_reward = -10
                self.truncated = True
        else:
            self.stuck_counter = 0
        reward += stuck_reward

        # # Penalize beeing too fast
        # suggested_speed = waypoint_utils.next_waypoints[0][WP_VX_IDX]
        # if speed > suggested_speed:
        #     reward -= (speed - suggested_speed) * 0.1

        # Update State
        self.last_steering = car_state[STEERING_ANGLE_IDX]

        # Debug Info (Optional)
        if self.print_info and reward != 0:
            print(f"Reward: {reward}")



        # # Penalize backward driving by hecking the car_state[LINEAR_VEL_X_IDX]
        # backward_driving_reward = 0.0
        # if car_state[LINEAR_VEL_X_IDX] < 0:
        #     backward_driving_reward = car_state[LINEAR_VEL_X_IDX] * 2.0
        # reward += backward_driving_reward


        self.reward = reward

        if Settings.SAVE_REWARDS:
            reward_components = {
                "progress": progress_reward,
                "crash_reward": crash_reward,
                "steering_penalty": steering_reward,
                "acceleration_penalty": acceleration_reward,
                "d_control_reward": d_control_reward,
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
            "steering_penalty",
            "acceleration_penalty",
            "d_control_reward",
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
