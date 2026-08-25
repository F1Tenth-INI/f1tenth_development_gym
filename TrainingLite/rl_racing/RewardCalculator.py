
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
    PROXIMITY_THRESHOLD_M = 0.5
    STUCK_MIN_SPEED = 0.3
    FAST_LAP_TIME_THRESHOLD_S = 25.0
    LAP_FINISHED_REWARD = 100.0
    FAST_LAP_REWARD = 200.0

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
        self.w_proximity = 0.0
        self.w_slip = 0.0  # f8e3040 baseline had no slip term; set >0 for low-slip driving
        # CBF safety-filter correction penalties (requires Settings.CBF_SAFETY_FILTER).
        self.w_cbf_steering = 1.5
        self.w_cbf_acceleration = 0.1
        self.w_cbf_slack = 1.0
        self.w_cbf_intervention = 0.0  # optional; steering/accel terms already cover adjustments
        # MPC predictive SF corrections (Tearle et al.; Settings.MPC_SAFETY_FILTER).
        self.w_mpc_steering = 1.5
        self.w_mpc_acceleration = 0.1
        self.w_mpc_intervention = 0.0  # optional; steering/accel terms already cover adjustments


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
        self.last_wp_index = 0
        self.last_s = None
        self.last_action = None
        self.reward = 0
        self.reward_history = []
        self.accumulated_reward = 0
        self.last_reward_components = {}

    @staticmethod
    def _cbf_metrics(cbf_info: dict | None) -> dict[str, float]:
        """Extract CBF filter adjustment metrics from CarSystem ``cbf_info``."""
        if not cbf_info:
            return {
                "cbf_active": 0.0,
                "cbf_intervention": 0.0,
                "cbf_slack": 0.0,
                "cbf_delta_correction": 0.0,
                "cbf_accel_correction": 0.0,
            }

        delta_nom = float(cbf_info.get("delta_nom", 0.0))
        delta_safe = float(cbf_info.get("delta_safe", delta_nom))
        accel_nom = float(cbf_info.get("accel_nom", 0.0))
        accel_safe = float(cbf_info.get("accel_safe", accel_nom))

        return {
            "cbf_active": float(bool(cbf_info.get("active", False))),
            "cbf_intervention": float(cbf_info.get("intervention", 0.0)),
            "cbf_slack": float(cbf_info.get("slack", 0.0)),
            "cbf_delta_correction": delta_safe - delta_nom,
            "cbf_accel_correction": accel_safe - accel_nom,
        }

    @staticmethod
    def _mpc_metrics(mpc_info: dict | None) -> dict[str, float]:
        """Extract MPC safety-filter adjustment metrics from CarSystem ``mpc_info``."""
        if not mpc_info:
            return {
                "mpc_active": 0.0,
                "mpc_intervene": 0.0,
                "mpc_intervention": 0.0,
                "mpc_delta_correction": 0.0,
                "mpc_accel_correction": 0.0,
            }

        delta_nom = float(mpc_info.get("delta_nom", 0.0))
        delta_safe = float(mpc_info.get("delta_safe", delta_nom))
        accel_nom = float(mpc_info.get("accel_nom", 0.0))
        accel_safe = float(mpc_info.get("accel_safe", accel_nom))

        return {
            "mpc_active": float(bool(mpc_info.get("active", False))),
            "mpc_intervene": float(bool(mpc_info.get("intervene", False))),
            "mpc_intervention": float(mpc_info.get("intervention", 0.0)),
            "mpc_delta_correction": delta_safe - delta_nom,
            "mpc_accel_correction": accel_safe - accel_nom,
        }

    def _calculate_reward(self, controller_obs: dict) -> dict:
        car_state = np.asarray(controller_obs["car_state"])
        next_waypoints = np.asarray(controller_obs["next_waypoints"])
        frenet_coordinates = np.asarray(controller_obs["frenet_coordinates"])
        reward = 0

        termination = controller_obs.get("episode_termination", {})
        leave_track = bool(termination.get("leave_track", False))
        collision = bool(termination.get("collision", False))
        virtual_opponent_collision = bool(
            termination.get("virtual_opponent_collision", False)
        )
        interrupted = bool(termination.get("interrupted", False))
        spinning = bool(termination.get("spinning", False))
        stuck = bool(termination.get("stuck", False))

        speed = math.sqrt(car_state[LINEAR_VEL_X_IDX]**2 + car_state[LINEAR_VEL_Y_IDX]**2)
        s, d, e, k = frenet_coordinates

        # Crash / leave track / interruption penalties (termination decided in CarSystem).
        crash_penalty = 0
        if leave_track or collision or virtual_opponent_collision or interrupted:
            crash_penalty = -self.w_crash
            crash_penalty -= 1.5 * speed
            reward += crash_penalty


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
        if self.last_action is None:
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

        # Quadratic proximity penalty from min lidar distance and virtual opponents.
        proximity_penalty = 0.0
        min_dist = float("inf")
        processed_ranges = controller_obs.get("processed_ranges")
        if processed_ranges is not None and len(processed_ranges) > 0:
            min_dist = float(np.min(processed_ranges))
        vo_dist = controller_obs.get("min_virtual_opponent_distance")
        if vo_dist is not None and np.isfinite(vo_dist):
            min_dist = min(min_dist, float(vo_dist))
        if np.isfinite(min_dist) and min_dist < self.PROXIMITY_THRESHOLD_M:
            proximity_value = (
                (self.PROXIMITY_THRESHOLD_M - min_dist) / self.PROXIMITY_THRESHOLD_M
            ) ** 2
            proximity_penalty = -self.w_proximity * proximity_value
        reward += proximity_penalty
        # Penalize lateral slip (body-frame y velocity).
        slip_penalty = -self.w_slip * abs(car_state[LINEAR_VEL_Y_IDX])
        reward += slip_penalty

        cbf_metrics = self._cbf_metrics(controller_obs.get("cbf_info"))
        cbf_penalty = -(
            self.w_cbf_steering * abs(cbf_metrics["cbf_delta_correction"])
            + self.w_cbf_acceleration * abs(cbf_metrics["cbf_accel_correction"])
            + self.w_cbf_intervention * cbf_metrics["cbf_intervention"]
            + self.w_cbf_slack * cbf_metrics["cbf_slack"]
        )
        reward += cbf_penalty

        mpc_metrics = self._mpc_metrics(controller_obs.get("mpc_info"))
        mpc_penalty = -(
            self.w_mpc_steering * abs(mpc_metrics["mpc_delta_correction"])
            + self.w_mpc_acceleration * abs(mpc_metrics["mpc_accel_correction"])
            + self.w_mpc_intervention * mpc_metrics["mpc_intervention"]
        )
        reward += mpc_penalty

        # Spin / stuck penalties when EpisodeTerminator flags termination this step.
        spin_reward = 0.0
        if spinning:
            spin_reward = -self.w_crash
        reward += spin_reward

        stuck_reward = 0.0
        if speed < self.STUCK_MIN_SPEED:
            stuck_reward = -0.05
        if stuck:
            stuck_reward = -self.w_crash
        reward += stuck_reward

        lap_finished_reward = 0.0
        fast_lap_reward = 0.0
        # if bool(controller_obs.get("lap_finished")):
        #     lap_finished_reward = self.LAP_FINISHED_REWARD
        #     reward += lap_finished_reward

        #     laptime = float(controller_obs.get("lap_time"))
        #     fast_lap_reward = 10 * (self.FAST_LAP_TIME_THRESHOLD_S - laptime) if laptime < self.FAST_LAP_TIME_THRESHOLD_S else 0.0
        #     reward += fast_lap_reward

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
            "proximity_penalty": float(proximity_penalty),
            "slip_penalty": float(slip_penalty),
            "cbf_penalty": float(cbf_penalty),
            "mpc_penalty": float(mpc_penalty),
            # **cbf_metrics,
            # **mpc_metrics,
            "stuck_reward": float(stuck_reward),
            "spin_reward": float(spin_reward),
            "lap_finished_reward": float(lap_finished_reward),
            "fast_lap_reward": float(fast_lap_reward),
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
        component_keys = sorted({
            key
            for comp in reward_components_history
            for key in comp
            if key not in ("total_reward", "difficulty")
        })
        reward_labels = [
            *component_keys,
            "difficulty",
            "total_reward",
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
