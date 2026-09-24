from TrainingLite.rl_racing.RewardCalculator import RewardCalculator

import os
import sys
import math
import numpy as np
from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import *

class RewardCalculatorMidlineReturner(RewardCalculator):
    def __init__(self):
        super().__init__()

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




        # Lateral error do raceline
        wp_distance_penalty = 0.0
        wp_distance_penalty = -self.w_lateral_error * (abs(d) + abs(e))
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
        
        #speed penalty, only penalize if car is on the midline & has correct heading
        reward -= speed**2 * 0.05 #* np.exp(-(e**2 + d**2))

        
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
            # pass
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
            "crash_reward": float(crash_penalty),
            "wp_distance_penalty": float(wp_distance_penalty),
            "d_action_penality": float(d_action_penality),
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