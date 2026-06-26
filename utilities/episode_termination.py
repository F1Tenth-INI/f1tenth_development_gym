import math

import numpy as np

from utilities.state_utilities import ANGULAR_VEL_Z_IDX, LINEAR_VEL_X_IDX, LINEAR_VEL_Y_IDX
from utilities.waypoint_utils import WP_D_LEFT_IDX, WP_D_RIGHT_IDX


class EpisodeTerminator:
    """Detect episode truncation/termination from car state and driver observation."""

    SPIN_ANGULAR_VEL_THRESHOLD = 15.0
    SPIN_STEPS_THRESHOLD = 50
    STUCK_SPEED_THRESHOLD = 0.3
    STUCK_STEPS_THRESHOLD = 50

    def __init__(self):
        self.reset()

    def reset(self):
        self.spin_counter = 0
        self.stuck_counter = 0

    def evaluate(self, controller_obs: dict, driver_obs: dict) -> dict:
        car_state = np.asarray(controller_obs["car_state"])
        frenet_coordinates = np.asarray(controller_obs["frenet_coordinates"])
        next_waypoints = np.asarray(controller_obs["next_waypoints"])
        _, d, _, _ = frenet_coordinates

        collision = bool(driver_obs.get("collision", False))
        virtual_opponent_collision = bool(
            controller_obs.get("virtual_opponent_collision", False)
        )
        interrupted = bool(driver_obs.get("interrupted", False))
        terminated = bool(driver_obs.get("terminated", False))

        wp_distances_l = next_waypoints[0, WP_D_LEFT_IDX]
        wp_distances_r = next_waypoints[0, WP_D_RIGHT_IDX]
        leave_track = bool(d < -wp_distances_r or d > wp_distances_l)

        speed = math.sqrt(
            float(car_state[LINEAR_VEL_X_IDX]) ** 2 + float(car_state[LINEAR_VEL_Y_IDX]) ** 2
        )

        spinning = False
        if abs(float(car_state[ANGULAR_VEL_Z_IDX])) > self.SPIN_ANGULAR_VEL_THRESHOLD:
            self.spin_counter += 1
            if self.spin_counter >= self.SPIN_STEPS_THRESHOLD:
                spinning = True
        else:
            self.spin_counter = 0

        is_slow = speed < self.STUCK_SPEED_THRESHOLD
        stuck = False
        if is_slow:
            self.stuck_counter += 1
            if self.stuck_counter >= self.STUCK_STEPS_THRESHOLD:
                stuck = True
        else:
            self.stuck_counter = 0

        truncated = bool(
            leave_track
            or collision
            or virtual_opponent_collision
            or interrupted
            or spinning
            or stuck
        )
        done = bool(
            truncated
            or terminated
            or driver_obs.get("done", False)
        )

        return {
            "truncated": truncated,
            "terminated": terminated,
            "done": done,
            "leave_track": leave_track,
            "collision": collision,
            "virtual_opponent_collision": virtual_opponent_collision,
            "interrupted": interrupted,
            "spinning": bool(spinning),
            "stuck": bool(stuck),
        }
