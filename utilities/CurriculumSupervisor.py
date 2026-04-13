import numpy as np
from collections import deque
from utilities.Settings import Settings


class CurriculumSupervisor:
    def __init__(self, initial_difficulty: float = 0.0, max_difficulty: float = 1.0, progress: float = 0.0):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty = initial_difficulty
        self.progress = progress
        self.sac_curriculum_t1 = Settings.SAC_CURRICULUM_T1
        self.sac_curriculum_t2 = Settings.SAC_CURRICULUM_T2
        self.debug = Settings.SAC_CURRICULUM_DEBUG

        window = getattr(Settings, "SAC_CURRICULUM_REWARD_WINDOW", 10)
        self._rewards_buffer = deque(maxlen=window)
        self._lengths_buffer = deque(maxlen=window)

    def on_episode_end(self, episode_reward: float, episode_length: int) -> None:
        """Update curriculum from episode outcome. Call this from the planner on every episode end."""
        self._rewards_buffer.append(episode_reward)
        self._lengths_buffer.append(episode_length)

        max_len = getattr(Settings, "MAX_EPISODE_LENGTH", 2000)
        boost = getattr(Settings, "SAC_CURRICULUM_FAST_TRACK_BOOST", 0.05)
        req_window = getattr(Settings, "SAC_CURRICULUM_REWARD_WINDOW", 10)

        # Adaptive update
        if len(self._lengths_buffer) >= req_window:
            avg_length = np.mean(self._lengths_buffer)
            if avg_length > max_len * 0.5:
                self._lengths_buffer.clear()
                print(f"Boosting difficulty from {self.difficulty:.3f} to {min(1.0, self.difficulty + boost):.3f}")
                self.difficulty = min(1.0, self.difficulty + boost)
                if self.debug:
                    print(f"[Curriculum Debug] Adaptive boost (length): avg={avg_length:.1f} > {max_len}")
        self.adjust_GLOBAL_SPEED_LIMIT()
        

        # self.update_difficulty()

    def update_progress(self, progress: float) -> None:
        self.progress = progress

    def get_progress(self) -> float:
        return self.progress
    
    def get_difficulty(self) -> float:
        return float(np.clip(self.difficulty, 0.0, 1.0))

    # def update_difficulty(self) -> None:
    #     # return 1.0
    #     """Update difficulty from progress and apply all dependent adjustments (translational clip, etc.)."""
    #     if self.progress <= self.sac_curriculum_t1:
    #             self.difficulty = self.initial_difficulty #starting difficulty
    #     elif self.progress >= self.sac_curriculum_t2:
    #         self.difficulty = 1.0
    #     else:
    #         self.difficulty = (self.initial_difficulty + (1.0 - self.initial_difficulty) 
    #         * ((self.progress - self.sac_curriculum_t1) / (self.sac_curriculum_t2 - self.sac_curriculum_t1)))

    #     if self.debug:
    #         print(f"[Curriculum Debug] Progress: {self.progress:.3f} | Difficulty: {self.difficulty:.3f}")

    #     # self.adjust_translational_clip()
    #     self.adjust_speed_limit()

    
    # def adjust_speed(self, speed_max):
    #     return 1.0
    #     velocity_factor = (self.difficulty) * (speed_max)
    #     Settings.GLOBAL_WAYPOINT_VEL_FACTOR = velocity_factor
    #     if self.debug:
    #         print(f"[Curriculum Debug] New Velocity Factor: {velocity_factor:.3f}")
    #     return velocity_factor

    # def adjust_speed_limit(self):
    #     """Set car model v_max and GLOBAL_SPEED_LIMIT based on difficulty: min at low difficulty -> max at high."""
    #     if not getattr(Settings, 'SAC_CURRICULUM_V_MAX_ENABLED', False):
    #         Settings.SAC_CURRICULUM_V_MAX = None
    #     else:
    #         v_min = Settings.SAC_CURRICULUM_V_MAX_MIN
    #         v_max = Settings.SAC_CURRICULUM_V_MAX_MAX
    #         denom = 1.0 - self.initial_difficulty
    #         frac = (self.difficulty - self.initial_difficulty) / denom if denom > 0 else 1.0
    #         Settings.SAC_CURRICULUM_V_MAX = v_min + (v_max - v_min) * frac
    #         if self.debug:
    #             print(f"[Curriculum Debug] Speed limit v_max: {Settings.SAC_CURRICULUM_V_MAX:.2f} m/s")

    def adjust_GLOBAL_SPEED_LIMIT(self):
        if Settings.GLOBAL_SPEED_LIMIT_CURRICULUM_ENABLED:
            cap_min = Settings.GLOBAL_SPEED_LIMIT_MIN
            cap_max = Settings.GLOBAL_SPEED_LIMIT_MAX
            new_speed_cap = cap_min + (cap_max - cap_min) * self.difficulty
            print(f"Adjusting GLOBAL_SPEED_LIMIT from {Settings.GLOBAL_SPEED_LIMIT:.2f} m/s to {new_speed_cap:.2f} m/s")
            if new_speed_cap != Settings.GLOBAL_SPEED_LIMIT:
                Settings.GLOBAL_SPEED_LIMIT = new_speed_cap
            if self.debug:
                print(f"[Curriculum Debug] GLOBAL_SPEED_LIMIT: {Settings.GLOBAL_SPEED_LIMIT:.2f} m/s")

