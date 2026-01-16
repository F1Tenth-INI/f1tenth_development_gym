import numpy as np
from utilities.Settings import Settings

class CurriculumSupervisor:
    def __init__(self, initial_difficulty: float = 5.0, max_difficulty: float = 1.0, progress = 0.0, sac_curriculum_t1 = 0.05, sac_curriculum_t2 = 0.8, debug = False):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty = initial_difficulty
        self.progress = progress

        self.sac_curriculum_t1 = sac_curriculum_t1
        self.sac_curriculum_t2 = sac_curriculum_t2

        self.debug = debug

        self.completed_episodes_length = 5
        self.completed_episodes = np.zeros(self.completed_episodes_length) 
        self.completed_episodes_counter = 0
        self.success_rate = 0.0

        self.dynamic_difficulty_step = 0.05

        self.speed_adjust_mode = 'vel_factor' # 'vel_factor' or 'speed_cap'


    def update_progress(self, progress):
        self.progress = progress

    def get_progress(self):
        return self.progress
    
    #TODO: maybe update difficulty should be one function, with param for linear vs dynamic
    def update_difficulty_linear(self):
        if self.progress <= self.sac_curriculum_t1:
                self.difficulty = self.initial_difficulty #starting difficulty
        elif self.progress >= self.sac_curriculum_t2:
            self.difficulty = 1.0
        else:
            self.difficulty = (self.initial_difficulty + (1.0 - self.initial_difficulty) 
            * ((self.progress - self.sac_curriculum_t1) / (self.sac_curriculum_t2 - self.sac_curriculum_t1)))

        if self.debug:
            print(f"[Curriculum Debug] Progress: {self.progress:.3f} | Difficulty: {self.difficulty:.3f}")
            
        return self.difficulty

    def get_difficulty(self):
        return self.difficulty
    
    def adjust_speed(self, speed_max):
        if self.speed_adjust_mode == 'vel_factor':
            velocity_factor = (self.difficulty) * (speed_max)
            Settings.GLOBAL_WAYPOINT_VEL_FACTOR = velocity_factor
            if self.debug:
                print(f"[Curriculum Debug] New Velocity Factor: {velocity_factor:.3f}")
        if self.speed_adjust_mode == 'speed_cap':
            Settings.SAC_SPEED_CAP = self.difficulty * Settings.SAC_SPEED_CAP_MAX
            # if speed_cap > speed_max:
            #     speed_cap = speed_max
            # Settings.GLOBAL_WAYPOINT_SPEED_CAP = speed_cap
            if self.debug:
                print(f"[Curriculum Debug] New Speed Cap: {Settings.SAC_SPEED_CAP:.3f}")
        return velocity_factor
    
    def update_completed_episodes(self, success):
        self.completed_episodes[self.completed_episodes_counter] = success
        self.completed_episodes_counter += 1
        if self.completed_episodes_counter >= self.completed_episodes_length:
            self.completed_episodes_counter = 0
        pass

    def clear_completed_episodes(self):
        self.completed_episodes = np.zeros(self.completed_episodes_length)
        self.completed_episodes_counter = 0
        return

    def calculate_success_rate(self):
        # Placeholder for success rate calculation
        self.success_rate = np.mean(self.completed_episodes)
        print("Current success rate:", self.success_rate)
        return self.success_rate

    def update_difficulty_dynamic(self):
        # Placeholder for dynamic difficulty adjustment based on success rate
        if self.difficulty >= self.max_difficulty:
            print("Already at max difficulty, not sure what to do here yet")
            return

        if self.success_rate > 0.5:
            print("i would adjust difficulty up")
            self.difficulty += self.dynamic_difficulty_step
            self.clear_completed_episodes()

        if self.debug:
            print(f"[Curriculum Debug] Success rate:: {self.success_rate:.3f}) | Difficulty: {self.difficulty:.3f}")
        
        return self.difficulty