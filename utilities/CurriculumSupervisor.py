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

        self.track_width_factor = 1.0

        self.speed_adjust_mode = Settings.SAC_CURRICULUM_SPEED_ADJUST_MODE # 'vel_factor' or 'speed_cap'


    def update_progress(self, progress):
        self.progress = progress

    def get_progress(self):
        return self.progress
    

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
    
    def update_difficulty_dynamic(self):
        # Placeholder for dynamic difficulty adjustment based on success rate
        if self.difficulty >= self.max_difficulty:
            print("Already at max difficulty, not sure what to do here yet")
            return

        if self.success_rate > 0.5:
            print("i would adjust difficulty up")
            self.difficulty += self.dynamic_difficulty_step
            self.difficulty = min(self.difficulty, self.max_difficulty)
            self.clear_completed_episodes()

        if self.debug:
            print(f"[Curriculum Debug] Success rate:: {self.success_rate:.3f} | Difficulty: {self.difficulty:.3f}")
        return self.difficulty


    def get_difficulty(self):
        return self.difficulty
    
    def adjust_speed(self, speed_max):
        
        if self.speed_adjust_mode == 'vel_factor':
            velocity_factor = (self.difficulty) * (speed_max)
            Settings.GLOBAL_WAYPOINT_VEL_FACTOR = velocity_factor
            if self.debug:
                print(f"[Curriculum Debug] New Velocity Factor: {velocity_factor:.3f}")

        if self.speed_adjust_mode == 'accel_cap':
            Settings.SAC_ACCEL_CAP = self.difficulty * Settings.SAC_ACCEL_CAP_MAX
            if self.debug:
                print(f"[Curriculum Debug] New acceleration Cap: {Settings.SAC_ACCEL_CAP:.3f}")

        if self.speed_adjust_mode == 'speed_cap':
            Settings.SAC_CURRICULUM_SPEED_LIMIT = self.difficulty * Settings.SAC_CURRICULUM_SPEED_LIMIT_MAX
            if self.debug:
                print(f"[Curriculum Debug] New Speed Cap: {Settings.SAC_CURRICULUM_SPEED_LIMIT:.3f}")

        return
    
    def adjust_track_width(self, base_width):
        width_factor = max(base_width, 3.0 - 2*self.difficulty)
        Settings.SAC_CURRICULUM_TRACK_WIDTH_FACTOR = width_factor
        if self.debug:
            print(f"[Curriculum Debug] New track width factor: {Settings.SAC_CURRICULUM_TRACK_WIDTH_FACTOR:.3f}")
        return
    
    def adjust_noise(self, base_noise):
        Settings.NOISE_LEVEL_CAR_STATE = [x * self.difficulty for x in Settings.SAC_NOISE_LEVEL_CAR_STATE_MAX]
        Settings.NOISE_LEVEL_CONTROL = [x * self.difficulty for x in Settings.SAC_NOISE_LEVEL_CONTROL_MAX]
        if self.debug:
            print(f"[Curriculum Debug] New car state noise levels: [{', '.join(f'{x:.3f}' for x in Settings.NOISE_LEVEL_CAR_STATE)}]")
            print(f"[Curriculum Debug] New control noise levels: [{', '.join(f'{x:.3f}' for x in Settings.NOISE_LEVEL_CONTROL)}]")
        return
    
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