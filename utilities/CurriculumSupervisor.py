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

    def update_progress(self, progress):
        self.progress = progress

    def get_progress(self):
        return self.progress
    
    def update_difficulty(self):
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
        velocity_factor = (self.difficulty) * (speed_max)
        Settings.GLOBAL_WAYPOINT_VEL_FACTOR = velocity_factor
        if self.debug:
            print(f"[Curriculum Debug] New Velocity Factor: {velocity_factor:.3f}")
        return velocity_factor

