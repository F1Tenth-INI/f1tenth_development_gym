import numpy as np

class SamplingTracker:
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.steps_taken = np.zeros(max_steps, dtype=np.int32)