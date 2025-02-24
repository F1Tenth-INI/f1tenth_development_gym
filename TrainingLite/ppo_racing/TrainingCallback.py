from stable_baselines3.common.callbacks import BaseCallback
import time
import numpy as np

class TrainingStatusCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super(TrainingStatusCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        """Called at each step"""
        if self.n_calls % self.check_freq == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            mean_reward = np.mean(self.locals["rewards"]) if "rewards" in self.locals else float('nan')

            print(f"🔄 Iteration {self.n_calls}, Timesteps: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, FPS: {fps:.2f}")

        return True  # Continue training