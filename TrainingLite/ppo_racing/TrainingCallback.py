from stable_baselines3.common.callbacks import BaseCallback
import time
import numpy as np

class TrainingStatusCallback(BaseCallback):
    def __init__(self, check_freq=5000, save_path='./models', verbose=1):
        super(TrainingStatusCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0

            mean_reward = np.mean(self.locals["rewards"]) if "rewards" in self.locals else float('nan')

            print(f"🔄 Iteration {self.n_calls}, Timesteps: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, FPS: {fps:.2f}")

            # Save the model periodically
            # model_filename = f"{self.save_path}_{self.num_timesteps}.zip"
            model_filename = f"{self.save_path}_running.zip"
            self.model.save(model_filename)
            if self.verbose > 0:
                print(f"💾 Model saved to {model_filename}")

        return True
