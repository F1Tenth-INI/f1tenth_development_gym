from stable_baselines3.common.callbacks import BaseCallback
import time
import numpy as np
import importlib
import subprocess
import os



class TrainingStatusCallback(BaseCallback):
    def __init__(self, check_freq=5000, save_path='./models', verbose=1):
        super(TrainingStatusCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.start_time = time.time()
        self.save_freq = 12500
        
    # Add another callback: Save environment:
    # Reset, do a rollout, reset again

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0

            mean_reward = np.mean(self.locals["rewards"]) if "rewards" in self.locals else float('nan')

            print(f"🔄 Iteration {self.n_calls}, Timesteps: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, FPS: {fps:.2f}")
            
        if self.n_calls % self.save_freq == 0:
            # Save the model periodically
            model_filename = f"{self.save_path}_{self.num_timesteps}.zip"
            self.model.save(model_filename)
            model_filename = f"{self.save_path}_running.zip"
            self.model.save(model_filename)
            
            if self.verbose > 0:
                print(f"💾 Model saved to {model_filename}")

            subprocess.Popen(["python", f"TrainingLite/rl_racing/evaluate_model.py", str(self.n_calls)])
            subprocess.Popen(["python", f"TrainingLite/rl_racing/plot_rewards.py", str(self.n_calls)])

        return True
    

class AdjustCheckpointsCallback(BaseCallback):
    def __init__(self, check_freq=1_000_000, model_name = "SAC", log_dir="tensorboard_logs", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq  # Check every 1M steps
    
    def _on_step(self) -> bool:
        # Every 1M steps, decrease checkpoints_per_lap and log it
        if self.n_calls % self.check_freq == 0:
            print(f"🔄 Adjusting checkpoints_per_lap at step {self.num_timesteps}")
            envs = self.training_env.envs if hasattr(self.training_env, "envs") else [self.training_env]
            for env in envs:
                env = env.env
                if hasattr(env, "checkpoints_per_lap"):
                    env.checkpoints_per_lap = max(5, env.checkpoints_per_lap - 10)  # Reduce by 5, min value 5
                    env.simulation.drivers[0].reward_calculator.checkpoint_fraction = 1 / env.checkpoints_per_lap

                    if self.verbose > 0:
                        print(f"🔄 Adjusted checkpoints_per_lap to {env.checkpoints_per_lap} at step {self.num_timesteps}")

        return True  # Continue training
    
    def _on_training_end(self):
        self.writer.close()  # ✅ Close TensorBoard writer properly
