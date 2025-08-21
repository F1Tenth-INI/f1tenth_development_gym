from stable_baselines3.common.callbacks import BaseCallback
import time
import numpy as np
import importlib
import subprocess
import os
import csv

from .plot_training import plot_training_csv



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
            # model_filename = f"{self.save_path}_{self.num_timesteps}.zip"
            # self.model.save(model_filename)
            model_filename = f"{self.save_path}_running.zip"
            self.model.save(model_filename)
            
            if self.verbose > 0:
                print(f"💾 Model saved to {model_filename}")

            subprocess.Popen(["python", f"TrainingLite/rl_racing/evaluate_model.py", str(self.n_calls)])
            subprocess.Popen(["python", f"TrainingLite/rl_racing/plot_rewards.py", str(self.n_calls)])

        return True
    

class AdjustCheckpointsCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.total_episodes = 0
        self.crash_count = 0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i, done in enumerate(dones):
            if done:
                self.total_episodes += 1
                if infos[i].get("crashed", False):
                    self.crash_count += 1

        if self.num_timesteps % self.check_freq == 0 and self.total_episodes > 0:
            crash_percentage = (self.crash_count / self.total_episodes) * 100

            print(f"\n🧠 Checkpoint Adjustment Check at {self.num_timesteps} steps:")
            print(f"Crash rate: {crash_percentage:.2f}% ({self.crash_count}/{self.total_episodes})")

            if crash_percentage < 20:
                self.training_env.call("set_adjust_checkpoint_flag", True)
            else:
                print("⚠️ Crash rate too high, no adjustment.")

            # Reset stats for next interval
            self.total_episodes = 0
            self.crash_count = 0

        return True



class EpisodeCSVLogger(BaseCallback):
    def __init__(self, model_name, csv_path: str, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self.csv_path = csv_path
        self.csv_file = None
        self.writer = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.counter = 0

        # Add any custom fields you want here
        self.fields = [
            'episode', 'reward', 'length', 'lap_time', 'crashed',
            'laptimes', 'laptime_min', 'laptime_max', 'laptime_mean',
            'step', 'global_step', 'checkpoints_per_lap',
            'exploration_rate', 'policy_loss', 'value_loss', 'entropy',
            'learning_rate', 'success', 'buffer_size', 'gradient_norm',
            'episode_time', 'action_mean', 'action_std'
        ]
    def _on_training_start(self):
        # Check if the file already exists and create a new one with an incremented suffix
        base_path, ext = os.path.splitext(self.csv_path)
        counter = 1
        while os.path.exists(self.csv_path):
            self.csv_path = f"{base_path}_{counter}{ext}"
            self.counter = counter

            counter += 1

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        # Open the new CSV file and write the header
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fields)
        self.writer.writeheader()

    def _on_step(self):
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        for i, done in enumerate(dones):
            if done:
                info = infos[i]

                # Collect data
                log_data = {
                    'episode': self.num_timesteps,
                    'reward': info.get('episode', {}).get('r', float('nan')),
                    'length': info.get('episode', {}).get('l', float('nan')),
                    'lap_time': info.get('lap_time', float('nan')),
                    'crashed': info.get('crashed', False),
                    'laptimes': info.get('laptimes', []),
                    'laptime_min': info.get('laptime_min', float('nan')),
                    'laptime_max': info.get('laptime_max', float('nan')),
                    'laptime_mean': info.get('laptime_mean', float('nan')),
                    'step': info.get('step', float('nan')),
                    'global_step': info.get('global_step', float('nan')),
                    'checkpoints_per_lap': info.get('checkpoints_per_lap', float('nan')),
                }

                self.writer.writerow(log_data)
                self.csv_file.flush()

        # Only plot every 10 environment steps
        if self.n_calls % 1000 == 0 and self.n_calls > 100:
            plot_training_csv(self.model_name, self.counter)
        return True

    def _on_training_end(self):
        if self.csv_file:
            self.csv_file.close()
