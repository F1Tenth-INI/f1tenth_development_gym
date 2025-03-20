import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utilities.Settings import Settings
from TrainingLite.rl_racing.train_model import make_env, model_dir, model_name, log_dir

import numpy as np

from tensorboardX import SummaryWriter

model_name = model_name + '_running'
# model_name = "sac_ini_1_rca1_22200000"
def evaluate_model(recording_name_extension=""):
    experiment_name = model_name
    Settings.RENDER_MODE = 'human'
    Settings.RECORDING_FOLDER = os.path.join(Settings.RECORDING_FOLDER, experiment_name) + '/'
    Settings.DATASET_NAME = Settings.DATASET_NAME + '_' + str(recording_name_extension)
    Settings.SAVE_RECORDINGS = True
    Settings.SAVE_PLOTS = True  

    time.sleep(0.1)

    from run.run_simulation import RacingSimulation  # Import inside function to avoid issues

    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path + ".zip"):
        print(f"Model {model_path} not found.")
        return

    model = SAC.load(model_path)
    env = make_env()()
    env.max_episode_steps = 8000

    obs, _ = env.reset()
    total_reward = 0
    rewards = []
    
    for _ in range(env.max_episode_steps):  # Run for fixed steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)
        total_reward += reward

        if done or truncated:
            break  # Stop if episode ends

    # ✅ Extract driver data AFTER evaluation
    driver: CarSystem = env.simulation.drivers[0]

    mean_lap_time = np.mean(driver.laptimes) if driver.laptimes else float('nan')
    mean_reward = np.mean(rewards)
    
    print(f"Mean reward: {mean_reward}, Mean Lap Time: {mean_lap_time}")

    # ✅ Log results to TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
    writer.add_scalar("evaluation/mean_lap_time", mean_lap_time, int(recording_name_extension))
    writer.add_scalar("evaluation/mean_reward", mean_reward, int(recording_name_extension))
    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PPO model')
    parser.add_argument('timestep', type=int, nargs='?', default=0, help='Training time step that is evaluated')
    
    args = parser.parse_args()
    timestep = int(args.timestep) 
    print(f"Evaluating model at timestep {timestep}")
    
    evaluate_model(timestep)