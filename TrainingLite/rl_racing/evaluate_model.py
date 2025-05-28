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
import torch
model_name = model_name + '_running'

device = 'cpu'
# model_name = "sac_ini_1_rca1_22200000"
def evaluate_model(recording_name_extension=""):
    experiment_name = model_name
    # Settings.RENDER_MODE = 'human'
    Settings.RENDER_MODE = None
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
    
    # Dynamically set device and map model to CPU if CUDA is unavailable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAC.load(model_path, map_location=device)  # Use map_location to handle CPU fallback

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
        
    env.simulation.on_simulation_end()
    env.close()
    print(f"Total reward: {total_reward}")
    print(f"Mean reward: {np.mean(rewards)}")
    
    # ✅ Extract driver data AFTER evaluation
    driver: CarSystem = env.simulation.drivers[0]

    mean_lap_time = np.mean(driver.laptimes) if driver.laptimes else float('nan')
    min_lap_time = np.min(driver.laptimes) if driver.laptimes else float('nan')
    max_lap_time = np.max(driver.laptimes) if driver.laptimes else float('nan')
    mean_reward = np.mean(rewards)
    


    # Save results into dictionary
    results = {
        'timestep': recording_name_extension,
        'mean_reward': mean_reward,
        'min_lap_time': min_lap_time,
        'max_lap_time': max_lap_time,
        'mean_lap_time': mean_lap_time,
        'total_reward': total_reward
    }

    # Print results
    print(f"Mean lap time: {mean_lap_time}")
    print(f"Min lap time: {min_lap_time}")
    print(f"Max lap time: {max_lap_time}")
    print(f"Total reward: {total_reward}")
    print(f"Mean reward: {mean_reward}")
    print(f"Mean lap time: {mean_lap_time}")
    


    # Save the results to a CSV file (add at the end of the file or create the file if it doesn't exist)
    results_file = os.path.join(Settings.RECORDING_FOLDER, 'results.csv')

    file_exists = os.path.exists(results_file)
    with open(results_file, 'a') as f:
        if not file_exists:
            f.write(','.join(results.keys()) + '\n') 
        # Write the values of the dictionary as a new row
        f.write(','.join(map(str, results.values())) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PPO model')
    parser.add_argument('timestep', type=int, nargs='?', default=0, help='Training time step that is evaluated')
    
    args = parser.parse_args()
    timestep = int(args.timestep) 
    print(f"Evaluating model at timestep {timestep}")
    
    evaluate_model(timestep)