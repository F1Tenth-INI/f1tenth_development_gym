import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utilities.Settings import Settings
from TrainingLite.rl_racing.train_model import make_env, model_dir, model_name


model_name = model_name + '_running'


def evaluate_model(recording_name_extension=""):
    experiment_name = model_name
    Settings.RENDER_MODE = 'human'
    Settings.RECORDING_FOLDER = os.path.join(Settings.RECORDING_FOLDER, experiment_name) + '/'
    Settings.DATASET_NAME = Settings.DATASET_NAME +'_'+ str(recording_name_extension)
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
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PPO model')
    parser.add_argument('recording_name_extension', type=str, nargs='?', default='', help='Recording name extension')
    args = parser.parse_args()

    
    evaluate_model(args.recording_name_extension)