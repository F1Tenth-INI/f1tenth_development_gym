import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
from typing import Optional

import zipfile

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import *
from TrainingLite.rl_racing.TrainingCallback import TrainingStatusCallback

from stable_baselines3.common.vec_env import VecMonitor

model_name = "sac_home_1_rca1"

model_dir = os.path.join(root_dir, "TrainingLite","rl_racing","models", model_name)
log_dir = os.path.join(root_dir,"TrainingLite","rl_racing","models", model_name, "logs") + '/'
tensorboard_log_dir = os.path.join(model_dir, "tensorboard_logs")

print_info = False

class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv, self).__init__()
        self.simulation:Optional[RacingSimulation]  = None  # Delay initialization for SubprocVecEnv compatibility
        
        lidar_size = 40
        car_state_size = 6
        waypoint_size = 0
        total_obs_size = car_state_size + lidar_size + waypoint_size
        
        # Define realistic bounds for the observation space
        lidar_low = np.zeros(lidar_size, dtype=np.float32)
        lidar_high = np.ones(lidar_size, dtype=np.float32) * 20.0  # Assuming lidar values are clipped between 0 and 10
        
        car_state_low = np.array([-np.inf,-np.inf, -10.0, -5, -10.0, -2*np.pi], dtype=np.float32)  # Example bounds for car state
        car_state_high = np.array([np.inf, np.inf, 15.0, 5, 10.0, 2*np.pi], dtype=np.float32)
        
        waypoint_low = np.zeros(waypoint_size, dtype=np.float32)  # Adjust based on actual waypoint data
        waypoint_high = np.ones(waypoint_size, dtype=np.float32)  # Adjust based on actual waypoint data
        
        low = np.concatenate([lidar_low, car_state_low, waypoint_low])
        high = np.concatenate([lidar_high, car_state_high, waypoint_high])
        
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.step_counter = 0
        self.last_progress = 0.0
        self.last_steering = 0.0
        self.stuck_counter = 0
        self.spin_counter = 0
        
        self.reward_history = []  # Store the last N rewards
        self.step_history = []  # Store the last N observations

    
    def reset(self, seed=None, options=None):
        from run.run_simulation import RacingSimulation
        self.step_counter = 0
        self.last_progress = 0.0
        self.last_steering = 0.0
        self.stuck_counter = 0
        self.spin_counter = 0
        
        
        self.reward_history = []  # Store the last N rewards
        self.step_history = []  # Store the last N observations
        self.N = 10  # Number of steps to penalize before a collision

        
        if self.simulation is None:
            self.simulation = RacingSimulation()
            self.simulation.prepare_simulation()
        else: 
            self.simulation.init_drivers() # at least re-init drivers in any case
            
        self.simulation.get_starting_positions() # reinitialize starting positions in case of randomization
        self.simulation.env.reset(poses=np.array(self.simulation.starting_positions))
        # Make sure env and self.simulation resets propperly
    
        # Check for open source race environment        
        return self.get_observation(), {}

    def step(self, action):
        
        simulation: Optional[RacingSimulation] = self.simulation
        self.step_counter += 1
        steering = np.clip(action[0], -1, 1) * 0.4
        throttle = np.clip(action[1], -1, 1) * 10
        
        agent_controls = np.array([[steering, throttle]])

        simulation.simulation_step(agent_controls=agent_controls)
        for index, driver in enumerate(simulation.drivers):
            driver : CarSystem = driver
            simulation.update_driver_state(driver, index)
            driver.update_render_utils()
            driver.update_waypoints()
            driver.process_data_post_control()

        obs = self.get_observation()
        reward = self._calculate_reward()
        terminated = self.check_termination()
        truncated = False

        # Store the last N steps
        self.reward_history.append(reward)
        self.step_history.append(obs)
        if len(self.reward_history) > self.N:
            self.reward_history.pop(0)
            self.step_history.pop(0)

        # If collision, penalize past N steps
        if terminated and simulation.obs["collisions"][0] == 1:
            penalty = -20  # Adjust penalty amount
            for i in range(len(self.reward_history)):
                self.reward_history[i] += penalty * (i / len(self.reward_history))  # Scale penalty over time

        if terminated:
            simulation.drivers[0].on_simulation_end()

        return obs, reward, terminated, truncated, {}

    def get_observation(self):
        driver : CarSystem = self.simulation.drivers[0]
        
        car_state = driver.car_state
        driver.waypoint_utils.update_next_waypoints(car_state)
        
        lidar_scan = driver.LIDAR.processed_ranges
        lidar_scan = np.clip(1.0 / (driver.LIDAR.processed_ranges + 1e-3), 0, 10.0)

        next_waypoints = driver.waypoint_utils.next_waypoint_positions_relative[:, 1]
        state_features = np.array([
            car_state[POSE_X_IDX],
            car_state[POSE_Y_IDX],
            car_state[LINEAR_VEL_X_IDX],
            car_state[LINEAR_VEL_Y_IDX],
            car_state[ANGULAR_VEL_Z_IDX],
            # car_state[POSE_THETA_COS_IDX],
            # car_state[POSE_THETA_SIN_IDX],
            car_state[STEERING_ANGLE_IDX]
        ], dtype=np.float32)

        return np.concatenate([lidar_scan, state_features]).astype(np.float32)
        # return np.concatenate([lidar_scan, state_features, next_waypoints]).astype(np.float32)

    def _calculate_reward(self):
        
        driver : CarSystem = self.simulation.drivers[0]
        reward = driver.reward_calculator._calculate_reward(driver)
        
        # Penalize crash
        if self.simulation.obs["collisions"][0] == 1:
            reward = -1000
            
        
        if(print_info):
            print(f"Reward: {reward}")
        return reward

    

    def check_termination(self):
        driver = self.simulation.drivers[0] # 
        car_state = driver.car_state
        
        # Terminate if the car is stuck
        if driver.reward_calculator.spin_counter > 200:
            print("Car is spinning!")
            return True
        
        if driver.reward_calculator.stuck_counter > 200:
            # print("Car is stuck!")
            return True
            
        # Terminate if the car is spinning
        if car_state[ANGULAR_VEL_Z_IDX] > 20.0:
            print("Car is spinning2!")
            return True
        
        if self.simulation.obs["collisions"][0] == 1:
            # print("Car crashed!")
            return True
        
        if self.step_counter > 20000:
            print("Max lenght reached!")
            return True
        
        return False


def make_env():
    """Factory function to create a new RacingEnv instance."""
    def _init():
        return RacingEnv()
    return _init


if __name__ == "__main__":
  
    
    # Load existing model or create new

    model_path = os.path.join(model_dir, model_name)

    
    from utilities.car_system import CarSystem
    from run.run_simulation import RacingSimulation

    debug = False
    print_info = False
    num_envs = 1
    
    if(debug): # Single environment
        env = make_env()()
        check_env(env)
        
    else: # Parallel environments 
        # fast (cumputationally heavy)
        # Settings.RENDER_MODE = 'human'
        num_envs = 24
        
        env = SubprocVecEnv([make_env() for _ in range(num_envs)])
        # env = DummyVecEnv([make_env() for _ in range(num_envs)])
            
        # Monitoring        
        env = VecMonitor(env, log_dir)
        
        env.reset()


    
    # Load existing model or create new
    try:
        model = SAC.load(model_path, env=env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No existing model found. Creating a new one.")
        
        # os.makedirs(model_path)
        policy_kwargs = dict(net_arch=[256, 256])
        model = SAC(
            "MlpPolicy", env, verbose=1,
            train_freq=1,
            gradient_steps=1,  # Number of gradient steps to perform after each rollout
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_dir,  # Enable TensorBoard logging
        )
        

    # model = PPO(
    #     "MlpPolicy", env, verbose=1,
    #     policy_kwargs=policy_kwargs,
    #     n_steps=int(2048/num_envs),  # Reduce step size for faster feedback
    #     tensorboard_log=tensorboard_log_dir,  # Enable TensorBoard logging
    # )
    

    

    # Save the current Python file under the model name


    # Save the current Python file under the model name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create a zip file
    zip_path = f"{model_path}_training.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(__file__, os.path.basename(__file__))
    
    
    then = time.time()
    model.learn(total_timesteps=30000000, callback=TrainingStatusCallback(check_freq=12500, save_path=model_path))
    
    model.save(model_path)
    print(f"Training took {time.time() - then} seconds.")
    
    
