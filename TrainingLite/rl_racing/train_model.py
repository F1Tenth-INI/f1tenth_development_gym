import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import torch # Important: only import torch in the main thread
    # Check if CUDA is available and set the device accordingly
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    
import sys
sys.modules["tensorflow"] = None
try:
    import tensorflow as tf
except Exception as e:
    pass
    # print(f"TensorFlow is forbidden: {e}")
    # print("not importing it saves you 200mb of memory :)")

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import time
from typing import Optional

import zipfile

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)
from utilities.state_utilities import *
from utilities.waypoint_utils import *
from TrainingLite.rl_racing.TrainingCallback import TrainingStatusCallback, AdjustCheckpointsCallback, EpisodeCSVLogger

from stable_baselines3.common.vec_env import VecMonitor

from TopKReplayBuffer import TopKTrajectoryBuffer, TopKTrajectoryCallback


model_load_name = "SAC_RCA1_wpts_lidar_4"
# model_load_name = "SAC_RCA1_wpts_lidar_2"
model_name = "SAC_RCA1_wpts_lidar_4"



experiment_index = 0
tensorboad_log_name = f"{model_name}_{experiment_index}"

model_dir = os.path.join(root_dir, "TrainingLite","rl_racing","models", model_name)
log_dir = os.path.join(root_dir,"TrainingLite","rl_racing","models", model_name, "logs") + '/'

print_info = False

topk = TopKTrajectoryBuffer(k=20)
topk_callback = TopKTrajectoryCallback(topk_buffer=topk)


class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv, self).__init__()
        self.simulation:Optional[RacingSimulation]  = None  # Delay initialization for SubprocVecEnv compatibility
                
        lidar_size = 40
        car_state_size = 4
        waypoint_size = 30 
        total_obs_size = car_state_size + lidar_size + waypoint_size
        
        # Define realistic bounds for the observation space
        lidar_low = np.zeros(lidar_size, dtype=np.float32)
        lidar_high = np.ones(lidar_size, dtype=np.float32) * 20.0  # Assuming lidar values are clipped between 0 and 10
        
        car_state_low = np.array([-1, -1, -1, -1], dtype=np.float32)  # Example bounds for car state
        car_state_high = np.array([1, 1, 1, 1], dtype=np.float32)
        
        # control_obs_low = np.array([-0.4, -10], dtype=np.float32)
        # control_obs_high= np.array([0.4, 10], dtype=np.float32)
        
        
        waypoint_low = - 1.0 * np.ones(waypoint_size, dtype=np.float32)  # Adjust based on actual waypoint data
        waypoint_high = 1.0 * np.ones(waypoint_size, dtype=np.float32)  # Adjust based on actual waypoint data
        
        # low = np.concatenate([lidar_low, car_state_low, waypoint_low])
        # high = np.concatenate([lidar_high, car_state_high, waypoint_high])
        
        low = np.concatenate([car_state_low, waypoint_low, lidar_low])
        high = np.concatenate([car_state_high, waypoint_high, lidar_high])

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.global_step_counter = 0
        self.eposode_step_counter = 0
        self.last_progress = 0.0
        self.last_steering = 0.0
        self.stuck_counter = 0
        self.spin_counter = 0
        
        self.checkpoints_per_lap = 20
        
        self.max_episode_steps = 10000
        self.episode_count = 0
        self.episode_crash_history = []  # Store the last N crashes
        self._adjust_checkpoint = False

        self.reward_history = []  # Store the last N rewards
        self.step_history = []  # Store the last N observations
        
    
    def reset(self, seed=None, options=None):
        from run.run_simulation import RacingSimulation
        self.eposode_step_counter = 0
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
        self.simulation.reset(poses=np.array(self.simulation.starting_positions))
        
        driver : CarSystem = self.simulation.drivers[0]
        driver.reward_calculator.checkpoint_fraction = 1 / self.checkpoints_per_lap
        # Make sure env and self.simulation resets propperly
    
        # Check for open source race environment        
        return self.get_observation(), {}

    def step(self, action):
        
        simulation: Optional[RacingSimulation] = self.simulation
        self.eposode_step_counter += 1  # Steps within an episode
        self.global_step_counter += 1   # Training-wide step counter

        steering_correction = np.clip(action[0], -1, 1) * 0.4
        throttle_correction = np.clip(action[1], -1, 1) * 5

        # steering_correction = 0
        # throttle_correction = 0
        # Get original PP controls
        # for index, driver in enumerate(simulation.drivers):
            
        driver: CarSystem = simulation.drivers[0]
        simulation.update_driver_state(driver, 0)
        driver.process_observation()
        steering_controller, throttle_controller = None, None
        # If there is no controller
        if(steering_controller is None or throttle_controller is None):
            steering_controller = 0
            throttle_controller = 0


        # Apply the action to the driver
        steering = np.clip(steering_controller + steering_correction, -0.4, 0.4)
        throttle = np.clip(throttle_controller + throttle_correction, -10, 10)

        agent_controls = np.array([[steering, throttle]])


        # Run multiple steps of simulation
        for _ in range(5):
            simulation.simulation_step(agent_controls=agent_controls)

        obs = self.get_observation()
        reward = self._calculate_reward()
        terminated = self.check_termination()
        truncated = False

        laptime_min = None
        laptime_max = None
        laptime_mean = None
        laptimes = None

        if hasattr(self.simulation, "drivers") and self.simulation.drivers and hasattr(self.simulation.drivers[0], "laptimes"):
            laptimes = self.simulation.drivers[0].laptimes
            if laptimes:
                laptime_min = np.min(laptimes)
                laptime_max = np.max(laptimes)
                laptime_mean = np.mean(laptimes)

            
        info = {
            "laptimes": laptimes,
            "laptime_min": laptime_min,
            "laptime_max": laptime_max,
            "laptime_mean": laptime_mean,
            "crashed": self.simulation.obs["collisions"][0] == 1,
            "episode": {
                "r": reward,
                "l": self.eposode_step_counter,
                "t": self.eposode_step_counter / 100.0,  # Assuming 100 FPS
            },
            "step": self.eposode_step_counter,
            "global_step": self.global_step_counter,
            "checkpoints_per_lap": self.checkpoints_per_lap,
        }


        return obs, reward, terminated, truncated, info

    
    def close(self):
        pass

    def get_observation(self):
        driver : CarSystem = self.simulation.drivers[0]
        
        car_state = driver.car_state
        driver.waypoint_utils.update_next_waypoints(car_state)
        
        lidar_scan = np.clip(driver.LIDAR.processed_ranges, 0.001, 10.0)  # Lidar scan values
        # lidar_scan = np.clip(1.0 / (driver.LIDAR.processed_ranges + 1e-3), 0, 10.0) 

        wpts_x = driver.waypoint_utils.next_waypoint_positions_relative[:, 0]
        wpts_y = driver.waypoint_utils.next_waypoint_positions_relative[:, 1]   
        # Concatenate to 1d array
        wpts = np.concatenate([wpts_x, wpts_y])
        wpts = wpts[::2]

    

        state_features = np.array([
            # car_state[POSE_X_IDX],
            # car_state[POSE_Y_IDX],
            car_state[LINEAR_VEL_X_IDX],
            car_state[LINEAR_VEL_Y_IDX],
            car_state[ANGULAR_VEL_Z_IDX],
            # car_state[POSE_THETA_COS_IDX],
            # car_state[POSE_THETA_SIN_IDX],
            car_state[STEERING_ANGLE_IDX]
        ], dtype=np.float32)
        
        control_features = np.array([
            driver.angular_control,
            driver.translational_control,
        ], dtype=np.float32)
        
        observation_array = np.concatenate([state_features, wpts, lidar_scan]).astype(np.float32)

        normalization_array =  [0.1, 1.0, 0.5, 1 / 0.4] + [0.1] * len(wpts)+ [0.1] * len(lidar_scan) # Adjust normalization factors for each feature

        observation_array_normalized = observation_array *  normalization_array
        # print(f"Observation: {observation_array_normalized}")

        return observation_array_normalized

    def _calculate_reward(self):
        
        driver : CarSystem = self.simulation.drivers[0]
        reward = driver.reward 
        # Penalize crash
        if self.simulation.obs["collisions"][0] == 1:
            reward = -300
        
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
        
        # if driver.reward_calculator.stuck_counter > 200:
        #     # print("Car is stuck!")
        #     return True
            
        # Terminate if the car is spinning
        if car_state[ANGULAR_VEL_Z_IDX] > 20.0:
            print("Car is spinning2!")
            return True
        
        if self.simulation.obs["collisions"][0] == 1:
            # print("Car crashed!")
            return True
        
        if self.eposode_step_counter > self.max_episode_steps:
            print("Max lenght reached!")
            return True
        
        return False

def lr_schedule(progress_remaining: float) -> float:
    """
    Learning rate schedule for Stable-Baselines3.
    - progress_remaining = 1.0 at the start of training
    - progress_remaining = 0.0 at the end of training
    """
    base_lr = 0.001  # Initial learning rate
    min_lr = 0.0001  # Minimum learning rate
    
    return max(min_lr, base_lr * progress_remaining)


from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, VecFrameStack

def make_env(repeat=4):
    def _init():
        env = RacingEnv()
        return env
    return _init


if __name__ == "__main__":
    # Load existing model or create new

    model_path = os.path.join(model_dir, model_name)

    from utilities.car_system import CarSystem
    from run.run_simulation import RacingSimulation

    debug = False
    print_info = False
    num_envs = 16
 
    if(debug): # Single environment
        num_envs = 1
        env = DummyVecEnv([make_env() for _ in range(num_envs)])
    else:
        env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    
    # env = VecFrameStack(env, n_stack=4, channels_order="last")
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    env = VecMonitor(env, log_dir)  # Keeps track of episode rewards
    env.reset()



    # Load existing model or create new
    model_load_path = os.path.join(model_dir, model_load_name)
    try:
        model = SAC.load(model_load_path, env=env)
        model.lr_schedule = lr_schedule  # Set the learning rate schedule
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No existing model found. Creating a new one.")
        
        # os.makedirs(model_path)
        import torch.nn as nn

        policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)
        model = SAC(
            "MlpPolicy", env, verbose=1,
            train_freq=1,
            gamma=0.98, # discount factor
            learning_rate=1e-3,
            # learning_rate=lr_schedule,  # Use the dynamic learning rate function
            gradient_steps=40,  # Number of gradient steps to perform after each rollout
            policy_kwargs=policy_kwargs,
            buffer_size=100_000,
            learning_starts=10_000,
            # tensorboard_log=os.path.join(log_dir),  # Enable TensorBoard logging
            device=device,  # Ensure the model is trained on GPU
            batch_size=256 
        )
        
        # UTD ≈ gradient_steps / (train_freq * n_envs)



    # Save the current Python file under the model name
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create a zip file
    zip_path = f"{model_path}_training.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(__file__, os.path.basename(__file__))

    starting_time = time.time()

    model.learn(total_timesteps=100_000,
                callback=[
                    # topk_callback,
                    TrainingStatusCallback(check_freq=12_500, save_path=model_path),
                    EpisodeCSVLogger(csv_path=os.path.join(model_dir, 'training_log.csv')),
                    # AdjustCheckpointsCallback()
                ],
    )

    model.save(model_path)
    print(f"Model saved to {model_path}.")
    # Save the top-K trajectories
    topk.save("top_k_trajectories.pkl")
    print("Top-K trajectories saved.")

    # Save VecNormalize statistics for later evaluation
    norm_path = os.path.join(model_dir, "vecnormalize.pkl")
    # env.save(norm_path)
    # print(f"VecNormalize statistics saved to {norm_path}.")

    env.close()  # Close TensorBoard writer after training

    print(f"Training took {time.time() - starting_time} seconds.")
    
    
