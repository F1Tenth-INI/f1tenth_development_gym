import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time

from utilities.Settings import Settings
from utilities.state_utilities import *
from utilities.waypoint_utils import *
from TrainingLite.ppo_racing.TrainingCallback import TrainingStatusCallback


class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv, self).__init__()
        self.simulation = None  # Delay initialization for SubprocVecEnv compatibility
        
        lidar_size = 40
        car_state_size = 8
        waypoint_size = 30
        total_obs_size = car_state_size + lidar_size + waypoint_size
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.last_progress = 0.0
        self.last_steering = 0.0
    
    def reset(self, seed=None, options=None):
        if self.simulation is None:
            self.simulation = RacingSimulation()
            self.simulation.prepare_simulation()
        
        self.simulation.env.reset(poses=np.array(self.simulation.starting_positions))
        self.last_progress = 0.0
        return self.get_observation(), {}

    def step(self, action):
        steering = np.clip(action[0], -1, 1) * 0.4
        throttle = np.clip(action[1], -1, 1) * 4.5 + 3.5
        agent_controls = np.array([[steering, throttle]])

        self.simulation.simulation_step(agent_controls=agent_controls)
        # print(agent_controls[0])
        for index, driver in enumerate(self.simulation.drivers):
            self.simulation.update_driver_state(driver, index)

        obs = self.get_observation()
        reward = self._calculate_reward()
        terminated = self.check_termination()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def get_observation(self):
        driver = self.simulation.drivers[0]
        car_state = driver.car_state
        driver.waypoint_utils.update_next_waypoints(car_state)
        
        lidar_scan = driver.LIDAR.processed_ranges
        next_waypoints = driver.waypoint_utils.next_waypoint_positions_relative[:, 1]
        state_features = np.array([
            car_state[POSE_X_IDX],
            car_state[POSE_Y_IDX],
            car_state[LINEAR_VEL_X_IDX],
            car_state[LINEAR_VEL_Y_IDX],
            car_state[ANGULAR_VEL_Z_IDX],
            car_state[POSE_THETA_COS_IDX],
            car_state[POSE_THETA_SIN_IDX],
            car_state[STEERING_ANGLE_IDX]
        ], dtype=np.float32)

        return np.concatenate([lidar_scan, state_features, next_waypoints]).astype(np.float32)

    def _calculate_reward(self):
        driver = self.simulation.drivers[0]
        waypoint_utils = driver.waypoint_utils
        car_state = driver.car_state

        
        reward = 0
        
        progress = waypoint_utils.get_cumulative_progress()
        delta_progress = progress - self.last_progress
        reward = delta_progress * 200.0 + progress * 5.0

        speed = car_state[LINEAR_VEL_X_IDX]
        reward += speed * 0.5

        steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        reward -= steering_diff * 0.05

        if self.simulation.obs["collisions"][0] == 1:
            reward -= 200

        nearest_waypoint_index, nearest_waypoint_dist = get_nearest_waypoint(car_state, waypoint_utils.waypoints)
        if nearest_waypoint_dist < 0.15:
            nearest_waypoint_dist = 0
        reward -= nearest_waypoint_dist * 100.0

        if speed < 0.5:
            reward -= 5 + (0.3 - speed) * 20

        if speed < 1.0 and abs(car_state[STEERING_ANGLE_IDX]) > 0.2:
            reward -= 1

        self.last_steering = car_state[STEERING_ANGLE_IDX]
        self.last_progress = progress
        return reward

    def check_termination(self):
        if self.simulation.obs["collisions"][0] == 1:
            return True
        return False


def make_env():
    """Factory function to create a new RacingEnv instance."""
    def _init():
        return RacingEnv()
    return _init

def evaluate_model(model_path=None):


    time.sleep(2)

    model = PPO.load(model_path)
    
    env = make_env()()
    check_env(env)
        
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    exit()
    

    
if __name__ == "__main__":
  
    
    # Load existing model or create new
    model_dir = "ppo_models"
    model_name = "ppo_parallel1m_1"
    model_path = os.path.join(model_dir, model_name)


    # Settings.RENDER_MODE = 'human'
    # from run.run_simulation import RacingSimulation
    # evaluate_model(model_path)

    
    
    from run.run_simulation import RacingSimulation

    debug = False
    num_envs = 4
    
    if(debug): # Single environment
        env = make_env()()
        check_env(env)
        
    else: # Parallel environments 
        # Superfast (cumputationally heavy)
        # Crash info not avaiable in terminal
        Settings.RENDER_MODE = None
        num_envs = 8
        
        env = SubprocVecEnv([make_env() for _ in range(num_envs)])
        # env = DummyVecEnv([make_env() for _ in range(num_envs)])
        
        env.reset()


    
    # Load existing model or create new
   
    try:
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No existing model found. Creating a new one.")
        policy_kwargs = dict(
            net_arch=[256, 128, 64],
            # activation_fn=torch.nn.ReLU
        )
        model = PPO("MlpPolicy", env, verbose=1,
            policy_kwargs=policy_kwargs,
            n_steps=2048 // num_envs,
            ent_coef=0.03,  # Start with high exploration
            gamma=0.995)     
        
    

    # Save the current Python file under the model name
    import shutil
    shutil.copy(__file__, f"{model_path}.py")
    
    
    then = time.time()
    model.learn(total_timesteps=100000, callback=TrainingStatusCallback(check_freq=5000))
    model.save(model_path)
    print(f"Training took {time.time() - then} seconds.")
    
    
    
   

