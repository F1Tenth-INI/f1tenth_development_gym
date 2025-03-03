import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
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


evaluation_mode = True
model_dir = "ppo_models"
model_name = "PPO_test"
model_name = "PPO_test"
model_save_name = "PPO_test"

class RacingEnv(gym.Env):
    def __init__(self):
        super(RacingEnv, self).__init__()
        self.simulation = None  # Delay initialization for SubprocVecEnv compatibility
        
        lidar_size = 40
        car_state_size = 1
        waypoint_size = 0
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
        self.stuck_counter = 0
        self.spin_counter = 0
        
        self.reward_history = []  # Store the last N rewards
        self.step_history = []  # Store the last N observations

    
    def reset(self, seed=None, options=None):
        
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
        
        self.simulation.env.reset(poses=np.array(self.simulation.starting_positions))
        self.last_progress = 0.0
        return self.get_observation(), {}

    def step(self, action):
        steering = np.clip(action[0], -1, 1) * 0.4
        throttle = np.clip(action[1], -1, 1) * 4.5 + 3.5
        agent_controls = np.array([[steering, throttle]])

        self.simulation.simulation_step(agent_controls=agent_controls)
        for index, driver in enumerate(self.simulation.drivers):
            self.simulation.update_driver_state(driver, index)
            driver.update_render_utils()

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
        if terminated and self.simulation.obs["collisions"][0] == 1:
            penalty = -50  # Adjust penalty amount
            for i in range(len(self.reward_history)):
                self.reward_history[i] += penalty * (i / len(self.reward_history))  # Scale penalty over time

        return obs, reward, terminated, truncated, {}

    def get_observation(self):
        driver : CarSystem = self.simulation.drivers[0]
        car_state = driver.car_state
        driver.waypoint_utils.update_next_waypoints(car_state)
        
        lidar_scan = driver.LIDAR.processed_ranges
        next_waypoints = driver.waypoint_utils.next_waypoint_positions_relative[:, 1]
        state_features = np.array([
            # car_state[POSE_X_IDX],
            # car_state[POSE_Y_IDX],
            car_state[LINEAR_VEL_X_IDX],
            # car_state[LINEAR_VEL_Y_IDX],
            # car_state[ANGULAR_VEL_Z_IDX],
            # car_state[POSE_THETA_COS_IDX],
            # car_state[POSE_THETA_SIN_IDX],
            # car_state[STEERING_ANGLE_IDX]
        ], dtype=np.float32)

        return np.concatenate([lidar_scan, state_features]).astype(np.float32)
        # return np.concatenate([lidar_scan, state_features, next_waypoints]).astype(np.float32)

    def _calculate_reward(self):
        driver = self.simulation.drivers[0]  
        waypoint_utils = driver.waypoint_utils
        car_state = driver.car_state

        reward = 0

        # Reward Track Progress (Incentivize Moving Forward)
        progress = waypoint_utils.get_cumulative_progress()
        delta_progress = progress - self.last_progress
        progress_reward = delta_progress * 1000.0 #+ progress * 100.0
        # if(progress_reward > 5.):
            # print(f"Progress: {progress}, Delta Progress: {delta_progress}, Reward: {progress_reward}")
        if progress_reward < 0:
            progress_reward *= 2.0  # Increase penalty for going backwards
        reward += progress_reward
        # print(f"Progress: {progress}, Reward: {progress_reward}")

        # ✅ 2. Reward Maintaining Speed (Prevent Stops and Stalls)
        # speed = car_state[LINEAR_VEL_X_IDX]
        # reward += speed * 0.8  # Encourage faster driving

        # # ✅ 3. Penalize Sudden Steering Changes
        # steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        # reward -= steering_diff * 0.1  # Increased penalty to discourage aggressive corrections

        # Penalize Collisions
        if self.simulation.obs["collisions"][0] == 1:
            reward -= 100  # Keep high penalty for crashes

        # ✅ 5. Penalize Distance from Raceline
        # nearest_waypoint_index, nearest_waypoint_dist = get_nearest_waypoint(car_state, waypoint_utils.next_waypoints)
        # if nearest_waypoint_dist < 0.05:
        #     nearest_waypoint_dist = 0
        # wp_penality = nearest_waypoint_dist**2 * 1.5
        # reward -= wp_penality
        # if evaluation_mode:
        #     print(f"Nearest waypoint distance: {nearest_waypoint_dist}, penality: {-wp_penality}")


        #  Penalize Spinning (Fixing Instability)
        if abs(car_state[ANGULAR_VEL_Z_IDX]) > 5.0:
            self.spin_counter += 1
            if self.spin_counter >= 50:
                reward -= self.spin_counter * 0.5
            
            if self.spin_counter >= 200:
                reward -= 500

        else:
            self.spin_counter = 0

       
        self.last_steering = car_state[STEERING_ANGLE_IDX]
        self.last_progress = progress
        
        if(evaluation_mode):
            print(f"Reward: {reward}")
        return reward


    def check_termination(self):
        driver = self.simulation.drivers[0] # 
        waypoint_utils = driver.waypoint_utils
        car_state = driver.car_state
        
        # Terminate if the car is stuck
        if self.spin_counter > 200:
            print("Car is spinning!")
            return True
            
        # Terminate if the car is spinning
        if car_state[ANGULAR_VEL_Z_IDX] > 20.0:
            print("Car is spinning2!")
            return True
        
        if self.simulation.obs["collisions"][0] == 1:
            # print("Car crashed!")
            return True
        return False


def make_env():
    """Factory function to create a new RacingEnv instance."""
    def _init():
        return RacingEnv()
    return _init

def evaluate_model():
    model_path = os.path.join(model_dir, model_save_name)


    time.sleep(1)

    model = PPO.load(model_path)
    # model = SAC.load(model_path)
    
    env = make_env()()
    check_env(env)
        
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    exit()
    

    
if __name__ == "__main__":
  
    
    # Load existing model or create new

    model_path = os.path.join(model_dir, model_name)


    if evaluation_mode:
        Settings.RENDER_MODE = 'human'
        # Settings.SAVE_RECORDINGS = True
        # Settings.SAVE_PLOTS = True  
        time.sleep(1)
        from run.run_simulation import RacingSimulation
        evaluate_model()  

    
    from utilities.car_system import CarSystem
    from run.run_simulation import RacingSimulation

    debug = False
    num_envs = 1
    
    if(debug): # Single environment
        env = make_env()()
        check_env(env)
        
    else: # Parallel environments 
        # fast (cumputationally heavy)
        Settings.RENDER_MODE = None
        num_envs = 12
        
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
            net_arch=[128, 128],
            # activation_fn=torch.nn.ReLU
        )
        model = PPO("MlpPolicy", env, verbose=1,
                    policy_kwargs=policy_kwargs,
                    n_steps=2048 // num_envs,
                    ent_coef=0.03,  # Start with high exploration
                    gamma=0.98)
        
    
    # # Load existing model or create new
    # try:
    #     model = SAC.load(model_path, env=env)
    #     print("SAC Model loaded successfully.")
    # except FileNotFoundError:
    #     print("No existing SAC model found. Creating a new one.")
    # policy_kwargs = dict(
    #     net_arch=[128, 128],
    # )
    # model = SAC("MlpPolicy", env, verbose=1,
    #             policy_kwargs=policy_kwargs,
    #             buffer_size=1000000,
    #             batch_size=256,
    #             ent_coef="auto",
    #             gamma=0.98,
    #             learning_rate=3e-4,
    #             train_freq=1,
    #             gradient_steps=1)
            
        

    # Save the current Python file under the model name
    import shutil
    shutil.copy(__file__, f"{model_path}.py")
    
    
    then = time.time()
    model.learn(total_timesteps=1000000, callback=TrainingStatusCallback(check_freq=5000, save_path=model_path))
    model.save(model_path)
    print(f"Training took {time.time() - then} seconds.")
    
    
    
   

