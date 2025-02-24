import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from run.run_simulation import RacingSimulation
from utilities.state_utilities import *
from utilities.waypoint_utils import *
from TrainingLite.ppo_racing.TrainingCallback import TrainingStatusCallback  # Import the new callback

class RacingEnv(gym.Env):
    def __init__(self, simulation):
        super(RacingEnv, self).__init__()
        self.simulation = simulation  # Use existing simulation
        self.simulation.prepare_simulation()        
        self.simulation.obs, self.simulation.step_reward, self.simulation.done, self.simulation.info = self.simulation.env.reset(poses=np.array(self.simulation.starting_positions) )


        # Observation space: LiDAR + car state
        lidar_size = len(self.simulation.drivers[0].LIDAR.processed_ranges)
        car_state_size = 8
        waypoint_size = 30
        total_obs_size = car_state_size + lidar_size + waypoint_size
        # total_obs_size = lidar_size + car_state_size + waypoint_size
        
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        # Action space: Steering and Throttle
        self.max_steer = np.radians(30)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),   # Normalized
            high=np.array([1, 1], dtype=np.float32),   # Normalized
            dtype=np.float32
        )

        self.last_next_waypoint_global_index = None
        self.spin_counter = 0  # Counter for consecutive spinning steps
        self.current_lap_time = 0.0  # Track lap time
        
        self.last_steering = 0.0  # Store last steering angle
        self.last_progress = 0.0

    def reset(self, seed=None, options=None):
        """Reset simulation and return initial observation."""
        self.simulation.prepare_simulation()
        self.simulation.env.reset(poses=np.array(self.simulation.starting_positions))

        self.current_lap_time = 0.0
        return self.get_observation(), {}

    def step(self, action):
        """Apply PPO agent's action and advance the simulation."""
        # steering, throttle = action
        
        steering = np.clip(action[0], -1, 1) * 0.4  # Scale to [-0.4, 0.4]
        throttle = np.clip(action[1], -1, 1) * 4.5 + 3.5  # Scale to [-1, 8]        
           
        agent_controls = np.array([[ steering, throttle]])

        # Run a single simulation step (like `simulation_step()`)
        # Overwrite control in `get_agent_controls()`
        self.simulation.simulation_step(agent_controls=agent_controls)

        # Update car state
        for index, driver in enumerate(self.simulation.drivers):
            self.simulation.update_driver_state(driver, index)

        # Extract new observation
        obs = self.get_observation()

        # Compute reward
        reward = self._calculate_reward()

        # Check for termination (crash or lap completion)
        terminated = self.check_termination()
        truncated = False  # No forced truncation

        return obs, reward, terminated, truncated, {}

    def get_observation(self):
        """Extract the current observation (LiDAR + car state)."""
        
        driver = self.simulation.drivers[0]
        car_state = driver.car_state
        driver.waypoint_utils.update_next_waypoints(car_state)
        
        lidar_scan = self.simulation.drivers[0].LIDAR.processed_ranges
        
        next_waypoints = self.simulation.drivers[0].waypoint_utils.next_waypoint_positions_relative
        next_waypoints_y = next_waypoints[:, 1]  # Y-coordinate of next waypoints

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

        
        # return state_features
        return np.concatenate([lidar_scan, state_features, next_waypoints_y]).astype(np.float32)  
    
    
    def _calculate_reward(self):
        """Compute the reward based on speed, stability, and braking before turns."""
        driver = self.simulation.drivers[0]
        waypoint_utils = driver.waypoint_utils
        car_state = driver.car_state

        reward = 0.0

        # ✅ Encourage Forward Progress, But Less Dominant
        #Delta progress in the last timestep:
        progress = waypoint_utils.get_cumulative_progress()
        delta_progress = progress - self.last_progress
        reward += delta_progress * 50.0  # Reduced from 25.0
        
        # Absolute progress
        progress = waypoint_utils.get_cumulative_progress()
        reward += progress * 5.0  # Reduced from 25.0

        # ✅ Stronger Speed Reward
        speed = car_state[LINEAR_VEL_X_IDX]
        reward += speed * 0.2  # Increased from 0.5

        # ✅ Penalize Steering Effort (Small Impact)
        # steering_penalty = abs(car_state[STEERING_ANGLE_IDX]) * 0.01
        # reward -= steering_penalty

        # ✅ Penalize Large Changes in Steering
        steering_diff = abs(car_state[STEERING_ANGLE_IDX] - self.last_steering)
        reward -= steering_diff * 0.05  # Reduced from 0.1 (to avoid discouraging corrections)

        # ✅ Penalize Collisions Strongly
        if self.simulation.obs["collisions"][0] == 1:
            reward -= 100

        # ✅ Penalize Distance from Raceline
        nearest_waypoint_index, nearest_waypoint_dist = get_nearest_waypoint(car_state, waypoint_utils.waypoints)
        if nearest_waypoint_dist < 0.2:
            nearest_waypoint_dist = 0
        reward -= nearest_waypoint_dist * 20.0  # Reduced from 0.1 (prevents overfitting to waypoints)

        # ✅ Penalize Standing Still for Too Long
        if speed < 0.3:  # Was < 0.5
            reward -= 5 + (0.3 - speed) * 20  # Progressive penalty for standing still

        # ✅ Penalize "Artificial" Braking
        if speed < 1.0 and abs(car_state[STEERING_ANGLE_IDX]) > 0.2:
            reward -= 10  # Increased from 10 to discourage stopping before corners

        self.last_steering = car_state[STEERING_ANGLE_IDX]  # Store last steering
        self.last_progress = progress
        return reward

        

    def check_termination(self):
        car_state = self.simulation.drivers[0].car_state
        spinning = abs(car_state[ANGULAR_VEL_Z_IDX]) > 10. and abs(car_state[LINEAR_VEL_X_IDX]) < 1.5
        
        if spinning:
            self.spin_counter += 1  # Count consecutive spinning steps
        else:
            self.spin_counter = 0  # Reset counter if normal behavior

        if self.spin_counter > 10:  # If spinning lasts for 10+ steps, reset
            print("🚨 Car is spinning! Restarting experiment.")
            return True

        """Check if the episode should end (crash or lap completed)."""
        if self.simulation.obs["collisions"][0] == 1:
            print("Car crashed! Ending episode.")
            return True

        if self.simulation.obs["lap_times"][0] > 0 and self.simulation.obs["lap_times"][0] < self.current_lap_time:
            print("Lap completed! Ending episode.")
            return True

        return False

# ---- Train PPO ---- #
if __name__ == "__main__":
    
    raceing_simulation = RacingSimulation()
    env = RacingEnv(simulation = raceing_simulation)
    check_env(env)  # Validate environment

    env = DummyVecEnv([lambda: env])

    try:
        model = PPO.load("ppo_test3", env=env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No existing model found. Creating a new one.")
        
        # Define the policy network architecture
        policy_kwargs = dict(
            net_arch=[256, 128],  # Two hidden layers: 256 neurons in first, 128 in second
            activation_fn=torch.nn.ReLU,  # Activation function for the layers (can be "tanh" or "relu")
            # log_std_init=-1,  # Reduce initial action variance to prevent extreme actions
            # ortho_init=False  # Disable orthogonal initialization (optional)
        )

        # Train PPO with custom network
        model = PPO("MlpPolicy", env, verbose=1,
                policy_kwargs=policy_kwargs,
                # learning_rate=2e-4,  # Keep default learning rate
                # gamma=0.99,  # Discount factor for long-term planning
                # clip_range=0.2,  # PPO clipping
                # ent_coef=0.005,  # Encourage exploration
                n_steps=1024,  # Batch size
                )


    model.learn(total_timesteps=100000, callback=TrainingStatusCallback(check_freq=5000))
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    model.save("ppo_test3")
