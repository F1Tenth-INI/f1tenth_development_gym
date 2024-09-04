import numpy as np
from f110_gym.envs.state_indices_pacejka import StateIndices, ControlIndices
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd


dt = 0.01

# Assuming StateIndices and vehicle_dynamics_pacejka are defined elsewhere

# Initialize the initial state array with zeros
initial_state = np.zeros(StateIndices.number_of_states)

# Number of control input sequences
number_of_control_sequences = 2000

# Initialize a list to store the history of states
state_history = []
control_history = []


state = initial_state
for i in trange(number_of_control_sequences):
    # Generate random control inputs for steering and acceleration
    steering = np.random.normal(0, 0.2)  # Random steering input
    acceleration = np.random.uniform(0, 10)  # Random acceleration input
    
    # Simulate for 20 timesteps with the current control inputs
    for j in range(50):
        
        if abs(state[StateIndices.yaw_rate]) > 10:
            steering =-abs(steering)

        
        state += dt * vehicle_dynamics_pacejka(state, [steering, acceleration])
        # Append the current state to the history
        state_history.append(state.copy())
        control_history.append([steering, acceleration])



state_history = np.array(state_history)
control_history = np.array(control_history)


# Plotting the positions
plt.figure(figsize=(10, 6))
plt.plot(state_history[:, StateIndices.pose_x], state_history[:, StateIndices.pose_y], marker='o')
plt.title('Vehicle Position Over Time')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)
# plt.show()


combined_history = np.hstack((state_history, control_history))

headers =  ['pose_x', 'pose_y', 'pose_theta', 'linear_vel_x', 'linear_vel_y', 'angular_vel_z', 'steering_angle', 'angular_control', 'translational_control']
df = pd.DataFrame(combined_history, columns=headers)

# Add state augmentations
df['pose_theta_sin'] = np.sin(df['pose_theta'])
df['pose_theta_cos'] = np.cos(df['pose_theta'])

# Add slip angle
slip_angles = np.arctan2(df['linear_vel_y'], df['linear_vel_x']) 
df['slip_angle'] = slip_angles

# Add time
num_steps = len(df)
time_array = np.arange(0, num_steps*dt, dt)  # Create a time array from 0 to num_steps*dt, stepping by dt
df['time'] = time_array

df.to_csv("vehicle_state_history_with_slip_angle.csv", index=False)
df.to_csv("vehicle_state_history_with_slip_angle.csv", index=False)