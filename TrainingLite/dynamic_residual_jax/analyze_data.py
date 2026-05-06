# Load csv file from training_data directory
import pandas as pd
import os
import matplotlib.pyplot as plt


from utilities.state_utilities import STATE_VARIABLES

state_cols = STATE_VARIABLES
control_cols = ['control_angular', 'control_translational']

# Load csv file from training_data directory
csv_file = os.path.join(os.path.dirname(__file__), 'training_data', '2025-10-10_02-08-52_Recording1_0_IPZ8_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None__training_data.csv')
df = pd.read_csv(csv_file)

# Print the first 5 rows of the dataframe
print(df.head())



# Plot only a range of the data

df = df.iloc[1000:1500]

# Plot the data
for col in state_cols:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot state column itself
    axes[0].plot(df[col])
    axes[0].plot(df[f'expected_state_{col}'])
    axes[0].plot(df[f'residual_state_{col}']) # use other scale for this plot   
    axes[0].set_title(f'{col}')
    axes[0].set_ylabel(col)
    axes[0].grid(True)
    
    # Plot delta_state_col
    delta_col = f'delta_state_{col}'
    axes[1].plot(df[delta_col])
    axes[1].plot(df[f'expected_state_delta_{col}'])
    axes[1].set_title(f'{delta_col}')
    axes[1].set_ylabel(delta_col)
    axes[1].grid(True)

    
    # Plot change_rate_state_col
    change_rate_col = f'change_rate_{col}'
    axes[2].plot(df[change_rate_col])
    axes[2].plot(df[f'expected_state_change_rate_{col}'])
    axes[2].set_title(f'{change_rate_col}')
    axes[2].set_ylabel(change_rate_col)
    axes[2].set_xlabel('Time step')
    axes[2].grid(True)
   
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', col + '.png'))
    plt.close()