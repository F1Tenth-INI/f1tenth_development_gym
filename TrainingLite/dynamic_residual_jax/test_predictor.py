import os
import pandas as pd
import numpy as np
from predictor import Predictor

from train import INPUT_COLS, OUTPUT_COLS, MODEL_NAME

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(script_dir, 'models', MODEL_NAME)
csv_path = "/home/florian/Documents/INI/f1tenth_development_gym/TrainingLite/dynamic_residual_jax/training_data/processed_data.csv"

# Load predictor
predictor = Predictor(model_dir)

# Load CSV and extract input columns
df = pd.read_csv(csv_path, comment='#')

# Generate predictions for all rows (starting from row 10)
sequence_length = 10
predictions = []
output_dim = len(OUTPUT_COLS)

print(f"Generating predictions for {len(df) - sequence_length} rows...")
for i in range(sequence_length, len(df)):
    input_sequence = df[INPUT_COLS].values[i - sequence_length:i]
    prediction = predictor.predict(input_sequence)
    # Ensure each prediction is a 1D array of length output_dim
    prediction = np.asarray(prediction, dtype=float).reshape(output_dim)
    predictions.append(prediction)

nan_pad = np.full((sequence_length, output_dim), np.nan, dtype=float)
predictions = np.asarray(predictions, dtype=float).reshape(-1, output_dim)
predictions = np.vstack([nan_pad, predictions])

# Add predictions to dataframe per output column
for idx, col in enumerate(OUTPUT_COLS):
    df[f'predicted_{col}'] = predictions[:, idx]


state_names = ['angular_vel_z', 'linear_vel_x', 'linear_vel_y']
# state_names = ['linear_vel_x']  # For current model

for state_name in state_names:

    # Backward compatibility / specific columns
    if f'residual_delta_{state_name}_0' in OUTPUT_COLS:
        df[f'predicted_residual_delta_{state_name}_0'] = predictions[:, OUTPUT_COLS.index(f'residual_delta_{state_name}_0')]
        df[f'corrected_delta_{state_name}_0'] = df[f'predicted_delta_{state_name}_0'] + df[f'predicted_residual_delta_{state_name}_0']
        # Integrate over corrected delta linear vel x 0
        df[f'corrected_{state_name}'] =  df[f'{state_name}'] +  0.04 * df[f'corrected_delta_{state_name}_0']
        df[f'corrected_{state_name}'] = df[f'corrected_{state_name}'].shift(1)
        

# Save to new CSV file
output_path = csv_path.replace('.csv', '_with_predictions.csv')
df.to_csv(output_path, index=False)
print(f"Saved predictions to: {output_path}")
# print(f"Total rows: {len(df)}, Predictions: {len([p for p in predictions if not np.isnan(p)])}")



# Plot data
import matplotlib.pyplot as plt

start_idx = 250
end_idx = 500 

df = df[start_idx:end_idx]




for state_name in state_names:
    # Add 2 figues: one for linear_vel_x, one for delta_linear_vel_x_0
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    # Linear Vel X
    axs[0].plot(df[state_name], label=f'Original {state_name}', color='blue')
    axs[0].plot(df[f'predicted_{state_name}_0'], label=f'ODE {state_name}', color='green')
    axs[0].plot(df[f'corrected_{state_name}'], label=f'Corrected {state_name}', color='orange')
    axs[0].set_title(f'{state_name} Comparison')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel(state_name.replace('_', ' ').title())
    axs[0].legend()
    # Delta {state} 0
    axs[1].plot(df[f'delta_{state_name}'], label=f'True Delta {state_name} 0', color='black')
    axs[1].plot(df[f'predicted_delta_{state_name}_0'], label=f'ODE Delta {state_name} 0', color='green')
    axs[1].plot(df[f'residual_delta_{state_name}_0'], label=f'Residual Delta {state_name} 0', color='red')
    axs[1].plot(df[f'predicted_residual_delta_{state_name}_0'], label=f'Predicted Residual Delta {state_name} 0', color='orange')
    axs[1].plot(df[f'corrected_delta_{state_name}_0'], label=f'Corrected Delta {state_name} 0', color='purple')
    axs[1].set_title(f'Delta {state_name} 0 Comparison')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel(f'Delta {state_name} 0')
    axs[1].legend() 

    plt.tight_layout()
    plt.savefig(output_path.replace('.csv', f'_{state_name}.png'))