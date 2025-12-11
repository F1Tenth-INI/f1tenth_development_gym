import os
import pandas as pd
import numpy as np
from predictor import Predictor

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'models')
csv_path = "/home/florian/Documents/INI/f1tenth_development_gym/TrainingLite/dynamic_residual_jax/training_data/processed_data.csv"

# Load predictor
predictor = Predictor(model_dir)

# Load CSV and extract input columns
df = pd.read_csv(csv_path, comment='#')
input_cols = ['linear_vel_x', 'angular_control_executed', 'translational_control_executed']

# Generate predictions for all rows (starting from row 10)
sequence_length = 10
predictions = []

print(f"Generating predictions for {len(df) - sequence_length} rows...")
for i in range(sequence_length, len(df)):
    input_sequence = df[input_cols].values[i - sequence_length:i]
    prediction = predictor.predict(input_sequence)
    # Convert to scalar value
    prediction = float(np.array(prediction).item())
    predictions.append(prediction)

# Add NaN for first 10 rows (no prediction possible)
predictions = [np.nan] * sequence_length + predictions

# Add predictions to dataframe
df['predicted_residual_delta_linear_vel_x_0'] = predictions
df['corrected_delta_linear_vel_x_0'] = df['predicted_delta_linear_vel_x_0'] + df['predicted_residual_delta_linear_vel_x_0']
# Integrate over corrected delta linear vel x 0
df['corrected_linear_vel_x'] =  df['linear_vel_x'] +  0.04 * df['corrected_delta_linear_vel_x_0']
df['corrected_linear_vel_x'] = df['corrected_linear_vel_x'].shift(1)


# Save to new CSV file
output_path = csv_path.replace('.csv', '_with_predictions.csv')
df.to_csv(output_path, index=False)
print(f"Saved predictions to: {output_path}")
print(f"Total rows: {len(df)}, Predictions: {len([p for p in predictions if not np.isnan(p)])}")

