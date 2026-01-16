#!/usr/bin/env python3
"""
Offline batch plotting for state comparison visualization.
Generates all plots for all states without UI (works on SSH without X server).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

# Import common utilities
from visualization_common import (
    VisualizationCommon, AVAILABLE_MODELS,
    STEERING_CONTROL_COLUMN, ACCELERATION_CONTROL_COLUMN
)

def load_config(config_path):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return {}

def plot_delta_prediction_with_gradient(ax, time_data, delta_data, label=None):
    """Plot delta prediction with gradient."""
    try:
        from matplotlib.colors import LinearSegmentedColormap
        colors = ["#5200F5", "#FF00BF", "#FF0000"]
        alpha_base = 0.4
        cmap = LinearSegmentedColormap.from_list("gradient", colors, N=100)
    except ImportError:
        start_color = np.array([1.0, 0.4, 0.4])
        end_color = np.array([0.5, 0.0, 0.0])
        alpha_base = 0.8
        cmap = None
    
    for i in range(len(delta_data) - 1):
        color_position = i / max(1, len(delta_data) - 1)
        if cmap is not None:
            color = cmap(color_position)
        else:
            color = start_color * (1 - color_position) + end_color * color_position
        alpha = alpha_base * (1.0 - 0.3 * color_position)
        ax.plot(time_data[i:i+2], delta_data[i:i+2], 'o', color=color, markersize=3, 
               alpha=alpha, label=label if i == 0 else None)

def generate_all_plots(common, comparison_data_dict, start_index, end_index, output_dir, plot_deltas=True, plot_controls=True):
    """Generate and save all plots for all states.
    
    Args:
        common: VisualizationCommon instance
        comparison_data_dict: Dictionary of comparison data
        start_index: Start index for plotting
        end_index: End index for plotting
        output_dir: Directory to save plots
        plot_deltas: If True, include delta plots
        plot_controls: If True, include control plots
    """
    data = common.data
    time_column = common.time_column
    
    # Get available states
    available_states = [col for col in common.state_columns if col in data.columns]
    
    if not available_states:
        print("Error: No valid state columns found in CSV.")
        return
    
    print(f"Generating plots for {len(available_states)} states...")
    print(f"  Plot deltas: {plot_deltas}, Plot controls: {plot_controls}")
    
    # Calculate number of subplots
    num_subplots = 1
    if plot_deltas:
        num_subplots += 1
    if plot_controls:
        num_subplots += 1
    
    # Dictionary to store error metrics for CSV export
    error_metrics_dict = {}
    
    for state_idx, state_name in enumerate(available_states):
        print(f"  Plotting {state_name} ({state_idx+1}/{len(available_states)})...")
        
        # Create figure with dynamic subplots
        fig = plt.figure(figsize=(16, 4 * num_subplots))
        gs = fig.add_gridspec(num_subplots, 1, hspace=0.3)
        
        # Main plot
        subplot_idx = 0
        ax_main = fig.add_subplot(gs[subplot_idx, 0])
        
        # Plot ground truth
        start_idx = start_index
        end_idx = end_index if end_index is not None else len(data)
        
        if time_column in data.columns:
            time_data = data[time_column].iloc[start_idx:end_idx]
        else:
            time_data = np.arange(start_idx, end_idx)
        
        state_data = data[state_name].iloc[start_idx:end_idx]
        ax_main.plot(time_data, state_data, 'k-', label='Ground Truth', linewidth=2)
        
        # Plot all predictions
        for comp_count, (comp_start_idx, comparison_data) in enumerate(comparison_data_dict.items()):
            if comp_start_idx < start_idx or comp_start_idx >= end_idx:
                continue
            
            if state_name in comparison_data:
                full_horizon = len(comparison_data[state_name])
                
                # Generate time data
                if time_column in data.columns:
                    if comp_start_idx + full_horizon <= len(data):
                        comp_time = data[time_column].iloc[comp_start_idx:comp_start_idx + full_horizon]
                    else:
                        available_time = data[time_column].iloc[comp_start_idx:]
                        dt = common.get_timestep()
                        missing_steps = full_horizon - len(available_time)
                        if len(available_time) > 0:
                            last_time = available_time.iloc[-1]
                            extra_time = np.arange(1, missing_steps + 1) * dt + last_time
                            comp_time = np.concatenate([available_time.to_numpy(), extra_time])
                        else:
                            comp_time = np.arange(comp_start_idx, comp_start_idx + full_horizon) * dt
                else:
                    dt = common.get_timestep()
                    comp_time = np.arange(comp_start_idx, comp_start_idx + full_horizon) * dt
                
                full_prediction = np.array(comparison_data[state_name])
                common.plot_prediction_with_gradient(
                    ax_main, comp_time, full_prediction, full_horizon,
                    label='Model Predictions' if comp_count == 0 else None
                )
        
        # Calculate error metrics by collecting all predictions and ground truth
        all_predictions = []
        all_ground_truth = []
        
        for comp_start_idx, comparison_data in comparison_data_dict.items():
            if comp_start_idx < start_idx or comp_start_idx >= end_idx:
                continue
            
            if state_name in comparison_data:
                prediction = np.array(comparison_data[state_name])
                horizon = len(prediction)
                
                # Get corresponding ground truth
                pred_end_idx = min(comp_start_idx + horizon, len(data))
                if pred_end_idx > comp_start_idx:
                    gt_slice = data[state_name].iloc[comp_start_idx:pred_end_idx].values
                    
                    # Ensure same length
                    min_len = min(len(prediction), len(gt_slice))
                    all_predictions.extend(prediction[:min_len])
                    all_ground_truth.extend(gt_slice[:min_len])
        
        # Calculate error metrics
        if len(all_predictions) > 0 and len(all_ground_truth) > 0:
            gt_array = np.array(all_ground_truth)
            pred_array = np.array(all_predictions)
            errors = pred_array - gt_array
            abs_errors = np.abs(errors)
            
            mean_abs_error = np.mean(abs_errors)
            max_error = np.max(abs_errors)
            
            # Store metrics for CSV
            error_metrics_dict[state_name] = {
                'mean_abs_error': mean_abs_error,
                'max_error': max_error
            }
            
            # Display metrics on plot
            error_text = f'Mean |Error|: {mean_abs_error:.6f}\nMax |Error|: {max_error:.6f}'
            ax_main.text(0.02, 0.98, error_text, transform=ax_main.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=10)
        
        ax_main.set_xlabel('Time (s)' if time_column in data.columns else 'Step')
        ax_main.set_ylabel(state_name.replace('_', ' ').title())
        ax_main.set_title(f'State Comparison: {state_name}')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        if time_column in data.columns:
            x_min = data[time_column].iloc[start_idx]
            x_max = data[time_column].iloc[end_idx - 1] if end_idx <= len(data) else data[time_column].iloc[-1]
            ax_main.set_xlim(x_min, x_max)
        else:
            ax_main.set_xlim(start_idx, end_idx - 1)
        
        # Delta plot (if enabled)
        if plot_deltas:
            subplot_idx += 1
            ax_delta = fig.add_subplot(gs[subplot_idx, 0], sharex=ax_main)
            state_data_values = data[state_name].iloc[start_idx:end_idx].values
            if len(state_data_values) > 1:
                if state_name == 'pose_theta':
                    delta_raw = np.diff(state_data_values)
                    delta_gt = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
                else:
                    delta_gt = np.diff(state_data_values)
                delta_time = time_data.iloc[:-1].values if hasattr(time_data, 'iloc') else time_data[:-1]
                ax_delta.plot(delta_time, delta_gt, 'k-', label='Ground Truth Delta', linewidth=2)
                
                # Plot delta for predictions
                for comp_count, (comp_start_idx, comparison_data) in enumerate(comparison_data_dict.items()):
                    if comp_start_idx < start_idx or comp_start_idx >= end_idx or state_name not in comparison_data:
                        continue
                    prediction = np.array(comparison_data[state_name])
                    if len(prediction) > 1:
                        if state_name == 'pose_theta':
                            delta_raw = np.diff(prediction)
                            delta_pred = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
                        else:
                            delta_pred = np.diff(prediction)
                        # Time data for delta
                        if time_column in data.columns:
                            if comp_start_idx + len(prediction) <= len(data):
                                comp_time = data[time_column].iloc[comp_start_idx:comp_start_idx + len(prediction)]
                            else:
                                available_time = data[time_column].iloc[comp_start_idx:]
                                dt = common.get_timestep()
                                missing_steps = len(prediction) - len(available_time)
                                if len(available_time) > 0:
                                    last_time = available_time.iloc[-1]
                                    extra_time = np.arange(1, missing_steps + 1) * dt + last_time
                                    comp_time = np.concatenate([available_time.to_numpy(), extra_time])
                                else:
                                    comp_time = np.arange(comp_start_idx, comp_start_idx + len(prediction)) * dt
                        else:
                            dt = common.get_timestep()
                            comp_time = np.arange(comp_start_idx, comp_start_idx + len(prediction)) * dt
                        delta_pred_time = comp_time.iloc[:-1].values if hasattr(comp_time, 'iloc') else comp_time[:-1]
                        plot_delta_prediction_with_gradient(
                            ax_delta, delta_pred_time, delta_pred,
                            label='Model Prediction Delta' if comp_count == 0 else None
                        )
            
                ax_delta.set_xlabel('Time (s)' if time_column in data.columns else 'Step')
                ax_delta.set_ylabel(f'Δ {state_name.replace("_", " ").title()}')
                ax_delta.set_title(f'Delta State: {state_name}')
                ax_delta.legend()
                ax_delta.grid(True, alpha=0.3)
        
        # Controls plot (if enabled)
        if plot_controls:
            subplot_idx += 1
            ax_controls = fig.add_subplot(gs[subplot_idx, 0], sharex=ax_main)
            if STEERING_CONTROL_COLUMN in data.columns:
                steering_data = data[STEERING_CONTROL_COLUMN].iloc[start_idx:end_idx]
                ax_controls.plot(time_data, steering_data, 'r-', label='Steering Angle', linewidth=2)
                ax_controls.set_ylabel('Steering Angle (rad)', color='r')
                ax_controls.tick_params(axis='y', labelcolor='r')
                ax_controls.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1)
            
            if ACCELERATION_CONTROL_COLUMN in data.columns:
                ax_controls2 = ax_controls.twinx()
                accel_data = data[ACCELERATION_CONTROL_COLUMN].iloc[start_idx:end_idx]
                ax_controls2.plot(time_data, accel_data, 'b-', label='Acceleration', linewidth=2)
                ax_controls2.set_ylabel('Acceleration (m/s²)', color='b')
                ax_controls2.tick_params(axis='y', labelcolor='b')
                ax_controls2.axhline(y=0, color='b', linestyle='--', alpha=0.7, linewidth=1)
                
                # Combined legend
                lines1, labels1 = ax_controls.get_legend_handles_labels()
                lines2, labels2 = ax_controls2.get_legend_handles_labels()
                ax_controls.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax_controls.legend()
            
            ax_controls.set_xlabel('Time (s)' if time_column in data.columns else 'Step')
            ax_controls.set_title('Control Inputs')
            ax_controls.grid(True, alpha=0.3)
        
        # Save figure
        safe_state_name = state_name.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f'{safe_state_name}_comparison.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
    
    # Save error metrics to CSV
    if error_metrics_dict:
        errors_csv_path = os.path.join(output_dir, 'errors.csv')
        error_rows = []
        for state_name in sorted(error_metrics_dict.keys()):
            metrics = error_metrics_dict[state_name]
            error_rows.append({
                'state': state_name,
                'mean_abs_error': metrics['mean_abs_error'],
                'max_error': metrics['max_error']
            })
        
        errors_df = pd.DataFrame(error_rows)
        errors_df.to_csv(errors_csv_path, index=False)
        print(f"Error metrics saved to: {errors_csv_path}")
        print("\nError Summary:")
        print(errors_df.to_string(index=False))
    
    print(f"All plots saved to: {output_dir}")

def main():
    """Main entry point for offline plotting."""
    # Get config file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'visualization_config.json')
    
    # Load config
    config = load_config(config_path)
    
    # Check CSV file path
    csv_path = config.get('csv_file_path', '')
    if not csv_path or not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        print("Please set 'csv_file_path' in visualization_config.json")
        return
    
    # Initialize common utilities
    common = VisualizationCommon()
    
    # Load CSV data
    print(f"Loading CSV: {csv_path}")
    common.data = pd.read_csv(csv_path, comment='#')
    common.csv_file_path = csv_path
    
    # Get model and parameters from config
    default_model = config.get('default_car_model', '')
    default_params = config.get('default_car_parameters', '')
    
    # Find model key
    model_name_key = None
    for key, value in AVAILABLE_MODELS.items():
        if value == default_model:
            model_name_key = key
            break
    if model_name_key is None:
        model_name_key = list(AVAILABLE_MODELS.keys())[0]
    
    # Find parameter file
    if default_params not in common.available_car_params.values():
        default_params = list(common.available_car_params.values())[0]
    
    print(f"Using model: {AVAILABLE_MODELS[model_name_key]}, parameters: {default_params}")
    
    # Get settings from config
    start_index = config.get('start_index', 0)
    end_index = config.get('end_index', None)
    horizon = config.get('horizon_steps', 50)
    steering_delay = config.get('steering_delay_steps', 2)
    acceleration_delay = config.get('acceleration_delay_steps', 2)
    plot_deltas = config.get('plot_deltas', True)  # Default to True for backward compatibility
    plot_controls = config.get('plot_controls', True)  # Default to True for backward compatibility
    
    # Determine range for comparisons
    effective_start = start_index if start_index is not None else 0
    effective_end = end_index if end_index is not None else len(common.data)
    max_start_idx = effective_end - horizon
    
    if max_start_idx <= effective_start:
        print(f"Error: Horizon ({horizon}) is larger than available data range.")
        return
    
    # Compute predictions for all start indices
    range_size = max_start_idx - effective_start
    step_size = max(1, range_size // 100)  # Compute ~100 predictions max
    start_indices = list(range(effective_start, max_start_idx, step_size))
    
    print(f"Computing {len(start_indices)} predictions...")
    comparison_data_dict = {}
    
    # Initialize car models
    common.reload_car_models()
    
    # Run comparisons
    successful = 0
    for i, start_idx in enumerate(start_indices):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(start_indices)}")
        
        result = common.run_single_comparison(
            start_idx, model_name_key, default_params, 
            horizon, steering_delay, acceleration_delay
        )
        if result is not None:
            comparison_data_dict[start_idx] = result
            successful += 1
    
    print(f"Completed {successful}/{len(start_indices)} successful predictions.")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(csv_path), 'plots_offline')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save all plots
    generate_all_plots(common, comparison_data_dict, start_index, end_index, output_dir, 
                      plot_deltas=plot_deltas, plot_controls=plot_controls)
    
    print("Done!")

if __name__ == "__main__":
    main()
