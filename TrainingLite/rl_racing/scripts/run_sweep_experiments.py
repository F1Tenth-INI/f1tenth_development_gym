#!/usr/bin/env python3
"""
Run a series of racing experiments across all models with a matching prefix.

Usage:
    python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "Sweep_rank_Ex1_A0.0_B0.4_R0.0"
    python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "2002_from_Ex1" --max-length 10000
    python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "Sweep_" --verbose
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add root dir to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, root_dir)

# Must be set BEFORE importing numpy, torch, etc.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from utilities.Settings import Settings
from run.run_simulation import RacingSimulation
from utilities.Exceptions import CarCrashException
from utilities.car_system import CarSystem


class SweepExperimentRunner:
    """Run racing experiments across multiple trained models."""
    
    def __init__(self, prefix: str, max_length: int = 8000, verbose: bool = False):
        """
        Initialize the sweep runner.
        
        Args:
            prefix: Model name prefix to filter (e.g., "Sweep_rank_Ex1_A0.0_B0.4_R0.0")
            max_length: Maximum experiment length in sim timesteps (evaluates for consistent time, not lap count)
            verbose: Print detailed output
        """
        self.prefix = prefix
        self.max_length = max_length
        self.verbose = verbose
        
        self.models_dir = Path(root_dir) / "TrainingLite" / "rl_racing" / "models"
        self.results_dir = Path(root_dir) / "sweep_experiment_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.results_csv = self.results_dir / f"sweep_{prefix}_results.csv"
        self.detailed_log = self.results_dir / f"sweep_{prefix}_detailed.txt"
        
    def find_models_by_prefix(self) -> List[str]:
        """Find all model directories matching the prefix."""
        if not self.models_dir.exists():
            print(f"Error: Models directory not found: {self.models_dir}")
            return []
        
        matching_models = []
        for model_dir in sorted(self.models_dir.iterdir()):
            if model_dir.is_dir() and model_dir.name.startswith(self.prefix):
                matching_models.append(model_dir.name)
        
        return matching_models
    
    def run_experiment_on_model(self, model_name: str) -> Dict[str, any]:
        """
        Run a racing experiment on a single model.
        
        Returns:
            Dictionary with experiment results (lap_times, avg_speed, crashes, etc.)
        """
        result = {
            'model_name': model_name,
            'status': 'unknown',
            'error_message': None,
            'num_laps_completed': 0,
            'num_laps_attempted': self.max_length,
            'lap_times': [],
            'avg_lap_time': None,
            'min_lap_time': None,
            'max_lap_time': None,
            'std_lap_time': None,
            'lap_time_range': None,
            'lap_consistency': None,
            'avg_speed': None,
            'max_speed': None,
            'min_speed': None,
            'std_speed': None,
            'avg_distance_to_raceline': None,
            'max_distance_to_raceline': None,
            'std_distance_to_raceline': None,
            'avg_steering_angle': None,
            'max_steering_angle': None,
            'steering_smoothness': None,
            'avg_acceleration': None,
            'max_acceleration': None,
            'acceleration_smoothness': None,
            'total_sim_time': 0.0,
            'avg_time_per_lap': None,
            'crash_occurred': False,
            'num_crashes': 0,
            'off_track_events': 0,
            'total_reward': None,
            'mean_step_reward': None,
            'num_reward_samples': 0,
            'max_lateral_deviation': None,
            'num_car_states_recorded': 0,
            'num_control_inputs': 0,
        }

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Running experiment on model: {model_name}")
            print(f"{'='*80}")
        else:
            print(f"Running: {model_name}...", end=" ", flush=True)

        Settings.SAC_INFERENCE_MODEL_NAME = model_name
        Settings.SIMULATION_LENGTH = self.max_length
        Settings.MAX_EPISODE_LENGTH = self.max_length
        Settings.NUMBER_OF_EXPERIMENTS = 1
        Settings.RESET_ON_DONE = True

        simulation = RacingSimulation()
        lap_times_buffer: List[float] = []
        reward_buffer: List[float] = []
        crash_count = 0  # Count all crashes: collisions + off-track terminations

        original_lap_complete_cb = CarSystem.lap_complete_cb
        original_handle_done = simulation.handle_done
        original_sim_on_step_end = simulation.on_step_end

        def patched_lap_complete_cb(driver_self, lap_time, mean_distance, std_distance, max_distance):
            lap_times_buffer.append(float(lap_time))
            return original_lap_complete_cb(driver_self, lap_time, mean_distance, std_distance, max_distance)

        def patched_handle_done():
            nonlocal crash_count
            if simulation.drivers and len(simulation.drivers) > 0:
                driver_obs = getattr(simulation.drivers[0], 'obs', None)
                if isinstance(driver_obs, dict):
                    # Count truncated events as crashes (includes leave_bounds, collision, spinning, stuck)
                    if driver_obs.get('truncated', False):
                        crash_count += 1
            return original_handle_done()

        def patched_on_step_end():
            # Let the simulation update driver.obs first, then sample reward.
            original_sim_on_step_end()
            if simulation.episode_index <= 0:
                return
            if simulation.drivers and len(simulation.drivers) > 0:
                driver_obs = getattr(simulation.drivers[0], 'obs', None)
                if isinstance(driver_obs, dict):
                    reward = driver_obs.get('reward')
                    if reward is not None:
                        try:
                            reward_buffer.append(float(reward))
                        except (TypeError, ValueError):
                            pass

        try:
            CarSystem.lap_complete_cb = patched_lap_complete_cb
            simulation.handle_done = patched_handle_done
            simulation.on_step_end = patched_on_step_end
            try:
                simulation.run_experiments()
            except Exception as e:
                if "Crash repetition disabled" not in str(e):
                    raise
        except Exception as e:
            result['status'] = 'failed'
            result['error_message'] = str(e)
            if self.verbose:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"✗ (Error: {type(e).__name__}: {e})")
                import traceback
                traceback.print_exc()
        finally:
            CarSystem.lap_complete_cb = original_lap_complete_cb
            simulation.handle_done = original_handle_done
            simulation.on_step_end = original_sim_on_step_end

        if result['status'] != 'failed' and simulation.drivers and len(simulation.drivers) > 0:
            main_driver = simulation.drivers[0]
            result['num_crashes'] = crash_count
            result['crash_occurred'] = result['num_crashes'] > 0

            if reward_buffer:
                rewards_array = np.array(reward_buffer, dtype=np.float64)
                result['num_reward_samples'] = len(reward_buffer)
                result['total_reward'] = float(np.sum(rewards_array))
                result['mean_step_reward'] = float(np.mean(rewards_array))

            if lap_times_buffer:
                result['lap_times'] = list(lap_times_buffer)
                result['num_laps_completed'] = len(lap_times_buffer)
                lap_times_array = np.array(lap_times_buffer)
                result['avg_lap_time'] = float(np.mean(lap_times_array))
                result['min_lap_time'] = float(np.min(lap_times_array))
                result['max_lap_time'] = float(np.max(lap_times_array))
                result['std_lap_time'] = float(np.std(lap_times_array))
                result['lap_time_range'] = result['max_lap_time'] - result['min_lap_time']
                if result['avg_lap_time'] > 0:
                    result['lap_consistency'] = 1.0 - (result['std_lap_time'] / result['avg_lap_time'])
                    result['avg_time_per_lap'] = simulation.sim_time / result['num_laps_completed']

            if hasattr(main_driver, 'car_state_history') and len(main_driver.car_state_history) > 0:
                car_states = np.array(main_driver.car_state_history)
                result['num_car_states_recorded'] = len(car_states)
                try:
                    from utilities.state_utilities import LINEAR_VEL_X_IDX
                    if car_states.shape[1] > LINEAR_VEL_X_IDX:
                        velocities = car_states[:, LINEAR_VEL_X_IDX]
                        result['avg_speed'] = float(np.mean(velocities))
                        result['max_speed'] = float(np.max(velocities))
                        result['min_speed'] = float(np.min(velocities))
                        result['std_speed'] = float(np.std(velocities))
                except Exception:
                    pass

            if hasattr(main_driver, 'lap_analyzer') and hasattr(main_driver.lap_analyzer, 'distance_log'):
                if len(main_driver.lap_analyzer.distance_log) > 0:
                    distances = np.array(main_driver.lap_analyzer.distance_log)
                    result['avg_distance_to_raceline'] = float(np.mean(distances))
                    result['max_distance_to_raceline'] = float(np.max(distances))
                    result['std_distance_to_raceline'] = float(np.std(distances))

            control_array = None
            if hasattr(main_driver, 'control_history') and len(main_driver.control_history) > 0:
                # Expected shape for driver history: [N, 2]
                control_array = np.array(main_driver.control_history)
            elif hasattr(simulation, 'control_history') and len(simulation.control_history) > 0:
                # Fallback shape for simulation history: [N, num_drivers, 2]
                simulation_control_array = np.array(simulation.control_history)
                if simulation_control_array.ndim == 3 and simulation_control_array.shape[1] > 0:
                    control_array = simulation_control_array[:, 0, :]

            if control_array is not None and control_array.size > 0 and control_array.shape[-1] >= 2:
                result['num_control_inputs'] = len(control_array)
                steering = control_array[:, 0]
                acceleration = control_array[:, 1]

                result['avg_steering_angle'] = float(np.mean(np.abs(steering)))
                result['max_steering_angle'] = float(np.max(np.abs(steering)))
                if len(steering) > 1:
                    steering_diffs = np.diff(steering)
                    result['steering_smoothness'] = float(np.std(steering_diffs))

                result['avg_acceleration'] = float(np.mean(np.abs(acceleration)))
                result['max_acceleration'] = float(np.max(np.abs(acceleration)))
                if len(acceleration) > 1:
                    accel_diffs = np.diff(acceleration)
                    result['acceleration_smoothness'] = float(np.std(accel_diffs))

            result['total_sim_time'] = simulation.sim_time

        if result['status'] != 'failed':
            result['status'] = 'completed'
            avg_str = f"{result['avg_lap_time']:.2f}s" if result['avg_lap_time'] is not None else "N/A"
            if self.verbose:
                print(f"Result: {result['num_laps_completed']} laps completed")
                if result['lap_times']:
                    print(f"  Lap times (s): {[f'{t:.2f}' for t in result['lap_times']]}")
                print(f"  Average lap time: {avg_str}")
            else:
                print(f"✓ ({result['num_laps_completed']} laps, avg: {avg_str})")

        return result
    
    def save_results(self, all_results: List[Dict]) -> None:
        """Save results to CSV and log file."""
        if not all_results:
            print("No results to save")
            return
        
        # Write CSV with all metrics
        csv_headers = [
            'model_name', 'status', 'num_laps_completed', 'num_laps_attempted',
            'avg_lap_time', 'min_lap_time', 'max_lap_time', 'std_lap_time', 'lap_time_range', 'lap_consistency',
            'avg_speed', 'max_speed', 'min_speed', 'std_speed',
            'avg_distance_to_raceline', 'max_distance_to_raceline', 'std_distance_to_raceline',
            'avg_steering_angle', 'max_steering_angle', 'steering_smoothness',
            'avg_acceleration', 'max_acceleration', 'acceleration_smoothness',
            'total_sim_time', 'avg_time_per_lap',
            'crash_occurred', 'num_crashes',
            'num_car_states_recorded', 'num_control_inputs',
            'error_message'
        ]
        
        with open(self.results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            
            for result in all_results:
                # Helper to format floats
                fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else 'N/A'
                
                row = {
                    'model_name': result['model_name'],
                    'status': result['status'],
                    'num_laps_completed': result['num_laps_completed'],
                    'num_laps_attempted': result['num_laps_attempted'],
                    'avg_lap_time': fmt(result['avg_lap_time']),
                    'min_lap_time': fmt(result['min_lap_time']),
                    'max_lap_time': fmt(result['max_lap_time']),
                    'std_lap_time': fmt(result['std_lap_time']),
                    'lap_time_range': fmt(result['lap_time_range']),
                    'lap_consistency': fmt(result['lap_consistency']),
                    'avg_speed': fmt(result['avg_speed']),
                    'max_speed': fmt(result['max_speed']),
                    'min_speed': fmt(result['min_speed']),
                    'std_speed': fmt(result['std_speed']),
                    'avg_distance_to_raceline': fmt(result['avg_distance_to_raceline']),
                    'max_distance_to_raceline': fmt(result['max_distance_to_raceline']),
                    'std_distance_to_raceline': fmt(result['std_distance_to_raceline']),
                    'avg_steering_angle': fmt(result['avg_steering_angle']),
                    'max_steering_angle': fmt(result['max_steering_angle']),
                    'steering_smoothness': fmt(result['steering_smoothness']),
                    'avg_acceleration': fmt(result['avg_acceleration']),
                    'max_acceleration': fmt(result['max_acceleration']),
                    'acceleration_smoothness': fmt(result['acceleration_smoothness']),
                    'total_sim_time': fmt(result['total_sim_time']),
                    'avg_time_per_lap': fmt(result['avg_time_per_lap']),
                    'crash_occurred': result['crash_occurred'],
                    'num_crashes': result['num_crashes'],
                    'num_car_states_recorded': result['num_car_states_recorded'],
                    'num_control_inputs': result['num_control_inputs'],
                    'error_message': result['error_message'] or '',
                }
                writer.writerow(row)
        
        print(f"\nResults saved to: {self.results_csv}")
        
        # Write detailed log
        with open(self.detailed_log, 'w') as f:
            f.write(f"Sweep Experiment Results for prefix: {self.prefix}\n")
            f.write(f"Maximum simulation length per model: {self.max_length} timesteps\n")
            f.write(f"{'='*80}\n\n")
            
            for result in all_results:
                f.write(f"Model: {result['model_name']}\n")
                f.write(f"  Status: {result['status']}\n")
                f.write(f"\n  === LAP PERFORMANCE ===")
                f.write(f"\n  Laps Completed: {result['num_laps_completed']}/{result['num_laps_attempted']}\n")
                if result['lap_times']:
                    f.write(f"  Lap Times (s): {[f'{t:.2f}' for t in result['lap_times']]}\n")
                    f.write(f"  Avg Lap Time: {result['avg_lap_time']:.4f}s\n")
                    f.write(f"  Min Lap Time: {result['min_lap_time']:.4f}s\n")
                    f.write(f"  Max Lap Time: {result['max_lap_time']:.4f}s\n")
                    f.write(f"  Std Dev: {result['std_lap_time']:.4f}s\n")
                    f.write(f"  Range: {result['lap_time_range']:.4f}s\n")
                    if result['lap_consistency'] is not None:
                        f.write(f"  Consistency (0-1): {result['lap_consistency']:.4f}\n")
                
                f.write(f"\n  === SPEED ===")
                if result['avg_speed'] is not None:
                    f.write(f"\n  Avg Speed: {result['avg_speed']:.4f} m/s\n")
                    f.write(f"  Max Speed: {result['max_speed']:.4f} m/s\n")
                    f.write(f"  Min Speed: {result['min_speed']:.4f} m/s\n")
                    f.write(f"  Std Dev: {result['std_speed']:.4f} m/s\n")
                
                f.write(f"\n  === RACELINE TRACKING ===")
                if result['avg_distance_to_raceline'] is not None:
                    f.write(f"\n  Avg Distance to Line: {result['avg_distance_to_raceline']:.4f} m\n")
                    f.write(f"  Max Distance to Line: {result['max_distance_to_raceline']:.4f} m\n")
                    f.write(f"  Std Dev: {result['std_distance_to_raceline']:.4f} m\n")
                
                f.write(f"\n  === CONTROL BEHAVIOR ===")
                if result['avg_steering_angle'] is not None:
                    f.write(f"\n  Avg Steering Angle: {result['avg_steering_angle']:.4f}\n")
                    f.write(f"  Max Steering Angle: {result['max_steering_angle']:.4f}\n")
                    if result['steering_smoothness'] is not None:
                        f.write(f"  Steering Smoothness (std of diffs): {result['steering_smoothness']:.4f}\n")
                if result['avg_acceleration'] is not None:
                    f.write(f"  Avg Acceleration: {result['avg_acceleration']:.4f}\n")
                    f.write(f"  Max Acceleration: {result['max_acceleration']:.4f}\n")
                    if result['acceleration_smoothness'] is not None:
                        f.write(f"  Acceleration Smoothness (std of diffs): {result['acceleration_smoothness']:.4f}\n")
                
                f.write(f"\n  === TIMING ===")
                f.write(f"\n  Total Sim Time: {result['total_sim_time']:.4f}s\n")
                if result['avg_time_per_lap'] is not None:
                    f.write(f"  Avg Time Per Lap: {result['avg_time_per_lap']:.4f}s\n")
                
                f.write(f"\n  === SAFETY ===")
                f.write(f"\n  Crash: {'YES' if result['crash_occurred'] else 'NO'}\n")
                f.write(f"  Num Crashes (collisions + off-track): {result['num_crashes']}\n")
                
                f.write(f"\n  === DATA QUALITY ===")
                f.write(f"\n  Car States Recorded: {result['num_car_states_recorded']}\n")
                f.write(f"  Control Inputs: {result['num_control_inputs']}\n")
                
                if result['error_message']:
                    f.write(f"\n  === ERROR ===")
                    f.write(f"\n  {result['error_message']}\n")
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Detailed log saved to: {self.detailed_log}")
    
    def run_all(self) -> None:
        """Run experiments on all matching models."""
        models = self.find_models_by_prefix()
        
        if not models:
            print(f"No models found matching prefix: {self.prefix}")
            return
        
        print(f"Found {len(models)} models matching prefix: {self.prefix}")
        print(f"Will run each model for max {self.max_length} sim timesteps\n")
        
        all_results = []
        
        for i, model_name in enumerate(models, 1):
            print(f"[{i}/{len(models)}]", end=" ")
            result = self.run_experiment_on_model(model_name)
            all_results.append(result)
        
        # Print summary
        print(f"\n{'='*80}")
        print("SWEEP EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        successful = sum(1 for r in all_results if r['status'] == 'completed')
        print(f"Successful experiments: {successful}/{len(all_results)}")
        
        if successful > 0:
            avg_times = [r['avg_lap_time'] for r in all_results if r['avg_lap_time'] is not None]
            if avg_times:
                print(f"Best average lap time: {min(avg_times):.4f}s")
                print(f"Worst average lap time: {max(avg_times):.4f}s")
                print(f"Overall average: {np.mean(avg_times):.4f}s")
        
        # Save results
        self.save_results(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Run racing experiments across models with matching prefix"
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Model name prefix to filter (e.g., 'Sweep_rank_Ex1_A0.0_B0.4_R0.0')"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8000,
        help="Maximum experiment length in sim timesteps (default: 8000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each model"
    )
    
    args = parser.parse_args()
    
    runner = SweepExperimentRunner(
        prefix=args.prefix,
        max_length=args.max_length,
        verbose=args.verbose
    )
    runner.run_all()


if __name__ == "__main__":
    main()
