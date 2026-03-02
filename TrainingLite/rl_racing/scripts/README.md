# Sweep Experiment Runner - Usage Guide

This guide explains how to run and analyze racing experiments across multiple trained models with a matching prefix.

## Overview

You have two scripts in `TrainingLite/rl_racing/scripts/`:
1. **`run_sweep_experiments.py`** - Runs racing experiments on all models matching a prefix
2. **`analyze_sweep_results.py`** - Analyzes and visualizes the results

## Quick Start

### Running Experiments

```bash
# Basic usage - run each model for 8000 sim timesteps (default)
python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "Sweep_rank_Ex1_A0.0_B0.4_R0.0"

# Run with longer evaluation time and verbose output
python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "2002_from_Ex1" --max-length 10000 --verbose

# Run experiments for all models starting with "Sweep_" for shorter evaluation
python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "Sweep_" --max-length 5000
```

**Arguments:**
- `--prefix` (required) - Model name prefix to filter (case-sensitive)
- `--max-length` (optional, default: 8000) - Maximum experiment length in simulation timesteps (not lap count, so all models are evaluated for consistent duration)
- `--verbose` (optional) - Print detailed output for each model

### Analyzing Results

```bash
# Analyze results and print summary
python TrainingLite/rl_racing/scripts/analyze_sweep_results.py --prefix "Sweep_rank_Ex1_A0.0_B0.4_R0.0"

# Analyze and generate plots
python TrainingLite/rl_racing/scripts/analyze_sweep_results.py --prefix "Sweep_rank_Ex1_A0.0_B0.4_R0.0" --plot

# Direct path to results file
python TrainingLite/rl_racing/scripts/analyze_sweep_results.py --results-file "sweep_experiment_results/sweep_2002_from_Ex1_results.csv"
```

## Output Files

After running experiments, you'll find results in `sweep_experiment_results/`:

```
sweep_experiment_results/
├── sweep_{prefix}_results.csv          # CSV with all metrics
├── sweep_{prefix}_detailed.txt         # Detailed text log with full stats
├── sweep_{prefix}_analysis.txt         # Top/bottom 10 models rankings
└── sweep_{prefix}_plot.png             # Comprehensive visualizations (if --plot used)
```

### Comprehensive Metrics Collected

The runner collects extensive metrics for each model:

**Basic Information:**
- `model_name` - Name of the model
- `status` - 'completed' or 'failed'
- `error_message` - Any error that occurred during execution

**Lap Performance:**
- `num_laps_completed` - Number of laps successfully completed
- `num_laps_attempted` - Number of laps requested
- `lap_times` - List of individual lap times (seconds)
- `avg_lap_time` - Average lap time across all laps
- `min_lap_time / max_lap_time` - Fastest and slowest lap
- `std_lap_time` - Standard deviation of lap times
- `lap_time_range` - Difference between max and min
- `lap_consistency` - Score 0-1 (higher = more consistent), calculated as 1 - (std_dev / mean)
- `avg_time_per_lap` - Average total time per lap (simulation time / laps)

**Speed Statistics:**
- `avg_speed` - Average speed during run (m/s)
- `max_speed` - Peak speed achieved (m/s)
- `min_speed` - Lowest speed during run (m/s)
- `std_speed` - Standard deviation of speeds

**Raceline Tracking:**
- `avg_distance_to_raceline` - Average deviation from ideal line (m)
- `max_distance_to_raceline` - Maximum deviation from ideal line (m)
- `std_distance_to_raceline` - Standard deviation of deviations

**Control Behavior:**
- `avg_steering_angle` - Average absolute steering input
- `max_steering_angle` - Maximum steering input
- `steering_smoothness` - Standard deviation of steering changes (lower = smoother)
- `avg_acceleration` - Average absolute acceleration
- `max_acceleration` - Maximum acceleration
- `acceleration_smoothness` - Standard deviation of acceleration changes (lower = smoother)

**Safety & Timing:**
- `crash_occurred` - Boolean, indicates if a car crash was detected
- `num_crashes` - Count of crashes during run
- `total_sim_time` - Total simulation time executed (seconds)

**Data Quality:**
- `num_car_states_recorded` - Number of state samples recorded
- `num_control_inputs` - Number of control decisions made

### CSV Format

All metrics above are exported to the CSV file for easy analysis in Excel or Python. Values are formatted to 4 decimal places for numeric data.

## Example Workflow

```bash
# 1. Run experiments on your sweep
conda activate f1t
python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "Sweep_rank_Ex1_A0.0_B0.4_" --verbose

# 2. Analyze results with comprehensive stats and plots
pip install pandas matplotlib  # if not already installed
python TrainingLite/rl_racing/scripts/analyze_sweep_results.py --prefix "Sweep_rank_Ex1_A0.0_B0.4_" --plot

# 3. Review results
# - Check CSV: sweep_experiment_results/sweep_Sweep_rank_Ex1_A0.0_B0.4__results.csv
# - Read summary: sweep_experiment_results/sweep_Sweep_rank_Ex1_A0.0_B0.4__analysis.txt
# - View plots: sweep_experiment_results/sweep_Sweep_rank_Ex1_A0.0_B0.4__plot.png

# 4. Find best/worst models in the analysis text file
cat sweep_experiment_results/sweep_Sweep_rank_Ex1_A0.0_B0.4__analysis.txt
```

The analyzer generates:
- **Summary statistics** for all metrics (speed, lap consistency, raceline tracking, etc.)
- **Top 10 models** ranked by average lap time with speed and consistency info
- **Comprehensive plots** including lap times, speed, raceline deviation, and consistency scores
- **Detailed analysis text** with full breakdown of performance across all dimensions

**Why timestep-based evaluation?**
Using `--max-length` (sim timesteps) instead of lap count ensures all models are evaluated fairly:
- Models that crash quickly won't artificially skew results
- Consistent evaluation time across all models
- Faster, more powerful models complete more laps in the same time window

## Configuration Notes

The scripts automatically configure:
- **Model Loading**: Sets `Settings.SAC_INFERENCE_MODEL_NAME` for each model
- **Experiment Duration**: Configured via `--max-length` in sim timesteps (default: 8000)
- **Crash Behavior**: Disables automatic restart on crash for consistency (`RESET_ON_DONE = False`)

If you need to customize other Settings (map, obstacles, rendering, etc.), you can edit `utilities/Settings.py` before running.

## Troubleshooting

**Models not found:**
- Check that the prefix matches exactly (case-sensitive)
- Verify models exist in `TrainingLite/rl_racing/models/`

**CSV with N/A values:**
- Could indicate a crash or error during the run
- Check the error_message column for details

**Results not saving:**
- Ensure `sweep_experiment_results/` directory exists and is writable
- Check file permissions

**Plotting errors:**
- Install pandas and matplotlib: `pip install pandas matplotlib`

## Advanced Usage

### Batch Processing Multiple Prefixes

Create a script (e.g., `run_all_sweeps.sh`):
```bash
#!/bin/bash
python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "Sweep_rank_Ex1_A0.0_" --num-laps 5
python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "Sweep_rank_Ex1_A0.4_" --num-laps 5
python TrainingLite/rl_racing/scripts/run_sweep_experiments.py --prefix "2002_from_Ex1_A0.0_" --num-laps 5
```

Then run: `bash run_all_sweeps.sh`

### Comparing Multiple Sweeps

After running multiple sweeps, you can:
1. Run analysis on each: `python TrainingLite/rl_racing/scripts/analyze_sweep_results.py --prefix "{prefix}" --plot`
2. Compare CSV files in a spreadsheet
3. Look at ranking in `sweep_{prefix}_analysis.txt`
