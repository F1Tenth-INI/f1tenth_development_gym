import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import LineString, Point
import os
import shutil
from pathlib import Path
import argparse
import re
import ast
from typing import Dict

import json

from utilities.Settings import Settings


def _sanitize_slice_bounds(data_length: int, start: int | None, end: int | None) -> tuple[int, int]:
    """Normalize start/end to safe dataframe slice bounds."""
    normalized_start = 0 if start is None else max(0, start)
    normalized_end = data_length if end is None else min(data_length, end)
    if normalized_end < normalized_start:
        raise ValueError(f"Invalid slice bounds: start={normalized_start}, end={normalized_end}")
    return normalized_start, normalized_end


def _extract_metadata_from_csv_header(csv_path: Path) -> dict[str, str]:
    """Parse key-value metadata from CSV comment header lines."""
    metadata: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            payload = stripped[1:].strip()
            if ":" not in payload:
                continue
            key, value = payload.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
    return metadata


def _extract_lap_times_from_csv_header(csv_path: Path) -> list[float]:
    """Parse lap times from CSV header line: '# Lap times: [..]'."""
    lap_times: list[float] = []
    with open(csv_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            if "Lap times:" not in stripped:
                continue
            match = re.search(r"Lap times:\s*\[(.*?)\]", stripped)
            if not match:
                continue
            payload = match.group(1).strip()
            if payload == "":
                return []
            values = [token.strip() for token in payload.split(",")]
            for value in values:
                if value:
                    lap_times.append(float(value))
    return lap_times


def _infer_lap_times_from_recording(recording: pd.DataFrame) -> list[float]:
    """
    Infer lap times from nearest waypoint index wrap-around.
    This is used as a fallback when lap times are not present in CSV headers.
    """
    required_cols = {"time", "nearest_wpt_idx"}
    if not required_cols.issubset(recording.columns):
        return []
    if len(recording) < 3:
        return []

    times = recording["time"].to_numpy(dtype=float)
    nearest_idx = recording["nearest_wpt_idx"].to_numpy(dtype=float).astype(int)
    max_idx = int(np.max(nearest_idx))
    if max_idx <= 1:
        return []

    low_threshold = int(0.25 * max_idx)
    high_threshold = int(0.75 * max_idx)

    crossing_times: list[float] = []
    for i in range(1, len(nearest_idx)):
        prev_idx = nearest_idx[i - 1]
        curr_idx = nearest_idx[i]
        if prev_idx >= high_threshold and curr_idx <= low_threshold:
            crossing_times.append(float(times[i]))

    if len(crossing_times) < 2:
        return []

    lap_times: list[float] = []
    for i in range(1, len(crossing_times)):
        lap_duration = crossing_times[i] - crossing_times[i - 1]
        # Filter numerical noise and obvious false wrap detections.
        if lap_duration > 1.0:
            lap_times.append(float(lap_duration))

    return lap_times


def _resolve_waypoint_file_for_error_stats(source_csv: Path, output_dir: Path) -> Path | None:
    """Resolve map waypoint CSV used to compute position error statistics."""
    candidate_paths = []
    configs_dir = output_dir / "configs"
    if configs_dir.exists():
        candidate_paths.extend(sorted(configs_dir.glob("*_wp.csv")))
    candidate_paths.append(Path(Settings.MAP_PATH) / f"{Settings.MAP_NAME}_wp.csv")
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def _resolve_map_dir_from_metadata(metadata: dict[str, str]) -> tuple[Path | None, str | None]:
    """Resolve map directory and map name from CSV metadata."""
    map_name = metadata.get("map name")
    map_path_meta = metadata.get("map path")

    if map_path_meta:
        map_dir = Path(map_path_meta).expanduser()
        if not map_dir.is_absolute():
            map_dir = (Path.cwd() / map_dir).resolve()
        if map_dir.exists() and map_name:
            return map_dir, map_name

    if map_name:
        map_dir = Path("utilities") / "maps" / map_name
        map_dir = (Path.cwd() / map_dir).resolve()
        if map_dir.exists():
            return map_dir, map_name

    return None, map_name


def _load_map_background_extent(map_dir: Path, map_name: str) -> tuple[np.ndarray, list[float]] | None:
    """
    Load map image and world extent from ROS-style map yaml.
    Returns: (image_array, [xmin, xmax, ymin, ymax]) or None.
    """
    yaml_path = map_dir / f"{map_name}.yaml"
    if not yaml_path.exists():
        return None

    yaml_fields: dict[str, str] = {}
    with open(yaml_path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text or text.startswith("#") or ":" not in text:
                continue
            key, value = text.split(":", 1)
            yaml_fields[key.strip()] = value.strip()

    image_name = yaml_fields.get("image")
    resolution_raw = yaml_fields.get("resolution")
    origin_raw = yaml_fields.get("origin")
    if image_name is None or resolution_raw is None or origin_raw is None:
        return None

    map_image_path = (map_dir / image_name).resolve()
    if not map_image_path.exists():
        return None

    resolution = float(resolution_raw)
    origin = ast.literal_eval(origin_raw)
    if not isinstance(origin, (list, tuple)) or len(origin) < 2:
        return None
    origin_x = float(origin[0])
    origin_y = float(origin[1])

    # Match simulator map loading in ScanSimulator2D.set_map():
    # image is flipped vertically after loading.
    image = np.flipud(plt.imread(map_image_path))
    height_px, width_px = image.shape[0], image.shape[1]
    extent = [
        origin_x,
        origin_x + width_px * resolution,
        origin_y,
        origin_y + height_px * resolution,
    ]
    return image, extent


def analyze_recording_csv(
    csv_path: str,
    start: int | None = None,
    end: int | None = None,
    output_dir: str | None = None,
    copy_recording: bool = True,
) -> Path:
    """
    Analyze a single recording CSV and generate plots in a dedicated folder.

    Output folder content:
    - copy of recording CSV
    - 2D trajectory plot
    - state plots
    """
    source_csv = Path(csv_path).expanduser().resolve()
    if not source_csv.exists():
        raise FileNotFoundError(f"Recording file not found: {source_csv}")
    if source_csv.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {source_csv}")

    recording = pd.read_csv(source_csv, comment="#")
    if recording.empty:
        raise ValueError(f"Recording file has no rows: {source_csv}")

    slice_start, slice_end = _sanitize_slice_bounds(len(recording), start, end)
    recording_slice = recording.iloc[slice_start:slice_end].copy()
    if recording_slice.empty:
        raise ValueError(
            f"Requested range is empty for file with {len(recording)} rows: "
            f"start={slice_start}, end={slice_end}"
        )

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_csv.parent / f"{source_csv.stem}_data"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if copy_recording:
        copied_csv_path = output_dir / source_csv.name
        shutil.copy2(source_csv, copied_csv_path)

    header_metadata = _extract_metadata_from_csv_header(source_csv)

    lap_times = _extract_lap_times_from_csv_header(source_csv)
    lap_time_source = "csv_header"
    if not lap_times:
        lap_times = _infer_lap_times_from_recording(recording_slice)
        lap_time_source = "inferred_from_nearest_wpt_idx"

    lap_time_summary = {
        "source": lap_time_source,
        "lap_times_s": lap_times,
        "count": len(lap_times),
        "best_s": float(min(lap_times)) if lap_times else None,
        "mean_s": float(np.mean(lap_times)) if lap_times else None,
        "std_s": float(np.std(lap_times)) if lap_times else None,
    }
    with open(output_dir / "lap_times_summary.json", "w", encoding="utf-8") as file:
        json.dump(lap_time_summary, file, indent=2)

    if lap_times:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(1, len(lap_times) + 1), lap_times, marker="o", linewidth=1.2)
        ax.set_xlabel("Lap Index")
        ax.set_ylabel("Lap Time [s]")
        ax.set_title(f"Lap Time Analysis ({source_csv.stem})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "lap_time_analysis.png", dpi=150)
        plt.close(fig)

    required_pose_cols = {"pose_x", "pose_y"}
    if not required_pose_cols.issubset(recording_slice.columns):
        missing = sorted(required_pose_cols.difference(recording_slice.columns))
        raise KeyError(f"Missing required trajectory columns: {missing}")

    fig, ax = plt.subplots(figsize=(9, 8))
    map_dir, map_name = _resolve_map_dir_from_metadata(header_metadata)
    if map_dir is not None and map_name is not None:
        map_background = _load_map_background_extent(map_dir, map_name)
        if map_background is not None:
            map_image, map_extent = map_background
            ax.imshow(map_image, cmap="gray", origin="lower", extent=map_extent, alpha=0.65)

    ax.plot(recording_slice["pose_x"], recording_slice["pose_y"], linewidth=1.2, color="tab:red", label="Trajectory")
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    title_map_suffix = f" | map={map_name}" if map_name is not None else ""
    ax.set_title(f"2D Trajectory ({source_csv.stem}{title_map_suffix})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_2d_map.png", dpi=150)
    plt.close(fig)

    waypoint_file = _resolve_waypoint_file_for_error_stats(source_csv, output_dir)
    error_stats_output: Dict[str, float | None | str] | None = None
    if waypoint_file is not None:
        waypoints = pd.read_csv(waypoint_file, comment="#")
        waypoints.columns = waypoints.columns.str.replace(" ", "")
        if {"x_m", "y_m"}.issubset(waypoints.columns):
            raceline = LineString(zip(waypoints["x_m"].to_numpy(), waypoints["y_m"].to_numpy()))
            positions = zip(recording_slice["pose_x"].to_numpy(), recording_slice["pose_y"].to_numpy())
            errors = np.array([Point(x, y).distance(raceline) for x, y in positions], dtype=float)
            error_stats_output = {
                "max": float(np.max(errors)),
                "min": float(np.min(errors)),
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "var": float(np.var(errors)),
            }
        else:
            error_stats_output = {
                "max": None,
                "min": None,
                "mean": None,
                "std": None,
                "var": None,
                "reason": f"Waypoint file missing required columns in {waypoint_file.name}",
            }
    else:
        error_stats_output = {
            "max": None,
            "min": None,
            "mean": None,
            "std": None,
            "var": None,
            "reason": "No waypoint file found for error calculation.",
        }

    with open(output_dir / "error_stats.json", "w", encoding="utf-8") as file:
        json.dump(error_stats_output, file, indent=4)

    candidate_state_cols = [
        "angular_vel_z",
        "linear_vel_x",
        "linear_vel_y",
        "pose_theta",
        "pose_theta_cos",
        "pose_theta_sin",
        "pose_x",
        "pose_y",
        "slip_angle",
        "steering_angle",
        "angular_control_calculated",
        "translational_control_calculated",
    ]
    state_cols = [col for col in candidate_state_cols if col in recording_slice.columns]
    if not state_cols:
        raise KeyError("No known state columns found for state plots.")

    x_axis = (
        recording_slice["time"].to_numpy()
        if "time" in recording_slice.columns
        else np.arange(len(recording_slice))
    )
    x_label = "Time [s]" if "time" in recording_slice.columns else "Sample Index"

    fig, axes = plt.subplots(len(state_cols), 1, figsize=(14, 2.5 * len(state_cols)), sharex=True)
    if len(state_cols) == 1:
        axes = [axes]

    for axis, state_name in zip(axes, state_cols):
        axis.plot(x_axis, recording_slice[state_name].to_numpy(), linewidth=1.0)
        axis.set_ylabel(state_name)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    fig.suptitle(f"State Plots ({source_csv.stem})", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "state_plots.png", dpi=150)
    plt.close(fig)

    control_cols = [col for col in ["angular_control_calculated", "translational_control_calculated"] if col in recording_slice.columns]
    if control_cols:
        fig, axes = plt.subplots(len(control_cols), 1, figsize=(12, 3 * len(control_cols)), sharex=True)
        if len(control_cols) == 1:
            axes = [axes]

        for axis, control_name in zip(axes, control_cols):
            axis.plot(x_axis, recording_slice[control_name].to_numpy(), linewidth=1.0)
            axis.set_ylabel(control_name)
            axis.grid(True, alpha=0.3)

        axes[-1].set_xlabel(x_label)
        fig.suptitle(f"Control Plots ({source_csv.stem})", y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / "control_plots.png", dpi=150)
        plt.close(fig)

    return output_dir


class ExperimentAnalyzer:
    def __init__(self, experiment_name, experiment_path = Settings.RECORDING_FOLDER, step_start=0, step_end= Settings.SIMULATION_LENGTH):
        
        self.step_start = 0
        self.step_end = step_end
        
        self.experiment_path = experiment_path
        self.experiment_name = experiment_name
        self.map_name = Settings.MAP_NAME
        self.map_path = Settings.MAP_PATH
        self.controller_name = 'neural'
        
        csv_path = os.path.join(self.experiment_path, self.experiment_name) 
        self.experiment_data_path = os.path.join(csv_path + "_data")
        self.experiment_configs_path = os.path.join(self.experiment_data_path, "configs")
        
        self.waypoints_file = os.path.join(self.map_path, self.map_name + "_wp")
        # self.waypoints_file = os.path.join(self.experiment_configs_path, self.map_name + "_wp")
        
        # Waypoints from 
        self.waypoints: pd.DataFrame = pd.read_csv(self.waypoints_file + ".csv", comment='#')   
        self.waypoints.columns = self.waypoints.columns.str.replace(' ', '') # Remove spaces from column names
        
        # Recording from csv file (cut alreay)
        self.recording: pd.DataFrame = pd.read_csv(csv_path + ".csv", comment='#')
        self.recording = self.recording.iloc[step_start:step_end]
        
        self.position_errors = self.get_position_error()
        

    def get_position_error(self) -> np.ndarray:
        optimal_x = self.waypoints['x_m'].values
        optimal_y = self.waypoints['y_m'].values

        optimal = list(zip(optimal_x, optimal_y))
        optimal_line = LineString(optimal).coords

        recorded_x = self.recording['pose_x'].values
        recorded_y = self.recording['pose_y'].values

        recorded = list(zip(recorded_x, recorded_y))
        recorded_line = LineString(recorded)

        errors = []

        for i in range(len(recorded_x)):
            errors.append(recorded_line.distance(Point(optimal_line[i%len(optimal_x)])))

        errors = np.array(errors)
        
        return errors

    

    def get_error_stats(self) -> Dict[str, float]:
        
        errors = self.position_errors
        error_stats: Dict[str, float] = {
            'max': float(np.max(errors)),
            'min': float(np.min(errors)),
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'var': float(np.var(errors))
        }
        
        return error_stats
    
    def plot_experiment(self):
        self.plot_errors()
        self.plot_states()
        self.plot_controls()
        self.plot_imu_data()
        self.save_error_stats()
        
        
    def plot_controls(self):
        # Plot States
        state_names = ['angular_vel_z','linear_vel_x','linear_vel_y','pose_theta','pose_theta_cos','pose_theta_sin','pose_x','pose_y',]
        
        # Create a new figure
        fig = plt.figure(figsize=(15, 20))  # width: 15 inches, height: 20 inches

        for index, state_name in enumerate(state_names):
            # Add subplot for each state
            plt.subplot(len(state_names), 1, index+1)  # 7 rows, 1 column, nth plot
            plt.title(state_name)
            plt.plot(self.recording[state_name].to_numpy()[1:], color="red")

    
        plt.savefig(os.path.join(self.experiment_data_path, "state_plots.png" ))
        plt.clf()
        
    def plot_states(self):
        # Plot Control
        
        angular_controls = self.recording['angular_control_calculated'].to_numpy()[1:]
        translational_controls = self.recording['translational_control_calculated'].to_numpy()[1:]   
        
        fig = plt.figure()
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
        plt.title("Angular Control")
        plt.plot(angular_controls, color="red")
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
        plt.title("Translational Control")
        plt.plot(translational_controls, color="blue")
        plt.savefig(os.path.join(self.experiment_data_path, "control_plots.png" ))
        plt.clf()
              
    def plot_errors(self):
        
        # Check of plot folder exists
        if not os.path.exists(self.experiment_data_path):
            os.makedirs(self.experiment_data_path)
            
        time_recorded = self.recording['time'].values
        
        controller_name = self.controller_name

        optimal_x = self.waypoints['x_m'].values
        optimal_y = self.waypoints['y_m'].values
        
        recorded_x = self.recording['pose_x'].values
        recorded_y = self.recording['pose_y'].values
        
        plt.clf()
        plt.figure()
        plt.plot(optimal_x, optimal_y, label='Raceline')
        plt.plot(recorded_x, recorded_y, label='Recorded Line')
        plt.legend()

        plt.xlabel('X-Position')
        plt.ylabel('Y-Position')
        # plt.title('Comparison between '+ controller_name +' raceline and recorded line waypoints on '+ map_name)
        plt.savefig(os.path.join(self.experiment_data_path, "position_error_birdview.png" ))


        plt.figure()
        plt.plot(time_recorded[self.step_start:self.step_end], self.position_errors[self.step_start:self.step_end],color='cyan', label='Position error with '+controller_name+' and waypoints')

        plt.xlabel('Time [s]', fontsize=24)
        plt.ylabel('Error [m]', fontsize=24)
        plt.title('Position error with Recording of '+controller_name+' and waypoints on '+ self.map_name, fontsize=24)
        plt.tick_params(axis='both', labelsize=24)
        plt.legend(loc='upper right', fontsize=24)
        plt.grid()
        
        plt.savefig(os.path.join(self.experiment_data_path, "position_error_distance.png" ))
        
        # Boxplot
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_ylabel('Error [m]')
        ax.set_title('Position error with Recording of '+controller_name+' and waypoints on '+ self.map_name)
        ax.boxplot(self.position_errors)    
        plt.savefig(os.path.join(self.experiment_data_path, "position_error_boxplot.png" ))


        # Plot estimated mu along track        
        try:
            
            mus = self.recording['mu'].values
            mus_predicted = self.recording['mu_predicted'].values
            mu_error = mus_predicted - mus
            max_abs_mu_error = np.max(np.abs(mu_error[10:]))

        
            plt.clf()   
            plt.figure(figsize=(8, 6), dpi=150)
            # Cut away first 10 values because they are not accurate and mess up color scale
            sc = plt.scatter(recorded_x[10:], recorded_y[10:], c=mu_error[10:], cmap='seismic', vmin=-max_abs_mu_error, vmax=max_abs_mu_error, label="Position")
            plt.colorbar(sc, label='Mu error')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Car Position with Mu Predicted ({Settings.SURFACE_FRICTION})')
            plt.legend()
            plt.savefig(os.path.join(self.experiment_data_path, "mu_predicted.png" ))
        except Exception as e:
            print(f"Warning: No mu values found in recording: {e}")

    def plot_imu_data(self):
        """Plot IMU data including accelerometer, gyroscope, and orientation data."""
        try:
            # Check if IMU data columns exist
            imu_columns = [col for col in self.recording.columns if col.startswith('imu_')]
            if not imu_columns:
                print("Warning: No IMU data found in recording")
                return
            
            # Get time data
            time_data = self.recording['time'].to_numpy()[1:]
            
            # Create figure with subplots for different IMU data types
            fig, axes = plt.subplots(4, 1, figsize=(15, 20))
            fig.suptitle('IMU Sensor Data', fontsize=16)
            
            # Plot 1: Accelerometer data
            ax1 = axes[0]
            accel_cols = [col for col in imu_columns if 'a_' in col and not 'quat' in col]
            for col in accel_cols:
                data = self.recording[col].to_numpy()[1:]
                # Remove gravity from Z-axis accelerometer
                if col == 'imu_a_z':
                    data = data - 9.81  # Remove gravity component
                ax1.plot(time_data, data, label=col.replace('imu_', ''))
            ax1.set_title('Accelerometer Data (m/s²) - Gravity Removed from Z-axis')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Gyroscope data
            ax2 = axes[1]
            gyro_cols = [col for col in imu_columns if 'gyro_' in col]
            for col in gyro_cols:
                ax2.plot(time_data, self.recording[col].to_numpy()[1:], label=col.replace('imu_', ''))
            ax2.set_title('Gyroscope Data (rad/s)')
            ax2.set_ylabel('Angular Velocity (rad/s)')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Euler angles
            ax3 = axes[2]
            euler_cols = [col for col in imu_columns if col in ['imu_roll', 'imu_pitch', 'imu_yaw']]
            for col in euler_cols:
                ax3.plot(time_data, self.recording[col].to_numpy()[1:], label=col.replace('imu_', ''))
            ax3.set_title('Euler Angles (rad)')
            ax3.set_ylabel('Angle (rad)')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Quaternion data
            ax4 = axes[3]
            quat_cols = [col for col in imu_columns if 'quat_' in col]
            for col in quat_cols:
                ax4.plot(time_data, self.recording[col].to_numpy()[1:], label=col.replace('imu_', ''))
            ax4.set_title('Quaternion Data')
            ax4.set_ylabel('Quaternion Component')
            ax4.set_xlabel('Time (s)')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_data_path, "imu_data.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create additional detailed plots for accelerometer and gyroscope
            self._plot_imu_detailed()
            
        except Exception as e:
            print(f"Warning: Error plotting IMU data: {e}")
    
    def _plot_imu_detailed(self):
        """Create detailed IMU plots with magnitude and individual components."""
        try:
            time_data = self.recording['time'].to_numpy()[1:]
            
            # Detailed accelerometer plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Detailed IMU Analysis', fontsize=16)
            
            # Accelerometer magnitude (with gravity removed from Z-axis)
            ax1 = axes[0, 0]
            accel_x = self.recording['imu_a_x'].to_numpy()[1:]
            accel_y = self.recording['imu_a_y'].to_numpy()[1:]
            accel_z = self.recording['imu_a_z'].to_numpy()[1:] - 9.81  # Remove gravity
            accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            ax1.plot(time_data, accel_magnitude, 'k-', linewidth=2, label='Magnitude (Z-gravity removed)')
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero reference')
            ax1.set_title('Accelerometer Magnitude (Gravity Removed from Z-axis)')
            ax1.set_ylabel('Acceleration (m/s²)')
            ax1.legend()
            ax1.grid(True)
            
            # Gyroscope magnitude
            ax2 = axes[0, 1]
            gyro_x = self.recording['imu_gyro_x'].to_numpy()[1:]
            gyro_y = self.recording['imu_gyro_y'].to_numpy()[1:]
            gyro_z = self.recording['imu_gyro_z'].to_numpy()[1:]
            gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            ax2.plot(time_data, gyro_magnitude, 'k-', linewidth=2, label='Magnitude')
            ax2.set_title('Gyroscope Magnitude')
            ax2.set_ylabel('Angular Velocity (rad/s)')
            ax2.legend()
            ax2.grid(True)
            
            # Quaternion magnitude (should be close to 1)
            ax3 = axes[1, 0]
            quat_w = self.recording['imu_quat_w'].to_numpy()[1:]
            quat_x = self.recording['imu_quat_x'].to_numpy()[1:]
            quat_y = self.recording['imu_quat_y'].to_numpy()[1:]
            quat_z = self.recording['imu_quat_z'].to_numpy()[1:]
            quat_magnitude = np.sqrt(quat_w**2 + quat_x**2 + quat_y**2 + quat_z**2)
            ax3.plot(time_data, quat_magnitude, 'k-', linewidth=2, label='Magnitude')
            ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Normalized (1.0)')
            ax3.set_title('Quaternion Magnitude')
            ax3.set_ylabel('Magnitude')
            ax3.set_xlabel('Time (s)')
            ax3.legend()
            ax3.grid(True)
            
            # Yaw angle over time
            ax4 = axes[1, 1]
            yaw_angle = self.recording['imu_yaw'].to_numpy()[1:]
            ax4.plot(time_data, yaw_angle, 'b-', linewidth=2, label='Yaw Angle')
            ax4.set_title('Yaw Angle Over Time')
            ax4.set_ylabel('Yaw Angle (rad)')
            ax4.set_xlabel('Time (s)')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.experiment_data_path, "imu_detailed_analysis.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Error creating detailed IMU plots: {e}")

    
    def save_error_stats(self):
        error_stats = self.get_error_stats()
        file = os.path.join(self.experiment_data_path, "error_stats.json") 
        with open(file, 'w') as json_file:
            json.dump(error_stats, json_file, indent=4)
    
     

def _parse_args():
    parser = argparse.ArgumentParser(description="Analyze recording CSV and generate plots.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to recording CSV file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional start index (inclusive). Defaults to beginning of file.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Optional end index (exclusive). Defaults to end of file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <recording_stem>_data next to the CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_path = analyze_recording_csv(
        csv_path=args.csv_path,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )
    print(f"Analysis completed. Output folder: {output_path}")