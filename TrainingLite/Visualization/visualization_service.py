#!/usr/bin/env python3
"""
Backend for the state comparison visualization webapp.
Handles CSV loading, JAX model predictions, config, and plot data.
"""

import importlib
import json
import os
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd

VIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(VIS_DIR))
CONFIG_PATH = os.path.join(VIS_DIR, "visualization_config.json")
UPLOADS_DIR = os.path.join(VIS_DIR, "uploads")
BROWSE_ROOTS = ["AnalyseData", "ExperimentRecordings"]
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "sim/f110_sim/envs"))
sys.path.append(os.path.join(REPO_ROOT, "utilities"))

from sim.f110_sim.envs.car_model_jax import CarModelJAX
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.imu_utilities import IMUUtilities
from utilities.state_utilities import STATE_INDICES, STATE_VARIABLES

STEERING_CONTROL_COLUMN = "angular_control_executed"
ACCELERATION_CONTROL_COLUMN = "translational_control_executed"
STEERING_CONTROL_ALIASES = (
    STEERING_CONTROL_COLUMN,
    "angular_control",
    "angular_control_calculated",
)
ACCELERATION_CONTROL_ALIASES = (
    ACCELERATION_CONTROL_COLUMN,
    "translational_control",
    "translational_control_calculated",
)

AVAILABLE_MODELS = {
    "pacejka": "Pure Pacejka Model",
    "ks_jax": "KS jax",
    "direct": "Direct Dynamics Neural Network",
    "residual": "Residual Dynamics Model",
}

@dataclass
class VisualizationSettings:
    csv_file_path: str = ""
    start_index: int = 0
    end_index: Optional[int] = None
    horizon_steps: int = 50
    steering_delay_steps: int = 0
    acceleration_delay_steps: int = 0
    enable_comparison: bool = True
    show_controls: bool = False
    show_delta_state: bool = False
    show_imu: bool = True
    show_all_comparisons: bool = False
    sync_scales: bool = False
    show_metrics: bool = True
    state_name: str = ""
    selected_other_data: List[str] = field(default_factory=list)
    comparison_start_index: int = 0
    default_car_model: str = "Pure Pacejka Model"
    default_car_parameters: str = "gym_car_parameters.yml"
    theme: str = "dark"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualizationSettings":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        if filtered.get("end_index") == "":
            filtered["end_index"] = None
        if filtered.get("theme") not in ("dark", "light"):
            filtered["theme"] = "dark"
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonJob:
    job_id: str
    status: str = "pending"
    current: int = 0
    total: int = 0
    message: str = ""
    error: Optional[str] = None


class VisualizationService:
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.csv_file_path: Optional[str] = None
        self.time_column = "time"
        self.state_columns = list(STATE_VARIABLES)
        self.control_columns = [STEERING_CONTROL_COLUMN, ACCELERATION_CONTROL_COLUMN]
        self.state_indices = STATE_INDICES

        self.car_models = None
        self._current_start_index: Optional[int] = None

        self.available_car_params: Dict[str, str] = {}
        car_files_dir = os.path.join(REPO_ROOT, "utilities", "car_files")
        if os.path.exists(car_files_dir):
            for filename in os.listdir(car_files_dir):
                if filename.endswith((".yml", ".yaml")):
                    self.available_car_params[filename] = filename

        self.reload_car_models()
        self.settings = VisualizationSettings()
        self.comparison_data_dict: Dict[int, Dict[str, np.ndarray]] = {}
        self._comparison_cache_key: Optional[tuple] = None
        self._jobs: Dict[str, ComparisonJob] = {}
        self._jobs_lock = threading.Lock()
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        self.load_config()
        self._try_load_config_csv()

    # ------------------------------------------------------------------ models
    def reload_car_models(self) -> None:
        for module_name in [
            "sim.f110_sim.envs.dynamic_model_pacejka_jax",
            "sim.f110_sim.envs.dynamic_model_ks_jax",
            "TrainingLite.dynamic_residual_jax.dynamics_model_residual",
            "TrainingLite.dynamic_residual_jax.predictor",
            "sim.f110_sim.envs.car_model_jax",
        ]:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])

        self.car_models = {
            "pacejka": CarModelJAX(model_type="pacejka", dt=0.04, intermediate_steps=4),
            "pacejka_custom": CarModelJAX(model_type="pacejka_custom", dt=0.04, intermediate_steps=4),
            "ks_jax": CarModelJAX(model_type="ks", dt=0.04, intermediate_steps=4),
            "residual": CarModelJAX(model_type="residual", dt=0.04, intermediate_steps=4),
        }

    def load_car_parameters(self, param_file: str) -> Optional[np.ndarray]:
        try:
            vehicle_params = VehicleParameters(param_file)
            params_array = vehicle_params.to_np_array()
            if len(params_array) >= 15:
                mu = params_array[0]
                c_pf_b = params_array[7]
                i_z = params_array[5]
                print(f"Loaded parameters from {param_file}: mu={mu:.3f}, C_Pf_B={c_pf_b:.3f}, I_z={i_z:.4f}")
            return params_array
        except Exception as exc:
            print(f"Error loading car parameters from {param_file}: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def get_model_key(self, display_name: str) -> str:
        for key, name in AVAILABLE_MODELS.items():
            if name == display_name:
                return key
        return list(AVAILABLE_MODELS.keys())[0]

    def get_car_parameters_filename(self, display_name: str) -> str:
        for filename in self.available_car_params:
            if self.available_car_params[filename] == display_name:
                return filename
        return list(self.available_car_params.keys())[0]

    def _resolve_control_column(self, aliases: tuple) -> Optional[str]:
        if self.data is None:
            return None
        for col in aliases:
            if col in self.data.columns:
                return col
        return None

    def get_resolved_control_columns(self) -> List[str]:
        cols: List[str] = []
        steering = self._resolve_control_column(STEERING_CONTROL_ALIASES)
        if steering:
            cols.append(steering)
        acceleration = self._resolve_control_column(ACCELERATION_CONTROL_ALIASES)
        if acceleration:
            cols.append(acceleration)
        return cols

    def extract_initial_state_at_index(self, index: int) -> jnp.ndarray:
        if self.data is None or index >= len(self.data):
            return jnp.zeros(10, dtype=jnp.float32)

        initial_state = np.zeros(10, dtype=np.float32)
        for col_name, idx in self.state_indices.items():
            if col_name in self.data.columns:
                initial_state[idx] = self.data[col_name].iloc[index]
        return jnp.array(initial_state)

    def extract_control_sequence_at_index(
        self,
        start_index: int,
        horizon: int,
        steering_delay: int = 0,
        acceleration_delay: int = 0,
    ) -> jnp.ndarray:
        if self.data is None:
            return jnp.zeros((horizon, 2), dtype=jnp.float32)

        control_sequence = np.zeros((horizon, 2), dtype=np.float32)
        steering_col = self._resolve_control_column(STEERING_CONTROL_ALIASES)
        accel_col = self._resolve_control_column(ACCELERATION_CONTROL_ALIASES)

        steering_start = start_index - steering_delay
        steering_end = min(steering_start + horizon, len(self.data))
        if steering_start >= 0 and steering_start < len(self.data):
            actual_horizon = max(0, steering_end - steering_start)
            if actual_horizon > 0 and steering_col:
                control_sequence[:actual_horizon, 0] = (
                    self.data[steering_col].iloc[steering_start:steering_end].values
                )
        elif steering_col:
            available_start = max(0, steering_start)
            available_end = min(available_start + horizon, len(self.data))
            if available_end > available_start:
                first_control = self.data[steering_col].iloc[0]
                offset = max(0, -steering_start)
                actual_length = min(horizon - offset, available_end - available_start)
                if actual_length > 0:
                    control_sequence[offset : offset + actual_length, 0] = (
                        self.data[steering_col]
                        .iloc[available_start : available_start + actual_length]
                        .values
                    )
                if offset > 0:
                    control_sequence[:offset, 0] = first_control

        accel_start = start_index - acceleration_delay
        accel_end = min(accel_start + horizon, len(self.data))
        if accel_start >= 0 and accel_start < len(self.data):
            actual_horizon = max(0, accel_end - accel_start)
            if actual_horizon > 0 and accel_col:
                control_sequence[:actual_horizon, 1] = (
                    self.data[accel_col].iloc[accel_start:accel_end].values
                )
        elif accel_col:
            available_start = max(0, accel_start)
            available_end = min(available_start + horizon, len(self.data))
            if available_end > available_start:
                first_control = self.data[accel_col].iloc[0]
                offset = max(0, -accel_start)
                actual_length = min(horizon - offset, available_end - available_start)
                if actual_length > 0:
                    control_sequence[offset : offset + actual_length, 1] = (
                        self.data[accel_col]
                        .iloc[available_start : available_start + actual_length]
                        .values
                    )
                if offset > 0:
                    control_sequence[:offset, 1] = first_control

        return jnp.array(control_sequence)

    def get_timestep(self) -> float:
        if self.data is not None and self.time_column in self.data.columns and len(self.data) > 1:
            return float(self.data[self.time_column].iloc[1] - self.data[self.time_column].iloc[0])
        return 0.04

    def _residual_histories_for_index(self, car_model, start_idx: int):
        history_length = 10
        if self.data is not None and start_idx >= history_length:
            state_history = np.array(
                [
                    self.extract_initial_state_at_index(start_idx - history_length + i)
                    for i in range(history_length)
                ],
                dtype=np.float32,
            )
            control_history = np.array(
                [
                    self.extract_control_sequence_at_index(
                        start_idx - history_length + i, 1, 0, 0
                    )[0]
                    for i in range(history_length)
                ],
                dtype=np.float32,
            )
            return state_history, control_history
        return (
            np.array(car_model.state_history),
            np.array(car_model.control_history),
        )

    def run_model_rollout(
        self,
        model_name: str,
        initial_state: jnp.ndarray,
        control_sequence: jnp.ndarray,
        car_params: np.ndarray,
        dt: float,
        horizon: int,
        start_index: Optional[int] = None,
        car_parameter_file: Optional[str] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Dynamics + IMU rollout; returns all physics states and IMU channels."""
        if self.car_models is None:
            self.reload_car_models()

        try:
            if model_name not in self.car_models:
                print(f"Unknown model: {model_name}")
                return None

            car_model = self.car_models[model_name]
            if car_params is not None:
                car_model.car_params = jnp.array(car_params)

            state_history = None
            control_history = None
            if model_name == "residual" and start_index is not None:
                state_history, control_history = self._residual_histories_for_index(
                    car_model, start_index
                )

            control_seq = control_sequence[:horizon]
            predicted_states, imu_series = IMUUtilities.rollout_dynamics_with_imu(
                car_model,
                np.asarray(initial_state),
                control_seq,
                dt,
                state_history=state_history,
                control_history=control_history,
                car_parameter_file=car_parameter_file,
            )
            return self.convert_predictions_to_dict(predicted_states, imu_series)
        except Exception as exc:
            print(f"Model prediction failed: {exc}")
            import traceback
            traceback.print_exc()
            return None

    def convert_predictions_to_dict(
        self,
        predicted_states: np.ndarray,
        imu_series: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        comparison_data = {}
        for col_name, idx in self.state_indices.items():
            if col_name in self.state_columns:
                comparison_data[col_name] = predicted_states[:, idx]
        if imu_series:
            for key in IMUUtilities.IMU_COMPARE_KEYS:
                if key in imu_series:
                    comparison_data[key] = np.asarray(imu_series[key])
        return comparison_data

    def run_single_comparison_at_index(
        self,
        start_index: int,
        model_name: str,
        param_file: str,
        horizon: int,
        steering_delay: int = 0,
        acceleration_delay: int = 0,
    ) -> Optional[Dict[str, np.ndarray]]:
        try:
            car_params = self.load_car_parameters(param_file)
            if car_params is None:
                return None

            initial_state = self.extract_initial_state_at_index(start_index)
            control_sequence = self.extract_control_sequence_at_index(
                start_index, horizon, steering_delay, acceleration_delay
            )
            self._current_start_index = start_index
            dt = self.get_timestep()

            return self.run_model_rollout(
                model_name,
                initial_state,
                control_sequence,
                car_params,
                dt,
                horizon,
                start_index=start_index,
                car_parameter_file=param_file,
            )
        except Exception as exc:
            print(f"Single comparison failed for index {start_index}: {exc}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------ config
    def load_config(self) -> VisualizationSettings:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
                self.settings = VisualizationSettings.from_dict(data)
            except Exception as exc:
                print(f"Could not load config: {exc}")
        return self.settings

    def save_config(self) -> None:
        config_dir = os.path.dirname(CONFIG_PATH)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.settings.to_dict(), f, indent=2)

    def update_settings(self, updates: Dict[str, Any]) -> VisualizationSettings:
        old_params_key = self._comparison_params_key() if self.data is not None else None
        current = self.settings.to_dict()
        for key, value in updates.items():
            if key in current:
                current[key] = value
        self.settings = VisualizationSettings.from_dict(current)
        if self.data is not None:
            self._normalize_range_indices()
            if old_params_key != self._comparison_params_key():
                self.clear_comparisons()
        self.save_config()
        return self.settings

    def _comparison_params_key(self) -> tuple:
        end = self.settings.end_index
        if self.data is not None and end is None:
            end = len(self.data)
        return (
            self.get_model_key(self.settings.default_car_model),
            self.get_param_filename(self.settings.default_car_parameters),
            int(self.settings.horizon_steps),
            int(self.settings.steering_delay_steps),
            int(self.settings.acceleration_delay_steps),
            int(self.settings.start_index),
            int(end) if end is not None else -1,
        )

    def _cached_comparison_states(self) -> List[str]:
        if not self.comparison_data_dict:
            return []
        sample = next(iter(self.comparison_data_dict.values()))
        return sorted(sample.keys())

    def _comparison_start_bounds(self) -> tuple:
        """Inclusive min/max comparison start indices for the active data range."""
        if self.data is None:
            return 0, -1
        horizon = self.settings.horizon_steps
        effective_start = self.settings.start_index
        effective_end = (
            self.settings.end_index if self.settings.end_index is not None else len(self.data)
        )
        last_start = effective_end - horizon
        if last_start < effective_start:
            return effective_start, effective_start - 1
        return effective_start, last_start

    def _expected_comparison_indices(self) -> List[int]:
        """Every valid comparison start index in the active data range."""
        min_start, last_start = self._comparison_start_bounds()
        if last_start < min_start:
            return []
        return list(range(min_start, last_start + 1))

    def is_comparison_cache_valid(self) -> bool:
        if not self.comparison_data_dict or self._comparison_cache_key is None:
            return False
        if self._comparison_cache_key != self._comparison_params_key():
            return False
        if self.data is None:
            return False
        expected = self._expected_comparison_indices()
        if not expected:
            return False
        if not set(expected).issubset(self.comparison_data_dict.keys()):
            return False
        available = set(self._physics_state_columns())
        required = available | set(IMUUtilities.IMU_COMPARE_KEYS)
        cached_states = set(self._cached_comparison_states())
        return required.issubset(cached_states)

    def get_comparison_cache_info(self) -> Dict[str, Any]:
        expected = self._expected_comparison_indices()
        indices_complete = (
            bool(expected)
            and set(expected).issubset(self.comparison_data_dict.keys())
        )
        valid = self.is_comparison_cache_valid()
        return {
            "valid": valid,
            "states": self._cached_comparison_states() if self.comparison_data_dict else [],
            "indices_count": len(self.comparison_data_dict),
            "expected_count": len(expected),
            "indices_complete": indices_complete,
            "params_key": list(self._comparison_cache_key) if self._comparison_cache_key else None,
        }

    def _normalize_range_indices(self) -> None:
        if self.data is None:
            return
        n = len(self.data)
        start = max(0, min(int(self.settings.start_index), max(0, n - 1)))
        end_raw = self.settings.end_index
        if end_raw is None:
            end = n
        else:
            end = max(start + 1, min(int(end_raw), n))
        self.settings.start_index = start
        self.settings.end_index = end
        self.get_comparison_slider_range()

    # ------------------------------------------------------------------ CSV I/O
    def _resolve_safe_path(self, path: str) -> str:
        if not path:
            raise ValueError("Path is empty")
        abs_path = os.path.abspath(path if os.path.isabs(path) else os.path.join(REPO_ROOT, path))
        if not abs_path.startswith(REPO_ROOT):
            raise ValueError("Path is outside repository root")
        return abs_path

    def browse_csv(self, rel_path: str = "") -> Dict[str, Any]:
        rel_path = rel_path.strip().strip("/")
        target = self._resolve_safe_path(rel_path) if rel_path else REPO_ROOT
        if not os.path.isdir(target):
            raise ValueError(f"Not a directory: {rel_path or '.'}")

        entries = []
        for name in sorted(os.listdir(target)):
            full = os.path.join(target, name)
            rel = os.path.relpath(full, REPO_ROOT)
            if os.path.isdir(full):
                entries.append({"name": name, "path": rel, "type": "dir"})
            elif name.endswith(".csv"):
                entries.append({"name": name, "path": rel, "type": "file"})

        parent = os.path.relpath(os.path.dirname(target), REPO_ROOT)
        if parent == ".":
            parent = ""
        return {
            "current_path": os.path.relpath(target, REPO_ROOT) if target != REPO_ROOT else "",
            "parent_path": parent,
            "roots": BROWSE_ROOTS,
            "entries": entries,
        }

    def load_csv_path(self, path: str) -> Dict[str, Any]:
        abs_path = self._resolve_safe_path(path)
        if not os.path.isfile(abs_path):
            raise ValueError(f"File not found: {path}")
        return self._load_csv_from_file(abs_path)

    def load_csv_upload(self, filename: str, content: bytes) -> Dict[str, Any]:
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
        safe_name = os.path.basename(filename)
        if not safe_name.endswith(".csv"):
            raise ValueError("Only CSV files are supported")
        dest = os.path.join(UPLOADS_DIR, safe_name)
        with open(dest, "wb") as f:
            f.write(content)
        return self._load_csv_from_file(dest)

    def reload_csv(self) -> Dict[str, Any]:
        if not self.settings.csv_file_path or not os.path.isfile(self.settings.csv_file_path):
            raise ValueError("No CSV file loaded")
        return self._load_csv_from_file(self.settings.csv_file_path)

    def _load_csv_from_file(self, abs_path: str) -> Dict[str, Any]:
        self.data = pd.read_csv(abs_path, comment="#")
        self.csv_file_path = abs_path
        self.settings.csv_file_path = abs_path
        self.clear_comparisons()

        selectable = self._selectable_state_names()
        if selectable and self.settings.state_name not in ("", *selectable):
            self.settings.state_name = selectable[0]

        if self.settings.end_index is None:
            self.settings.end_index = len(self.data)

        self.save_config()
        return self.get_session_info()

    def get_session_info(self) -> Dict[str, Any]:
        data = self.data
        return {
            "csv_file_path": self.settings.csv_file_path,
            "filename": os.path.basename(self.settings.csv_file_path)
            if self.settings.csv_file_path
            else None,
            "row_count": len(data) if data is not None else 0,
            "columns": list(data.columns) if data is not None else [],
            "available_states": self._selectable_state_names() if data is not None else [],
            "settings": self.settings.to_dict(),
            "comparison_slider": self.get_comparison_slider_range(),
            "comparison_indices": sorted(self.comparison_data_dict.keys()),
            "comparison_cache": self.get_comparison_cache_info(),
        }

    def _try_load_config_csv(self) -> None:
        path = self.settings.csv_file_path
        if path and os.path.isfile(path):
            try:
                self._load_csv_from_file(path)
                print(f"Loaded CSV from config: {path}")
            except Exception as exc:
                print(f"Could not auto-load CSV from config: {exc}")

    # ------------------------------------------------------------------ options
    def get_options(self) -> Dict[str, Any]:
        return {
            "models": [{"key": k, "label": v} for k, v in AVAILABLE_MODELS.items()],
            "car_parameters": list(self.available_car_params.values()),
            "state_columns": list(self.state_columns),
            "control_columns": list(self.control_columns),
        }

    def get_param_filename(self, display_name: str) -> str:
        return self.get_car_parameters_filename(display_name)

    # ------------------------------------------------------------------ comparison
    def get_comparison_slider_range(self) -> Dict[str, int]:
        if self.data is None:
            return {"min": 0, "max": 0}
        min_start, last_start = self._comparison_start_bounds()
        if last_start < min_start:
            return {"min": min_start, "max": min_start}
        comp_idx = self.settings.comparison_start_index
        comp_idx = max(min_start, min(comp_idx, last_start))
        self.settings.comparison_start_index = comp_idx
        return {"min": min_start, "max": last_start}

    def _get_delays(self) -> tuple:
        return (
            max(0, int(self.settings.steering_delay_steps)),
            max(0, int(self.settings.acceleration_delay_steps)),
        )

    def run_single_comparison(self, start_index: Optional[int] = None) -> Dict[str, Any]:
        if self.data is None:
            raise ValueError("No data loaded")
        horizon = self.settings.horizon_steps
        if horizon <= 0:
            raise ValueError("Horizon must be positive")

        idx = start_index if start_index is not None else self.settings.comparison_start_index
        model_key = self.get_model_key(self.settings.default_car_model)
        param_file = self.get_param_filename(self.settings.default_car_parameters)
        steering_delay, acceleration_delay = self._get_delays()

        result = self.run_single_comparison_at_index(
            idx, model_key, param_file, horizon, steering_delay, acceleration_delay
        )
        if result is None:
            raise RuntimeError(f"Comparison failed at index {idx}")

        self.comparison_data_dict[idx] = result
        return {"start_index": idx, "horizon": horizon, "states": list(result.keys())}

    def start_full_comparison(self, force: bool = False) -> str:
        if self.data is None:
            raise ValueError("No data loaded")

        if not force and self.is_comparison_cache_valid():
            job_id = str(uuid.uuid4())
            with self._jobs_lock:
                self._jobs[job_id] = ComparisonJob(
                    job_id=job_id,
                    status="completed",
                    total=1,
                    current=1,
                    message="Using cached comparisons (all states ready)",
                )
            return job_id

        start_indices = self._expected_comparison_indices()
        if not start_indices:
            raise ValueError("Horizon is larger than available data in current range")

        job_id = str(uuid.uuid4())
        job = ComparisonJob(job_id=job_id, status="running", total=len(start_indices))
        with self._jobs_lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_full_comparison_job,
            args=(job_id, start_indices),
            daemon=True,
        )
        thread.start()
        return job_id

    def _run_full_comparison_job(self, job_id: str, start_indices: List[int]) -> None:
        self.reload_car_models()
        self.comparison_data_dict = {}
        model_key = self.get_model_key(self.settings.default_car_model)
        param_file = self.get_param_filename(self.settings.default_car_parameters)
        horizon = self.settings.horizon_steps
        steering_delay, acceleration_delay = self._get_delays()

        successful = 0
        for i, start_idx in enumerate(start_indices):
            with self._jobs_lock:
                job = self._jobs.get(job_id)
                if job:
                    job.current = i
                    job.message = (
                        f"Computing prediction {i + 1}/{len(start_indices)} (index {start_idx})"
                    )

            result = self.run_single_comparison_at_index(
                start_idx, model_key, param_file, horizon, steering_delay, acceleration_delay
            )
            if result is not None:
                self.comparison_data_dict[start_idx] = result
                successful += 1

        slider = self.get_comparison_slider_range()
        if self.comparison_data_dict:
            valid = [
                idx for idx in self.comparison_data_dict if slider["min"] <= idx <= slider["max"]
            ]
            if valid:
                self.settings.comparison_start_index = min(valid)
            else:
                self.settings.comparison_start_index = slider["min"]
            self.settings.enable_comparison = True
            self.save_config()

        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                if successful == 0:
                    job.status = "failed"
                    job.error = "No successful comparisons computed"
                else:
                    job.status = "completed"
                    job.current = job.total
                    job.message = f"Completed {successful}/{len(start_indices)} comparisons"
                    self._comparison_cache_key = self._comparison_params_key()

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("Job not found")
            return {
                "job_id": job.job_id,
                "status": job.status,
                "current": job.current,
                "total": job.total,
                "message": job.message,
                "error": job.error,
                "comparison_indices": sorted(self.comparison_data_dict.keys()),
            }

    def clear_comparisons(self) -> None:
        self.comparison_data_dict = {}
        self._comparison_cache_key = None

    # ------------------------------------------------------------------ plot data
    def _get_time_slice(self, start_idx: int, end_idx: int) -> np.ndarray:
        if self.time_column in self.data.columns:
            return self.data[self.time_column].iloc[start_idx:end_idx].to_numpy()
        return np.arange(start_idx, end_idx, dtype=float)

    def _get_comp_time(self, comp_start_idx: int, horizon: int) -> np.ndarray:
        if self.time_column in self.data.columns:
            if comp_start_idx + horizon <= len(self.data):
                return self.data[self.time_column].iloc[
                    comp_start_idx : comp_start_idx + horizon
                ].to_numpy()
            available = self.data[self.time_column].iloc[comp_start_idx:].to_numpy()
            dt = self.get_timestep()
            missing = horizon - len(available)
            if len(available) > 0 and missing > 0:
                extra = np.arange(1, missing + 1) * dt + available[-1]
                return np.concatenate([available, extra])
            if len(available) == 0:
                return np.arange(comp_start_idx, comp_start_idx + horizon) * dt
            return available
        dt = self.get_timestep()
        return np.arange(comp_start_idx, comp_start_idx + horizon) * dt

    def _compute_delta(self, values: np.ndarray, state_name: str) -> np.ndarray:
        if len(values) <= 1:
            return np.array([])
        if state_name == "pose_theta":
            delta_raw = np.diff(values)
            return np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
        return np.diff(values)

    def _physics_state_columns(self) -> List[str]:
        if self.data is None:
            return []
        return [col for col in self.state_columns if col in self.data.columns]

    def _selectable_state_names(self) -> List[str]:
        """Physics states from CSV plus IMU channels (compared like other states)."""
        names = list(self._physics_state_columns())
        for imu_key in IMUUtilities.IMU_COMPARE_KEYS:
            if imu_key not in names:
                names.append(imu_key)
        return names

    def _states_matrix_from_dataframe(self, start_idx: int, end_idx: int) -> np.ndarray:
        n = end_idx - start_idx
        states = np.zeros((n, len(STATE_VARIABLES)), dtype=np.float64)
        for col_name, idx in self.state_indices.items():
            if col_name in self.data.columns:
                states[:, idx] = (
                    self.data[col_name].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
                )
        return states

    def _imu_ground_truth_from_states(self, start_idx: int, end_idx: int) -> Dict[str, List[float]]:
        dt = self.get_timestep()
        states = self._states_matrix_from_dataframe(start_idx, end_idx)
        prime_state = None
        if start_idx > 0:
            prime_state = self.extract_initial_state_at_index(start_idx - 1)
        param_file = self.get_param_filename(self.settings.default_car_parameters)
        return IMUUtilities.simulate_imu_series(
            states, dt, prime_state=prime_state, car_parameter_file=param_file
        )

    def _ground_truth_series(self, column_name: str, start_idx: int, end_idx: int) -> np.ndarray:
        if IMUUtilities.is_imu_channel(column_name):
            simulated = self._imu_ground_truth_from_states(start_idx, end_idx)
            return IMUUtilities.ground_truth_imu_series(
                self.data, column_name, start_idx, end_idx, simulated=simulated
            )
        if column_name in self.data.columns:
            return self.data[column_name].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
        raise ValueError(f"Invalid state selection: {column_name}")

    def _plot_slice_context(self) -> tuple:
        if self.data is None:
            raise ValueError("No data loaded")
        start_idx = self.settings.start_index
        end_idx = (
            self.settings.end_index if self.settings.end_index is not None else len(self.data)
        )
        time_data = self._get_time_slice(start_idx, end_idx)
        state_cols = self._selectable_state_names()
        return start_idx, end_idx, time_data, state_cols

    def _plot_shared_columns(self, start_idx: int, end_idx: int) -> Dict[str, Any]:
        other_data: Dict[str, List[float]] = {}
        for col in self.settings.selected_other_data:
            if col in self.data.columns:
                other_data[col] = (
                    self.data[col].iloc[start_idx:end_idx].to_numpy().tolist()
                )

        controls: Dict[str, List[float]] = {}
        if self.settings.show_controls:
            for col in self.get_resolved_control_columns():
                controls[col] = (
                    self.data[col].iloc[start_idx:end_idx].to_numpy().tolist()
                )
        return {"other_data": other_data, "controls": controls}

    def _prediction_entries_for_state(
        self,
        state_name: str,
        start_idx: int,
        end_idx: int,
        include_delta: bool,
    ) -> List[Dict[str, Any]]:
        if not self.settings.enable_comparison or not self.comparison_data_dict:
            return []

        if self.settings.show_all_comparisons:
            comp_items = [
                (idx, data)
                for idx, data in self.comparison_data_dict.items()
                if start_idx <= idx < end_idx and state_name in data
            ]
        else:
            comp_idx = self.settings.comparison_start_index
            if (
                comp_idx in self.comparison_data_dict
                and state_name in self.comparison_data_dict[comp_idx]
            ):
                comp_items = [(comp_idx, self.comparison_data_dict[comp_idx])]
            else:
                comp_items = []

        entries: List[Dict[str, Any]] = []
        for comp_idx, comp_data in comp_items:
            pred = np.array(comp_data[state_name])
            comp_time = self._get_comp_time(comp_idx, len(pred))
            pred_entry: Dict[str, Any] = {
                "start_index": comp_idx,
                "time": comp_time.tolist(),
                "values": pred.tolist(),
            }
            if include_delta and len(pred) > 1:
                delta_pred = self._compute_delta(pred, state_name)
                pred_entry["delta"] = {
                    "time": comp_time[:-1].tolist(),
                    "values": delta_pred.tolist(),
                }
            entries.append(pred_entry)
        return entries

    def get_plot_bundle(self) -> Dict[str, Any]:
        """All states + cached predictions in one payload for client-side state switching."""
        start_idx, end_idx, time_data, state_cols = self._plot_slice_context()
        shared = self._plot_shared_columns(start_idx, end_idx)

        ground_truth: Dict[str, List[float]] = {}
        for col in state_cols:
            ground_truth[col] = self._ground_truth_series(col, start_idx, end_idx).tolist()

        predictions: List[Dict[str, Any]] = []
        if self.comparison_data_dict:
            for comp_idx, comp_data in self.comparison_data_dict.items():
                if comp_idx < start_idx or comp_idx >= end_idx:
                    continue
                states_out: Dict[str, List[float]] = {}
                for state_name in state_cols:
                    if state_name in comp_data:
                        states_out[state_name] = np.asarray(comp_data[state_name]).tolist()
                if not states_out:
                    continue
                horizon = len(next(iter(states_out.values())))
                comp_time = self._get_comp_time(comp_idx, horizon)
                predictions.append({
                    "start_index": comp_idx,
                    "time": comp_time.tolist(),
                    "states": states_out,
                })

        return {
            "time": time_data.tolist(),
            "start_index": start_idx,
            "end_index": end_idx,
            "state_names": state_cols,
            "ground_truth": ground_truth,
            "predictions": predictions,
            **shared,
        }

    def get_plot_data(self) -> Dict[str, Any]:
        if self.data is None:
            raise ValueError("No data loaded")

        start_idx, end_idx, time_data, _ = self._plot_slice_context()
        shared = self._plot_shared_columns(start_idx, end_idx)
        state_name = self.settings.state_name

        if not state_name:
            return {
                "state_name": "",
                "time": time_data.tolist(),
                "ground_truth": [],
                "predictions": [],
                "other_data": shared["other_data"],
                "controls": shared["controls"],
                "delta": {},
            }

        if state_name not in self._selectable_state_names():
            raise ValueError("Invalid state selection")

        state_data = self._ground_truth_series(state_name, start_idx, end_idx)

        result: Dict[str, Any] = {
            "state_name": state_name,
            "time": time_data.tolist(),
            "ground_truth": state_data.tolist(),
            "predictions": self._prediction_entries_for_state(
                state_name, start_idx, end_idx, self.settings.show_delta_state
            ),
            "other_data": shared["other_data"],
            "controls": shared["controls"],
            "delta": {},
        }

        if self.settings.show_delta_state:
            delta_col = f"delta_state_{state_name}"
            if delta_col in self.data.columns:
                delta_gt = self.data[delta_col].iloc[start_idx : end_idx - 1].to_numpy()
            else:
                delta_gt = self._compute_delta(state_data, state_name)
            delta_time = time_data[:-1] if len(time_data) > 1 else np.array([])
            result["delta"]["ground_truth"] = {
                "time": delta_time.tolist(),
                "values": delta_gt.tolist(),
            }

        return result

    # ------------------------------------------------------------------ metrics
    def get_metrics(self) -> Optional[Dict[str, float]]:
        if not self.settings.show_metrics or self.data is None:
            return None

        state_name = self.settings.state_name
        if not state_name or state_name not in self._selectable_state_names():
            return None
        if not self.comparison_data_dict:
            return None

        start_idx = self.settings.start_index
        end_idx = (
            self.settings.end_index if self.settings.end_index is not None else len(self.data)
        )

        all_predictions: List[float] = []
        all_ground_truth: List[float] = []

        if self.settings.show_all_comparisons:
            comp_items = self.comparison_data_dict.items()
        else:
            comp_idx = self.settings.comparison_start_index
            if comp_idx in self.comparison_data_dict:
                comp_items = [(comp_idx, self.comparison_data_dict[comp_idx])]
            else:
                return None

        for comp_start_idx, comparison_data in comp_items:
            if self.settings.show_all_comparisons and (
                comp_start_idx < start_idx or comp_start_idx >= end_idx
            ):
                continue
            if state_name not in comparison_data:
                continue
            prediction = np.array(comparison_data[state_name])
            pred_end = min(comp_start_idx + len(prediction), end_idx)
            gt_slice = self._ground_truth_series(state_name, comp_start_idx, pred_end)
            min_len = min(len(prediction), len(gt_slice))
            all_predictions.extend(prediction[:min_len].tolist())
            all_ground_truth.extend(gt_slice[:min_len].tolist())

        if not all_predictions:
            return None

        pred_data = np.array(all_predictions)
        gt_data = np.array(all_ground_truth)
        error = pred_data - gt_data
        return {
            "mean_error": float(np.mean(np.abs(error))),
            "max_error": float(np.max(np.abs(error))),
            "error_std": float(np.std(error)),
            "rmse": float(np.sqrt(np.mean(error**2))),
        }
