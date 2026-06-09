"""IMU data format helpers and rollout utilities."""

import numpy as np


class IMUUtilities:
    # Linear acceleration (3-axis)
    ACCEL_X_IDX = 0
    ACCEL_Y_IDX = 1
    ACCEL_Z_IDX = 2

    # Angular velocity (3-axis)
    GYRO_X_IDX = 3
    GYRO_Y_IDX = 4
    GYRO_Z_IDX = 5

    # Euler angles (3-axis)
    EULER_ROLL_IDX = 6
    EULER_PITCH_IDX = 7
    EULER_YAW_IDX = 8

    # Quaternion (4-axis)
    QUAT_W_IDX = 9
    QUAT_X_IDX = 10
    QUAT_Y_IDX = 11
    QUAT_Z_IDX = 12

    IMU_DATA_DIM = 13

    IMU_DICT_KEYS = (
        "imu_a_x",
        "imu_a_y",
        "imu_a_z",
        "imu_gyro_x",
        "imu_gyro_y",
        "imu_gyro_z",
        "imu_roll",
        "imu_pitch",
        "imu_yaw",
        "imu_quat_w",
        "imu_quat_x",
        "imu_quat_y",
        "imu_quat_z",
    )

    # Primary simulated / recorded channel names used for comparison plots
    IMU_COMPARE_KEYS = ("imu_a_x", "imu_a_y", "imu_gyro_z")
    IMU_OVERLAY_LABELS = {
        "imu_a_x": "imu: a_x",
        "imu_a_y": "imu: a_y",
        "imu_gyro_z": "imu: av_z",
    }
    RECORDED_IMU_ALIASES = {
        "imu_a_x": ("imu1_a_x", "imu_accel_x", "imu_a_x"),
        "imu_a_y": ("imu1_a_y", "imu_accel_y", "imu_a_y"),
        "imu_gyro_z": ("imu1_av_z", "imu_av_z", "imu_gyro_z"),
    }

    @staticmethod
    def zeros_dict():
        return {
            "imu_a_x": 0.0,
            "imu_a_y": 0.0,
            "imu_a_z": 0.0,
            "imu_gyro_x": 0.0,
            "imu_gyro_y": 0.0,
            "imu_gyro_z": 0.0,
            "imu_roll": 0.0,
            "imu_pitch": 0.0,
            "imu_yaw": 0.0,
            "imu_quat_w": 1.0,
            "imu_quat_x": 0.0,
            "imu_quat_y": 0.0,
            "imu_quat_z": 0.0,
        }

    @staticmethod
    def imu_array_to_dict(imu_array):
        imu_array = np.asarray(imu_array, dtype=np.float64)
        return {
            "imu_a_x": float(imu_array[IMUUtilities.ACCEL_X_IDX]),
            "imu_a_y": float(imu_array[IMUUtilities.ACCEL_Y_IDX]),
            "imu_a_z": float(imu_array[IMUUtilities.ACCEL_Z_IDX]),
            "imu_gyro_x": float(imu_array[IMUUtilities.GYRO_X_IDX]),
            "imu_gyro_y": float(imu_array[IMUUtilities.GYRO_Y_IDX]),
            "imu_gyro_z": float(imu_array[IMUUtilities.GYRO_Z_IDX]),
            "imu_roll": float(imu_array[IMUUtilities.EULER_ROLL_IDX]),
            "imu_pitch": float(imu_array[IMUUtilities.EULER_PITCH_IDX]),
            "imu_yaw": float(imu_array[IMUUtilities.EULER_YAW_IDX]),
            "imu_quat_w": float(imu_array[IMUUtilities.QUAT_W_IDX]),
            "imu_quat_x": float(imu_array[IMUUtilities.QUAT_X_IDX]),
            "imu_quat_y": float(imu_array[IMUUtilities.QUAT_Y_IDX]),
            "imu_quat_z": float(imu_array[IMUUtilities.QUAT_Z_IDX]),
        }

    array_to_dict = imu_array_to_dict

    @staticmethod
    def coerce_dict(imu):
        if isinstance(imu, dict):
            return imu
        return IMUUtilities.imu_array_to_dict(np.asarray(imu))

    @staticmethod
    def overlay_label_dict(imu_data):
        if not imu_data:
            return {}
        return {
            IMUUtilities.IMU_OVERLAY_LABELS[key]: float(imu_data[key])
            for key in IMUUtilities.IMU_COMPARE_KEYS
            if key in imu_data
        }

    @staticmethod
    def dict_to_array(imu_dict):
        return np.array([float(imu_dict[key]) for key in IMUUtilities.IMU_DICT_KEYS])

    @staticmethod
    def is_imu_channel(name: str) -> bool:
        return name in IMUUtilities.IMU_COMPARE_KEYS

    @staticmethod
    def resolve_recorded_column(data_columns, imu_key: str):
        """Find the best recorded CSV column for a canonical IMU key."""
        for alias in IMUUtilities.RECORDED_IMU_ALIASES.get(imu_key, ()):
            if alias in data_columns:
                return alias
        return None

    @staticmethod
    def ground_truth_imu_series(data, imu_key: str, start_idx: int, end_idx: int, simulated=None):
        """
        Ground-truth IMU for plotting: prefer physical CSV columns over empty sim placeholders.
        """
        for alias in IMUUtilities.RECORDED_IMU_ALIASES.get(imu_key, ()):
            if alias not in data.columns:
                continue
            series = data[alias].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
            if alias.startswith("imu1_") or np.max(np.abs(series)) > 1e-12:
                return series
        if simulated is not None and imu_key in simulated:
            return np.asarray(simulated[imu_key], dtype=np.float64)
        return np.zeros(max(0, end_idx - start_idx), dtype=np.float64)

    @staticmethod
    def rollout_dynamics_with_imu(
        car_model,
        initial_state,
        control_sequence,
        dt: float,
        state_history=None,
        control_history=None,
        car_parameter_file=None,
    ):
        """
        Roll out vehicle dynamics and IMU together (one IMU sample per dynamics step).

        Returns:
            states: (horizon, n_state)
            imu_series: dict imu_a_x / imu_a_y / imu_gyro_z each (horizon,)
        """
        from utilities.imu_simulator import IMUSimulator

        controls = np.asarray(control_sequence, dtype=np.float64)
        horizon = len(controls)
        initial_state = np.asarray(initial_state, dtype=np.float64)
        n_state = len(initial_state)
        states_out = np.zeros((horizon, n_state), dtype=np.float64)
        imu_out = {
            key: np.zeros(horizon, dtype=np.float64)
            for key in IMUUtilities.IMU_COMPARE_KEYS
        }
        imu_sim = IMUSimulator(car_parameter_file=car_parameter_file)
        imu_sim.prime(initial_state)

        if getattr(car_model, "model_type", None) == "residual":
            sh = np.array(
                state_history
                if state_history is not None
                else car_model.state_history,
                dtype=np.float32,
            )
            ch = np.array(
                control_history
                if control_history is not None
                else car_model.control_history,
                dtype=np.float32,
            )
            state = initial_state.astype(np.float32)
            for i in range(horizon):
                control = controls[i]
                traj = car_model.car_steps_sequential(
                    state,
                    np.array([control], dtype=np.float32),
                    state_history=sh,
                    control_history=ch,
                )
                state = np.asarray(traj[0], dtype=np.float64)
                states_out[i] = state
                imu_dict = imu_sim.update_car_state(state, dt)
                for key in IMUUtilities.IMU_COMPARE_KEYS:
                    imu_out[key][i] = imu_dict[key]
                sh = np.roll(sh, -1, axis=0)
                sh[-1] = state.astype(np.float32)
                ch = np.roll(ch, -1, axis=0)
                ch[-1] = control.astype(np.float32)
            return states_out, imu_out

        states_out = np.asarray(
            car_model.car_steps_sequential(
                initial_state,
                controls,
                state_history=state_history,
                control_history=control_history,
            ),
            dtype=np.float64,
        )
        for i, state in enumerate(states_out):
            imu_dict = imu_sim.update_car_state(state, dt)
            for key in IMUUtilities.IMU_COMPARE_KEYS:
                imu_out[key][i] = imu_dict[key]
        return states_out, imu_out

    @staticmethod
    def simulate_imu_series(states, dt: float, prime_state=None, car_parameter_file=None):
        """
        Run IMUSimulator over a sequence of car states.

        Returns:
            dict with imu_a_x, imu_a_y, imu_gyro_z lists
        """
        from utilities.imu_simulator import IMUSimulator

        imu_sim = IMUSimulator(car_parameter_file=car_parameter_file)
        if prime_state is not None:
            imu_sim.prime(prime_state)
        out = {key: [] for key in IMUUtilities.IMU_COMPARE_KEYS}
        for state in states:
            imu_dict = imu_sim.update_car_state(np.asarray(state, dtype=np.float64), dt)
            for key in IMUUtilities.IMU_COMPARE_KEYS:
                out[key].append(float(imu_dict[key]))
        return out

    @staticmethod
    def states_matrix_from_dict(state_dict, horizon: int) -> np.ndarray:
        """Build (horizon, NUMBER_OF_STATES) matrix from per-variable arrays."""
        from utilities.state_utilities import NUMBER_OF_STATES, STATE_INDICES

        states = np.zeros((horizon, NUMBER_OF_STATES), dtype=np.float64)
        for col_name, idx in STATE_INDICES.items():
            if col_name in state_dict:
                values = np.asarray(state_dict[col_name], dtype=np.float64)
                states[: len(values), idx] = values[:horizon]
        return states
