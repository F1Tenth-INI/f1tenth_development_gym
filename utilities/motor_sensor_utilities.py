"""Motor / VESC sensor format helpers and rollout utilities."""

import numpy as np

from utilities.motor_sensor_simulator import MotorSensorSimulator


class MotorSensorUtilities:
  # Primary simulated / recorded channels used for comparison plots
  MOTOR_COMPARE_KEYS = (
      "motor_angular_velocity",
      "motor_current_a",
  )
  MOTOR_OVERLAY_LABELS = {
      "motor_angular_velocity": "motor: ω (ERPM)",
      "motor_current_a": "motor: current [A]",
  }
  RECORDED_MOTOR_ALIASES = {
      "motor_angular_velocity": (
          "motor_angular_velocity",
          "motor_speed_erpm",
          "vesc_speed",
      ),
      "motor_current_a": (
          "motor_current_a",
          "vesc_current_motor",
      ),
  }

  @staticmethod
  def is_motor_channel(name: str) -> bool:
      return name in MotorSensorUtilities.MOTOR_COMPARE_KEYS

  @staticmethod
  def resolve_recorded_column(data_columns, motor_key: str):
      for alias in MotorSensorUtilities.RECORDED_MOTOR_ALIASES.get(motor_key, ()):
          if alias in data_columns:
              return alias
      return None

  @staticmethod
  def ground_truth_motor_series(
      data,
      motor_key: str,
      start_idx: int,
      end_idx: int,
      simulated=None,
  ):
      """Ground-truth motor sensor for plotting: prefer physical CSV over simulation."""
      for alias in MotorSensorUtilities.RECORDED_MOTOR_ALIASES.get(motor_key, ()):
          if alias not in data.columns:
              continue
          series = data[alias].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
          if alias.startswith("vesc_") or np.max(np.abs(series)) > 1e-12:
              return series
      if simulated is not None and motor_key in simulated:
          return np.asarray(simulated[motor_key], dtype=np.float64)
      return np.zeros(max(0, end_idx - start_idx), dtype=np.float64)

  @staticmethod
  def _load_car_params(car_parameter_file=None):
      from utilities.car_files.vehicle_parameters import VehicleParameters

      param_file = car_parameter_file or "gym_car_parameters.yml"
      return VehicleParameters(param_file)

  @staticmethod
  def simulate_motor_series(
      states,
      controls,
      dt: float,
      prime_state=None,
      car_parameter_file=None,
  ):
      """
      Run MotorSensorSimulator over state/control sequences.

      Returns:
          dict with motor_angular_velocity and motor_current_a lists
      """
      states = np.asarray(states, dtype=np.float64)
      controls = np.asarray(controls, dtype=np.float64)
      if controls.ndim == 1:
          controls = controls.reshape(-1, 2)
      horizon = min(len(states), len(controls))
      car_params = MotorSensorUtilities._load_car_params(car_parameter_file)

      out = {key: [] for key in MotorSensorUtilities.MOTOR_COMPARE_KEYS}
      prev_state = (
          np.asarray(prime_state, dtype=np.float64)
          if prime_state is not None
          else states[0].copy()
      )
      for i in range(horizon):
          state = states[i]
          control = controls[i]
          motor_dict = MotorSensorSimulator.from_states(
              state, prev_state, control, car_params, dt=dt
          )
          for key in MotorSensorUtilities.MOTOR_COMPARE_KEYS:
              out[key].append(float(motor_dict[key]))
          prev_state = state
      return out

  @staticmethod
  def simulate_motor_arrays(
      states,
      controls,
      dt: float,
      prime_state=None,
      car_parameter_file=None,
  ):
      """Like simulate_motor_series but returns numpy arrays keyed by channel."""
      series = MotorSensorUtilities.simulate_motor_series(
          states,
          controls,
          dt,
          prime_state=prime_state,
          car_parameter_file=car_parameter_file,
      )
      return {
          key: np.asarray(values, dtype=np.float64)
          for key, values in series.items()
      }

  @staticmethod
  def control_sequence_from_dataframe(data, start_idx: int, horizon: int):
      """Build (horizon, 2) control sequence from recorded CSV columns."""
      steering_aliases = (
          "angular_control_executed",
          "angular_control",
          "angular_control_calculated",
      )
      accel_aliases = (
          "translational_control_executed",
          "translational_control",
          "translational_control_calculated",
      )

      def _pick_column(aliases):
          for col in aliases:
              if col in data.columns:
                  return col
          return None

      steering_col = _pick_column(steering_aliases)
      accel_col = _pick_column(accel_aliases)
      controls = np.zeros((horizon, 2), dtype=np.float64)
      end_idx = min(start_idx + horizon, len(data))
      actual = max(0, end_idx - start_idx)
      if actual > 0:
          if steering_col:
              controls[:actual, 0] = (
                  data[steering_col].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
              )
          if accel_col:
              controls[:actual, 1] = (
                  data[accel_col].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
              )
      if actual < horizon and actual > 0:
          controls[actual:] = controls[actual - 1]
      return controls
