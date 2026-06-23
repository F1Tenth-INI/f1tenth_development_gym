import math

import numpy as np

from utilities.Settings import Settings
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.imu_utilities import IMUUtilities
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)


class IMUSimulator:
    """Simulate 6-DOF IMU readings from 2D vehicle state."""

    def __init__(self, noise_level=0.01, car_parameter_file=None):
        self.noise_level = noise_level
        self._load_imu_position(car_parameter_file)
        self._reset_state()

    def _load_imu_position(self, car_parameter_file=None):
        """Load IMU mount offset from car parameters (body frame, relative to rear axle)."""
        param_file = car_parameter_file or Settings.ENV_CAR_PARAMETER_FILE
        params = VehicleParameters(param_file)
        self.imu_x = float(params.imu_x)
        self.imu_y = float(params.imu_y)
        # Dynamics velocities are at CoG; rear axle is lr behind CoG along body +x.
        self.imu_dx_cog = float(self.imu_x - params.lr)
        self.imu_dy_cog = self.imu_y

    def _reset_state(self):
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_avz = 0.0
        self.first_update = True

    @staticmethod
    def body_velocity_at_imu(v_x, v_y, av_z, dx_cog, dy_cog):
        """Rigid-body velocity at an IMU offset from the CoG in the body frame."""
        return (
            float(v_x - av_z * dy_cog),
            float(v_y + av_z * dx_cog),
        )

    @staticmethod
    def body_acceleration_at_imu(
        v_x,
        v_y,
        av_z,
        last_v_x,
        last_v_y,
        last_av_z,
        dx_cog,
        dy_cog,
        delta_time,
    ):
        """
        Proper acceleration at an IMU offset in the rotating body frame.

        Uses a_body = dv/dt + ω×v at CoG plus rigid-body offset terms
        α×r and ω×(ω×r).
        """
        if delta_time <= 0.0:
            return 0.0, 0.0

        alpha_z = (av_z - last_av_z) / delta_time
        dv_x = (v_x - last_v_x) / delta_time
        dv_y = (v_y - last_v_y) / delta_time

        a_x_cog = dv_x - av_z * v_y
        a_y_cog = dv_y + av_z * v_x

        a_x_imu = a_x_cog - alpha_z * dy_cog - av_z**2 * dx_cog
        a_y_imu = a_y_cog + alpha_z * dx_cog - av_z**2 * dy_cog
        return float(a_x_imu), float(a_y_imu)

    @staticmethod
    def from_states(state, prev_state, dt, car_params):
        """Compute IMU dict from current and previous vehicle states (no internal history)."""
        state = np.asarray(state, dtype=np.float64)
        prev_state = np.asarray(prev_state, dtype=np.float64)
        if dt <= 0.0 or np.array_equal(state, prev_state):
            return IMUUtilities.zeros_dict()

        yaw = float(state[POSE_THETA_IDX])
        v_x = float(state[LINEAR_VEL_X_IDX])
        v_y = float(state[LINEAR_VEL_Y_IDX])
        av_z = float(state[ANGULAR_VEL_Z_IDX])
        dx_cog = float(car_params.imu_x - car_params.lr)
        dy_cog = float(car_params.imu_y)
        a_x, a_y = IMUSimulator.body_acceleration_at_imu(
            v_x,
            v_y,
            av_z,
            float(prev_state[LINEAR_VEL_X_IDX]),
            float(prev_state[LINEAR_VEL_Y_IDX]),
            float(prev_state[ANGULAR_VEL_Z_IDX]),
            dx_cog,
            dy_cog,
            dt,
        )
        quat_w, quat_x, quat_y, quat_z = IMUSimulator._yaw_to_quaternion(yaw)
        return {
            "imu_a_x": a_x,
            "imu_a_y": a_y,
            "imu_a_z": float(car_params.g),
            "imu_gyro_x": 0.0,
            "imu_gyro_y": 0.0,
            "imu_gyro_z": av_z,
            "imu_roll": 0.0,
            "imu_pitch": 0.0,
            "imu_yaw": yaw,
            "imu_quat_w": quat_w,
            "imu_quat_x": quat_x,
            "imu_quat_y": quat_y,
            "imu_quat_z": quat_z,
        }

    def _set_last_kinematics(self, car_state):
        car_state = np.asarray(car_state)
        v_x = (
            float(car_state[LINEAR_VEL_X_IDX])
            if len(car_state) > LINEAR_VEL_X_IDX
            else 0.0
        )
        v_y = (
            float(car_state[LINEAR_VEL_Y_IDX])
            if len(car_state) > LINEAR_VEL_Y_IDX
            else 0.0
        )
        av_z = (
            float(car_state[ANGULAR_VEL_Z_IDX])
            if len(car_state) > ANGULAR_VEL_Z_IDX
            else 0.0
        )
        self.last_x = float(car_state[POSE_X_IDX])
        self.last_y = float(car_state[POSE_Y_IDX])
        self.last_yaw = float(car_state[POSE_THETA_IDX])
        self.last_vx = v_x
        self.last_vy = v_y
        self.last_avz = av_z

    def update_car_state(self, car_state, delta_time=Settings.TIMESTEP_CONTROL):
        """
        Compute IMU sample from the current car state.

        Returns:
            dict with imu_a_*, imu_gyro_*, imu_roll/pitch/yaw, imu_quat_* keys
        """
        if self.first_update:
            self._set_last_kinematics(car_state)
            self.first_update = False
            return IMUUtilities.zeros_dict()

        car_state = np.asarray(car_state)
        pose_x = float(car_state[POSE_X_IDX])
        pose_y = float(car_state[POSE_Y_IDX])
        yaw = float(car_state[POSE_THETA_IDX])

        if len(car_state) > LINEAR_VEL_X_IDX:
            v_x = float(car_state[LINEAR_VEL_X_IDX])
            v_y = float(car_state[LINEAR_VEL_Y_IDX])
        else:
            v_x = (pose_x - self.last_x) / delta_time
            v_y = (pose_y - self.last_y) / delta_time

        if len(car_state) > ANGULAR_VEL_Z_IDX:
            av_z = float(car_state[ANGULAR_VEL_Z_IDX])
        else:
            delta_yaw = yaw - self.last_yaw
            delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
            av_z = delta_yaw / delta_time

        a_x_car, a_y_car = self.body_acceleration_at_imu(
            v_x,
            v_y,
            av_z,
            self.last_vx,
            self.last_vy,
            self.last_avz,
            self.imu_dx_cog,
            self.imu_dy_cog,
            delta_time,
        )

        # 2D simulation: level car, gravity on z only.
        a_z = 9.81
        gyro_z = av_z
        quat_w, quat_x, quat_y, quat_z = self._yaw_to_quaternion(yaw)

        noise = self.noise_level
        imu_data = np.zeros(IMUUtilities.IMU_DATA_DIM)
        imu_data[IMUUtilities.ACCEL_X_IDX] = a_x_car + np.random.normal(0, noise)
        imu_data[IMUUtilities.ACCEL_Y_IDX] = a_y_car + np.random.normal(0, noise)
        imu_data[IMUUtilities.ACCEL_Z_IDX] = a_z + np.random.normal(0, noise)
        imu_data[IMUUtilities.GYRO_X_IDX] = np.random.normal(0, noise)
        imu_data[IMUUtilities.GYRO_Y_IDX] = np.random.normal(0, noise)
        imu_data[IMUUtilities.GYRO_Z_IDX] = gyro_z + np.random.normal(0, noise)
        imu_data[IMUUtilities.EULER_ROLL_IDX] = 0.0
        imu_data[IMUUtilities.EULER_PITCH_IDX] = 0.0
        imu_data[IMUUtilities.EULER_YAW_IDX] = yaw
        imu_data[IMUUtilities.QUAT_W_IDX] = quat_w
        imu_data[IMUUtilities.QUAT_X_IDX] = quat_x
        imu_data[IMUUtilities.QUAT_Y_IDX] = quat_y
        imu_data[IMUUtilities.QUAT_Z_IDX] = quat_z

        self.last_x = pose_x
        self.last_y = pose_y
        self.last_yaw = yaw
        self.last_vx = v_x
        self.last_vy = v_y
        self.last_avz = av_z

        return IMUUtilities.imu_array_to_dict(imu_data)

    @staticmethod
    def _yaw_to_quaternion(yaw):
        half_yaw = yaw * 0.5
        return math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)

    def prime(self, car_state):
        """Seed history so the next update returns a real sample (no zero first reading)."""
        self._set_last_kinematics(car_state)
        self.first_update = False

    def reset(self):
        self._reset_state()
