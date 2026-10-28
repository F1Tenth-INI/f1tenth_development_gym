from __future__ import annotations

from typing import Any, Optional

import numpy as np

from f110_sim.envs.collision_models import get_vertices
from f110_sim.envs.laser_models import ScanSimulator2D, ray_cast
from utilities.Settings import Settings
from utilities.car_files.vehicle_parameters import VehicleParameters, car_parameter_file_for_agent
from utilities.state_utilities import POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX


class LidarSimulator:
    """Simulate 2D lidar scans from vehicle pose and multi-agent environment state."""

    _scan_simulator: Optional[ScanSimulator2D] = None
    _scan_angles: Optional[np.ndarray] = None

    def __init__(
        self,
        seed: int = 12345,
        num_beams: int = 1080,
        fov: float = 4.7,
        car_parameter_file: Optional[str] = None,
    ):
        self.seed = seed
        self.num_beams = num_beams
        self.fov = fov
        self.scan_rng = np.random.default_rng(seed=self.seed)

        param_file = car_parameter_file or Settings.ENV_CAR_PARAMETER_FILE
        self.params = VehicleParameters(param_file).to_dict()
        ego_params = VehicleParameters(car_parameter_file_for_agent(True))
        opponent_params = VehicleParameters(car_parameter_file_for_agent(False))
        self.ego_length = float(ego_params.length)
        self.ego_width = float(ego_params.width)
        self.opponent_length = float(opponent_params.length)
        self.opponent_width = float(opponent_params.width)
        self.scan_placeholder = np.full((num_beams,), 30.0, dtype=np.float32)

        self._ensure_scan_tables_initialized(num_beams, fov)

    @classmethod
    def _ensure_scan_tables_initialized(cls, num_beams: int, fov: float) -> None:
        if cls._scan_simulator is not None:
            return

        cls._scan_simulator = ScanSimulator2D(num_beams, fov)
        scan_ang_incr = cls._scan_simulator.get_increment()
        cls._scan_angles = np.array(
            [-fov / 2.0 + i * scan_ang_incr for i in range(num_beams)],
            dtype=np.float64,
        )

    @property
    def scan_simulator(self) -> ScanSimulator2D:
        return self._scan_simulator

    def set_map(self, map_path: str, map_ext: str) -> None:
        if Settings.BLANK_MAP:
            return
        self._scan_simulator.set_map(map_path, map_ext)

    def reset_rng(self, seed: Optional[int] = None) -> None:
        self.scan_rng = np.random.default_rng(seed=seed if seed is not None else self.seed)

    def opponent_poses_from_env(self, env_state: dict[str, Any], driver_index: int):
        car_states = env_state.get("car_states", [])
        poses = []
        sizes = []
        for i, state in enumerate(car_states):
            if i == driver_index:
                continue
            state = np.asarray(state)
            poses.append(
                np.array([state[POSE_X_IDX], state[POSE_Y_IDX], state[POSE_THETA_IDX]], dtype=np.float64)
            )
            sizes.append(self._body_size_for_agent(i))
        return poses, sizes

    def _body_size_for_agent(self, agent_index: int) -> tuple[float, float]:
        """Ego is agent 0. Every other simulated car uses the opponent car file."""
        if agent_index == 0:
            return self.ego_length, self.ego_width
        return self.opponent_length, self.opponent_width

    def from_env_state(
        self,
        driver_index: int,
        env_state: dict[str, Any],
        *,
        simulate: bool = True,
    ) -> np.ndarray:
        """Build a lidar scan for one agent from the environment snapshot."""
        if not simulate:
            return self.scan_placeholder.copy()

        car_states = env_state.get("car_states", [])
        if driver_index >= len(car_states):
            raise IndexError(
                f"driver_index {driver_index} out of range for env_state car_states (len={len(car_states)})"
            )

        car_state = np.asarray(car_states[driver_index])
        opponent_poses, opponent_sizes = self.opponent_poses_from_env(env_state, driver_index)
        return self.compute(car_state, opponent_poses, opponent_sizes)

    def compute(
        self,
        car_state: np.ndarray,
        opponent_poses: list[np.ndarray],
        opponent_sizes: Optional[list[tuple[float, float]]] = None,
    ) -> np.ndarray:
        car_state = np.asarray(car_state)
        pose = np.array(
            [car_state[POSE_X_IDX], car_state[POSE_Y_IDX], car_state[POSE_THETA_IDX]],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(pose)):
            return self.scan_placeholder.copy()

        scan = self._scan_simulator.scan(pose, self.scan_rng)
        return self._ray_cast_agents(pose, scan, opponent_poses, opponent_sizes)

    def _ray_cast_agents(
        self,
        pose: np.ndarray,
        scan: np.ndarray,
        opponent_poses: list[np.ndarray],
        opponent_sizes: Optional[list[tuple[float, float]]] = None,
    ) -> np.ndarray:
        new_scan = scan
        for index, opp_pose in enumerate(opponent_poses):
            if opponent_sizes is not None and index < len(opponent_sizes):
                length, width = opponent_sizes[index]
            else:
                length, width = self.opponent_length, self.opponent_width
            opp_vertices = get_vertices(opp_pose, length, width)
            new_scan = ray_cast(pose, new_scan, self._scan_angles, opp_vertices)
        return new_scan
