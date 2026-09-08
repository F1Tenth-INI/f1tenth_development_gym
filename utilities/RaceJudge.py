"""Motorsport-style incident judge for ego vs opponent contact.

Default rule: the car that is behind along the raceline is responsible.
That covers rear-ends and most failed overtaking dives.

Exception: if the cars are overlapping along-track and the car ahead moved
laterally into the trailer over the last few steps ("closed the door"), the
car ahead is at fault.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Mapping, Optional, Sequence

import numpy as np

from utilities.state_utilities import (
    POSE_THETA_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
)
from utilities.waypoint_utils import WP_S_IDX


FAULT_EGO = "ego"
FAULT_OPPONENT = "opponent"
FAULT_NONE = "none"

REASON_REAR_END = "rear_end"
REASON_SIDE_TRAILING = "side_contact_trailing"
REASON_CLOSED_DOOR = "closed_door"
REASON_WALL = "wall"
REASON_LEAVE_TRACK = "leave_track"
REASON_SPIN = "spin"
REASON_CLEAN_PASS = "clean_pass"
REASON_UNKNOWN = "unknown"


class RaceJudge:
    HISTORY_STEPS = 12
    # |along-track gap| below this is treated as overlapping / alongside.
    ALONGSIDE_GAP_FRAC = 0.6
    # Opponent must close this much more laterally than ego to flip fault.
    DOOR_CLOSE_MARGIN_M = 0.05

    def __init__(self) -> None:
        self._history: dict[int, deque] = {}

    def reset(self) -> None:
        self._history.clear()

    def update(self, controller_obs: dict, *, step: int = 0) -> None:
        """Store relative geometry for door-closing detection."""
        snapshots = self._slot_snapshots(controller_obs, step=step)
        live = set()
        for snapshot in snapshots:
            slot = int(snapshot["slot"])
            live.add(slot)
            history = self._history.setdefault(slot, deque(maxlen=self.HISTORY_STEPS))
            history.append(snapshot)
        for slot in list(self._history):
            if slot not in live:
                self._history.pop(slot, None)

    def judge_crash(self, controller_obs: dict, kinds: Sequence[str]) -> dict:
        """Return fault fields for a terminating incident."""
        kind_set = {str(kind) for kind in kinds}
        if "opponent" in kind_set:
            return self.judge_opponent_collision(controller_obs)
        if "wall" in kind_set:
            return self._solo_verdict(REASON_WALL, controller_obs)
        if "leave_track" in kind_set:
            return self._solo_verdict(REASON_LEAVE_TRACK, controller_obs)
        if "spin" in kind_set:
            return self._solo_verdict(REASON_SPIN, controller_obs)
        return {
            "fault": FAULT_EGO,
            "fault_reason": REASON_UNKNOWN,
            "slot": "",
            "gap_m": "",
            "range_m": "",
            "left_m": "",
            "opponent_wp": "",
        }

    def judge_opponent_collision(self, controller_obs: dict) -> dict:
        """Who caused ego–opponent contact. Default: the car behind."""
        snapshot = self._closest_slot_snapshot(controller_obs)
        if snapshot is None:
            return {
                "fault": FAULT_EGO,
                "fault_reason": REASON_UNKNOWN,
                "slot": "",
                "gap_m": "",
                "range_m": "",
                "left_m": "",
                "opponent_wp": "",
            }

        gap_m = float(snapshot["gap_m"])
        car_length = float(snapshot["car_length_m"])
        alongside = abs(gap_m) <= self.ALONGSIDE_GAP_FRAC * max(car_length, 1e-3)

        behind = FAULT_EGO if gap_m >= 0.0 else FAULT_OPPONENT
        reason = REASON_SIDE_TRAILING if alongside else REASON_REAR_END
        fault = behind

        if alongside and self._ahead_closed_the_door(snapshot, behind):
            fault = FAULT_OPPONENT if behind == FAULT_EGO else FAULT_EGO
            reason = REASON_CLOSED_DOOR

        return {
            "fault": fault,
            "fault_reason": reason,
            "slot": int(snapshot["slot"]),
            "gap_m": gap_m,
            "range_m": float(snapshot["range_m"]),
            "left_m": float(snapshot["left_m"]),
            "opponent_wp": int(snapshot["opponent_wp"]),
            "ego_x": float(snapshot["ego_x"]),
            "ego_y": float(snapshot["ego_y"]),
            "ego_theta": float(snapshot["ego_theta"]),
            "opponent_x": float(snapshot["opponent_x"]),
            "opponent_y": float(snapshot["opponent_y"]),
            "opponent_theta": float(snapshot["opponent_theta"]),
        }

    @staticmethod
    def overtake_verdict() -> dict:
        return {"fault": FAULT_NONE, "fault_reason": REASON_CLEAN_PASS}

    @staticmethod
    def summarize(overtakes: Sequence[Mapping], crashes: Sequence[Mapping]) -> dict:
        """Counts for CSV / plot / console scorecard."""
        kind_counts: Counter[str] = Counter()
        opponent_fault_counts: Counter[str] = Counter()
        for crash in crashes:
            kinds = str(crash.get("kind", "") or "").split("+")
            primary = kinds[0] if kinds and kinds[0] else "other"
            kind_counts[primary] += 1
            if "opponent" in kinds:
                fault = str(crash.get("fault", "") or FAULT_EGO)
                opponent_fault_counts[fault] += 1

        n_overtakes = len(overtakes)
        n_crashes = len(crashes)
        n_opp = int(kind_counts.get("opponent", 0))
        ego_fault_hits = int(opponent_fault_counts.get(FAULT_EGO, 0))
        opponent_fault_hits = int(opponent_fault_counts.get(FAULT_OPPONENT, 0))
        return {
            "overtakes": n_overtakes,
            "crashes": n_crashes,
            "opponent_crashes": n_opp,
            "ego_fault_opponent_crashes": ego_fault_hits,
            "opponent_fault_opponent_crashes": opponent_fault_hits,
            "wall_crashes": int(kind_counts.get("wall", 0)),
            "leave_track_crashes": int(kind_counts.get("leave_track", 0)),
            "spin_crashes": int(kind_counts.get("spin", 0)),
            "clean_passes": n_overtakes,
        }

    @staticmethod
    def format_summary(stats: Mapping) -> str:
        return (
            f"{int(stats.get('overtakes', 0))} overtakes, "
            f"{int(stats.get('crashes', 0))} crashes "
            f"(opp {int(stats.get('opponent_crashes', 0))}: "
            f"ego-fault {int(stats.get('ego_fault_opponent_crashes', 0))} / "
            f"opp-fault {int(stats.get('opponent_fault_opponent_crashes', 0))}; "
            f"wall {int(stats.get('wall_crashes', 0))}, "
            f"off-track {int(stats.get('leave_track_crashes', 0))}, "
            f"spin {int(stats.get('spin_crashes', 0))})"
        )

    def _ahead_closed_the_door(self, snapshot: dict, behind: str) -> bool:
        """True if the lead car moved into the trailer more than the trailer steered in."""
        history = list(self._history.get(int(snapshot["slot"]), ()))
        if len(history) < 3:
            return False
        older = history[0]
        left_now = float(snapshot["left_m"])
        if not np.isfinite(left_now) or abs(left_now) < 1e-4:
            return False

        ego_now = np.array([snapshot["ego_x"], snapshot["ego_y"]], dtype=np.float64)
        opp_now = np.array([snapshot["opponent_x"], snapshot["opponent_y"]], dtype=np.float64)
        ego_then = np.array([older["ego_x"], older["ego_y"]], dtype=np.float64)
        opp_then = np.array([older["opponent_x"], older["opponent_y"]], dtype=np.float64)
        if not np.all(np.isfinite(np.concatenate([ego_now, opp_now, ego_then, opp_then]))):
            return False

        theta = float(snapshot["ego_theta"])
        left_axis = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
        d_ego = ego_now - ego_then
        d_opp = opp_now - opp_then
        side = float(np.sign(left_now))
        opp_closes = float(-side * np.dot(d_opp, left_axis))
        ego_closes = float(side * np.dot(d_ego, left_axis))
        lead_closes = opp_closes if behind == FAULT_EGO else ego_closes
        trailer_closes = ego_closes if behind == FAULT_EGO else opp_closes
        return lead_closes > trailer_closes + self.DOOR_CLOSE_MARGIN_M

    def _closest_slot_snapshot(self, controller_obs: dict) -> Optional[dict]:
        snapshots = self._slot_snapshots(controller_obs)
        if not snapshots:
            return None
        return min(snapshots, key=lambda item: float(item["range_m"]))

    def _slot_snapshots(self, controller_obs: dict, *, step: int = 0) -> list[dict]:
        vo_states = np.asarray(
            controller_obs.get("virtual_opponent_states", []), dtype=np.float64
        )
        if vo_states.ndim == 1:
            vo_states = vo_states.reshape(1, -1)
        poses = np.asarray(controller_obs.get("virtual_opponent_poses", []), dtype=np.float64)
        vo_wps = np.asarray(
            controller_obs.get("virtual_opponent_waypoint_indices", []), dtype=np.int32
        )
        car_state = np.asarray(controller_obs.get("car_state", []), dtype=np.float64)
        if (
            vo_states.ndim != 2
            or vo_states.shape[0] == 0
            or vo_states.shape[1] < 3
            or car_state.size <= POSE_Y_IDX
        ):
            return []

        ego_wp = int(controller_obs.get("ego_waypoint_index", -1) or -1)
        waypoints = controller_obs.get("waypoints")
        track_length_m = self._track_length_m(waypoints)
        car_length_m = self._car_length_m()
        n_slots = vo_states.shape[0]
        snapshots = []
        for slot in range(n_slots):
            present = float(vo_states[slot, 0]) >= 0.5
            if not present:
                continue
            forward_m = float(vo_states[slot, 1])
            left_m = float(vo_states[slot, 2])
            range_m = float(np.hypot(forward_m, left_m))
            opponent_wp = int(vo_wps[slot]) if slot < vo_wps.size else -1
            if (
                ego_wp >= 0
                and opponent_wp >= 0
                and waypoints is not None
                and track_length_m > 0.0
            ):
                gap_m = self._signed_along_track_gap_m(
                    waypoints, ego_wp, opponent_wp, track_length_m
                )
            else:
                gap_m = forward_m
            opponent_x = opponent_y = opponent_theta = float("nan")
            if poses.ndim == 2 and poses.shape[0] > slot and poses.shape[1] >= 2:
                opponent_x = float(poses[slot, 0])
                opponent_y = float(poses[slot, 1])
                if poses.shape[1] >= 3:
                    opponent_theta = float(poses[slot, 2])
            snapshots.append(
                {
                    "slot": slot,
                    "step": int(step),
                    "present": True,
                    "forward_m": forward_m,
                    "left_m": left_m,
                    "range_m": range_m,
                    "gap_m": float(gap_m),
                    "car_length_m": car_length_m,
                    "ego_wp": ego_wp,
                    "opponent_wp": opponent_wp,
                    "ego_x": float(car_state[POSE_X_IDX]),
                    "ego_y": float(car_state[POSE_Y_IDX]),
                    "ego_theta": (
                        float(car_state[POSE_THETA_IDX])
                        if car_state.size > POSE_THETA_IDX
                        else 0.0
                    ),
                    "opponent_x": opponent_x,
                    "opponent_y": opponent_y,
                    "opponent_theta": opponent_theta,
                }
            )
        return snapshots

    def _solo_verdict(self, reason: str, controller_obs: dict) -> dict:
        car_state = np.asarray(controller_obs.get("car_state", []), dtype=np.float64)
        return {
            "fault": FAULT_EGO,
            "fault_reason": reason,
            "slot": "",
            "gap_m": "",
            "range_m": "",
            "left_m": "",
            "opponent_wp": "",
            "ego_x": float(car_state[POSE_X_IDX]) if car_state.size > POSE_X_IDX else float("nan"),
            "ego_y": float(car_state[POSE_Y_IDX]) if car_state.size > POSE_Y_IDX else float("nan"),
            "ego_theta": (
                float(car_state[POSE_THETA_IDX]) if car_state.size > POSE_THETA_IDX else float("nan")
            ),
        }

    @staticmethod
    def _car_length_m() -> float:
        try:
            from utilities.virtual_opponents import get_ego_car_dimensions

            length, _width = get_ego_car_dimensions()
            return float(length)
        except Exception:
            return 0.38

    @staticmethod
    def _track_length_m(waypoints) -> float:
        if waypoints is None:
            return 0.0
        wps = np.asarray(waypoints)
        if wps.ndim != 2 or wps.shape[0] < 2:
            return 0.0
        s = wps[:, WP_S_IDX].astype(np.float64)
        span = float(s[-1] - s[0])
        if span <= 0.0:
            return 0.0
        return span + float(s[-1] - s[-2])

    @staticmethod
    def _signed_along_track_gap_m(
        waypoints, ego_wp: int, opponent_wp: int, track_length_m: float
    ) -> float:
        """Raceline arclength from ego to opponent; positive = opponent ahead."""
        n = int(len(waypoints))
        if n == 0 or track_length_m <= 0.0:
            return 0.0
        s_ego = float(waypoints[int(ego_wp) % n, WP_S_IDX])
        s_opponent = float(waypoints[int(opponent_wp) % n, WP_S_IDX])
        gap = (s_opponent - s_ego) % track_length_m
        if gap > 0.5 * track_length_m:
            gap -= track_length_m
        return float(gap)
