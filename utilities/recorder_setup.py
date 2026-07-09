"""Configure CarSystem CSV recording and optional metric channels."""

from typing import TYPE_CHECKING, Any, Optional

from utilities.Settings import Settings

if TYPE_CHECKING:
    from utilities.car_system import CarSystem

from utilities.Recorder import Recorder


def _cbf_recorder_fields(car_system: "CarSystem") -> dict[str, Any]:
    return {
        "cbf_active": lambda: float((car_system.cbf_info or {}).get("active", False)),
        "cbf_intervention": lambda: float((car_system.cbf_info or {}).get("intervention", 0.0)),
        "cbf_slack": lambda: float((car_system.cbf_info or {}).get("slack", 0.0)),
        "cbf_h_left": lambda: float((car_system.cbf_info or {}).get("h_left", float("inf"))),
        "cbf_h_right": lambda: float((car_system.cbf_info or {}).get("h_right", float("inf"))),
        "cbf_h_speed": lambda: float((car_system.cbf_info or {}).get("h_speed", float("inf"))),
        "cbf_kappa_ahead": lambda: float((car_system.cbf_info or {}).get("kappa_ahead", float("nan"))),
        "cbf_delta_nom": lambda: float((car_system.cbf_info or {}).get("delta_nom", float("nan"))),
        "cbf_delta_safe": lambda: float((car_system.cbf_info or {}).get("delta_safe", float("nan"))),
        "cbf_delta_correction": lambda: float((car_system.cbf_info or {}).get("delta_safe", float("nan")))
        - float((car_system.cbf_info or {}).get("delta_nom", float("nan"))),
        "cbf_accel_nom": lambda: float((car_system.cbf_info or {}).get("accel_nom", float("nan"))),
        "cbf_accel_safe": lambda: float((car_system.cbf_info or {}).get("accel_safe", float("nan"))),
        "cbf_accel_correction": lambda: float((car_system.cbf_info or {}).get("accel_safe", float("nan")))
        - float((car_system.cbf_info or {}).get("accel_nom", float("nan"))),
    }


def init_car_recorder(
    car_system: "CarSystem", recorder_dict: Optional[dict[str, Any]] = None
) -> None:
    """Create and configure the CSV recorder on ``car_system`` when enabled."""
    if recorder_dict is None:
        recorder_dict = {}

    car_system.recorder = None
    if not (Settings.SAVE_RECORDINGS and car_system.save_recordings):
        return

    car_system.recorder = Recorder(driver=car_system)
    car_system.recorder.dict_data_to_save_basic.update(
        {
            "nearest_wpt_idx": lambda: car_system.waypoint_utils.nearest_waypoint_index,
            "reward": lambda: car_system.reward,
        }
    )

    if car_system.cbf_safety_filter is not None:
        car_system.recorder.dict_data_to_save_basic.update(_cbf_recorder_fields(car_system))

    car_system.recorder.dict_data_to_save_basic.update(recorder_dict)

    if car_system.virtual_opponents is not None:
        from utilities.recording_replay import get_virtual_opponent_recording_dict

        car_system.recorder.dict_data_to_save_basic.update(
            get_virtual_opponent_recording_dict(
                car_system, len(car_system.virtual_opponents.opponents)
            )
        )

    if Settings.FORGE_HISTORY and hasattr(car_system, "history_forger"):
        car_system.recorder.dict_data_to_save_basic.update(
            {
                "forged_history_applied": lambda: car_system.history_forger.forged_history_applied,
            }
        )

    if Settings.SAVE_STATE_METRICS:
        from utilities.StateMetricCalculator import StateMetricCalculator

        car_system.state_metric_calculator = StateMetricCalculator(
            environment_name="Car",
            initial_environment_attributes={
                "next_waypoints": car_system.waypoint_utils.next_waypoints,
            },
            recorder_base_dict=car_system.recorder.dict_data_to_save_basic,
        )
