import numpy as np

from utilities.state_utilities import POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, STEERING_ANGLE_IDX


def _to_xy_points(data, state_like=False):
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size < 2:
            return None
        if state_like and arr.size > max(POSE_X_IDX, POSE_Y_IDX):
            return [[float(arr[POSE_X_IDX]), float(arr[POSE_Y_IDX])]]
        if arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
            return [[float(p[0]), float(p[1])] for p in arr]
        return [[float(arr[0]), float(arr[1])]]

    flat = arr.reshape(-1, arr.shape[-1])
    if state_like and flat.shape[1] > max(POSE_X_IDX, POSE_Y_IDX):
        return [[float(p[POSE_X_IDX]), float(p[POSE_Y_IDX])] for p in flat]
    if flat.shape[1] >= 2:
        return [[float(p[0]), float(p[1])] for p in flat]
    return None


def _to_trajectory_list(data):
    if data is None:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        pts = _to_xy_points(arr, state_like=True)
        return [pts] if pts else None
    if arr.ndim >= 3:
        trajectories = []
        for traj in arr:
            pts = _to_xy_points(traj, state_like=True)
            if pts:
                trajectories.append(pts)
        return trajectories if trajectories else None
    return None


def build_web_overlay(drivers):
    if not drivers:
        return {}
    render_utils = getattr(drivers[0], "render_utils", None)
    if render_utils is None:
        return {}

    overlay = {
        "waypoints": _to_xy_points(render_utils.waypoints),
        "waypoints_alternative": _to_xy_points(render_utils.waypoints_alternative),
        "next_waypoints": _to_xy_points(render_utils.next_waypoints),
        "next_waypoints_alternative": _to_xy_points(render_utils.next_waypoints_alternative),
        "lidar_border_points": _to_xy_points(render_utils.lidar_border_points),
        "track_border_points": _to_xy_points(render_utils.track_border_points),
        "largest_gap_middle_point": _to_xy_points(render_utils.largest_gap_middle_point),
        "target_point": _to_xy_points(render_utils.target_point),
        "obstacles": _to_xy_points(render_utils.obstacles),
        "past_car_states_alternative": _to_xy_points(render_utils.past_car_states_alternative, state_like=True),
        "past_car_states_gt": _to_xy_points(render_utils.past_car_states_gt, state_like=True),
        "past_car_states_prior": _to_xy_points(render_utils.past_car_states_prior, state_like=True),
        "past_car_states_prior_full": _to_xy_points(render_utils.past_car_states_prior_full, state_like=True),
        "rollout_trajectory": _to_trajectory_list(render_utils.rollout_trajectory),
        "optimal_trajectory": _to_trajectory_list(render_utils.optimal_trajectory),
        "label_dict": dict(render_utils.label_dict),
        "colors": {
            "waypoints": list(render_utils.waypoint_visualization_color),
            "next_waypoints": list(render_utils.next_waypoint_visualization_color),
            "next_waypoints_alternative": list(render_utils.next_waypoints_alternative_visualization_color),
            "lidar": list(render_utils.lidar_visualization_color),
            "gap": list(render_utils.gap_visualization_color),
            "mppi": list(render_utils.mppi_visualization_color),
            "optimal": list(render_utils.optimal_trajectory_visualization_color),
            "target": list(render_utils.target_point_visualization_color),
            "obstacles": list(render_utils.obstacle_visualization_color),
            "track_border": list(render_utils.track_border_visualization_color),
            "history_alt": [255, 255, 0],
            "history_gt": list(render_utils.gt_history_color),
            "history_prior": list(render_utils.prior_history_color),
            "history_prior_full": list(render_utils.prior_full_history_color),
        },
    }

    slowdown = render_utils.emergency_slowdown_sprites
    if slowdown is not None:
        overlay["emergency_slowdown"] = {
            "left_line": _to_xy_points(slowdown.get("left_line")),
            "right_line": _to_xy_points(slowdown.get("right_line")),
            "stop_line": _to_xy_points(slowdown.get("stop_line")),
            "display_position": _to_xy_points(slowdown.get("display_position")),
            "speed_reduction_factor": float(slowdown.get("speed_reduction_factor", 1.0)),
        }

    car_state = render_utils.car_state
    if car_state is not None:
        car_state_arr = np.asarray(car_state)
        if car_state_arr.size > max(POSE_X_IDX, POSE_Y_IDX, POSE_THETA_IDX, STEERING_ANGLE_IDX):
            start_x = float(car_state_arr[POSE_X_IDX])
            start_y = float(car_state_arr[POSE_Y_IDX])
            heading = float(car_state_arr[POSE_THETA_IDX] + car_state_arr[STEERING_ANGLE_IDX])
            overlay["steering_arrow"] = {
                "start": [start_x, start_y],
                "end": [start_x + np.cos(heading), start_y + np.sin(heading)],
            }

    return overlay

