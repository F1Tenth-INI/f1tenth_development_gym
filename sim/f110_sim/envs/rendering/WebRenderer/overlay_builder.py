"""Build the web_overlay dict from a driver's RenderScene / RenderUtils."""

from utilities.render_adapters import WebOverlayAdapter
from utilities.virtual_opponents import get_virtual_opponent_dimensions

# Re-export converters for any callers that imported them from here.
from utilities.render_adapters import (  # noqa: F401
    to_pose_points as _to_pose_points,
    to_track_border_lines as _to_track_border_lines,
    to_trajectory_list as _to_trajectory_list,
    to_xy_points as _to_xy_points,
)


def build_web_overlay(drivers):
    if not drivers:
        return {}
    render_utils = getattr(drivers[0], "render_utils", None)
    if render_utils is None:
        return {}

    driver = drivers[0]
    force_plot_publish = bool((getattr(driver, "obs", None) or {}).get("done"))

    # Pick up attribute assigns (e.g. waypoints = ...) that bypassed update().
    scene = render_utils.sync_scene()

    default_colors = {
        "waypoints": render_utils.waypoint_visualization_color,
        "next_waypoints": render_utils.next_waypoint_visualization_color,
        "next_waypoints_polynomial": render_utils.next_waypoints_polynomial_visualization_color,
        "next_waypoints_alternative": render_utils.next_waypoints_alternative_visualization_color,
        "lidar": render_utils.lidar_visualization_color,
        "gap": render_utils.gap_visualization_color,
        "mppi": render_utils.mppi_visualization_color,
        "optimal": render_utils.optimal_trajectory_visualization_color,
        "target": render_utils.target_point_visualization_color,
        "obstacles": render_utils.obstacle_visualization_color,
        "virtual_opponents": render_utils.virtual_opponent_visualization_color,
        "detected_opponents": getattr(
            render_utils, "detected_opponent_visualization_color", (0, 255, 128)
        ),
        "track_border": render_utils.track_border_visualization_color,
        "history_alt": (255, 255, 0),
        "history_gt": render_utils.gt_history_color,
        "history_prior": render_utils.prior_history_color,
        "history_prior_full": render_utils.prior_full_history_color,
    }

    virtual_opponent_size = None
    if render_utils.virtual_opponents is not None:
        opponent_length, opponent_width = get_virtual_opponent_dimensions()
        virtual_opponent_size = [opponent_width, opponent_length]

    return WebOverlayAdapter.to_overlay(
        scene,
        force_plot_publish=force_plot_publish,
        default_colors=default_colors,
        virtual_opponent_size=virtual_opponent_size,
        car_state=render_utils.car_state,
    )
