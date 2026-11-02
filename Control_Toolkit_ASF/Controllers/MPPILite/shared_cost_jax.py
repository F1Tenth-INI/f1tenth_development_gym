"""Cost and wall-map utilities shared by the JAX MPPI and RPGD planners.

Both planners optimize the same stage, smoothness and terminal costs. The numeric
weights stay in each planner: they are tuned separately and passed in as ``w``.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from utilities.Settings import Settings
from utilities.map_scale import scale_map_metadata
from utilities.state_utilities import LINEAR_VEL_X_IDX, POSE_THETA_IDX, POSE_X_IDX, POSE_Y_IDX
from utilities.waypoint_utils import (
    WP_D_LEFT_IDX,
    WP_D_RIGHT_IDX,
    WP_PSI_IDX,
    WP_VX_IDX,
    WP_X_IDX,
    WP_Y_IDX,
)


def load_wall_map(car_length, car_width, device):
    """Wall distance grid of the current map, built like the simulator's collision check (laser_models.check_body_map_collision)."""
    yaml_path = Settings.MAP_CONFIG_FILE
    with open(yaml_path, 'r') as f:
        metadata = scale_map_metadata(yaml.safe_load(f))
    image = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), metadata['image'])).transpose(Image.FLIP_TOP_BOTTOM))
    if image.ndim == 3:
        image = image[..., 0]
    resolution = float(metadata['resolution'])
    distance = resolution * ndimage.distance_transform_edt(image.astype(np.float64) > 128.0)
    if Settings.BLANK_MAP:
        distance = np.full_like(distance, 1e3)
    origin = metadata['origin']
    half_l, half_w = 0.5 * car_length, 0.5 * car_width
    with jax.default_device(device):
        return {
            'distance': jnp.array(distance, dtype=jnp.float32),  # row 0 = lowest y
            'origin': jnp.array([origin[0], origin[1]], dtype=jnp.float32),
            'rotation': jnp.array([np.cos(origin[2]), np.sin(origin[2])], dtype=jnp.float32),
            'resolution': jnp.float32(resolution),
            # Body corners in the car frame: rl, rr, fr, fl
            'corners': jnp.array([[-half_l, half_w], [-half_l, -half_w], [half_l, -half_w], [half_l, half_w]], dtype=jnp.float32),
        }

@jax.jit
def wall_distance_jax(state, wall_map):
    """Distance from the closest body corner to a wall. Points outside the map count as in the wall."""
    c, s = jnp.cos(state[POSE_THETA_IDX]), jnp.sin(state[POSE_THETA_IDX])
    corners = wall_map['corners']
    x = state[POSE_X_IDX] + c * corners[:, 0] - s * corners[:, 1] - wall_map['origin'][0]
    y = state[POSE_Y_IDX] + s * corners[:, 0] + c * corners[:, 1] - wall_map['origin'][1]
    oc, os_ = wall_map['rotation']
    # Bilinear between cell centres: continuous, so gradient-based planners (RPGD) also see the walls
    u = (x * oc + y * os_) / wall_map['resolution'] - 0.5
    v = (-x * os_ + y * oc) / wall_map['resolution'] - 0.5
    height, width = wall_map['distance'].shape
    inside = (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
    col, row = jnp.floor(u).astype(jnp.int32), jnp.floor(v).astype(jnp.int32)
    fu, fv = u - col, v - row
    grid = lambda r, c: wall_map['distance'][jnp.clip(r, 0, height - 1), jnp.clip(c, 0, width - 1)]
    distance = ((1 - fv) * ((1 - fu) * grid(row, col) + fu * grid(row, col + 1))
                + fv * ((1 - fu) * grid(row + 1, col) + fu * grid(row + 1, col + 1)))
    return jnp.min(jnp.where(inside, distance, 0.0))

@jax.jit
def compute_waypoint_distance_jax(state, waypoints):
    dx = state[POSE_X_IDX] - waypoints[:, WP_X_IDX]
    dy = state[POSE_Y_IDX] - waypoints[:, WP_Y_IDX]
    dist_sq = dx ** 2 + dy ** 2
    min_dist_sq = jnp.min(dist_sq)
    min_idx = jnp.argmin(dist_sq)
    return min_dist_sq, min_idx

@jax.jit
def raceline_errors_jax(state, waypoints):
    """Nearest waypoint, signed lateral offset (left positive) and heading error to the raceline."""
    _, i = compute_waypoint_distance_jax(state, waypoints)
    wp = waypoints[i]
    psi = wp[WP_PSI_IDX]
    dx = state[POSE_X_IDX] - wp[WP_X_IDX]
    dy = state[POSE_Y_IDX] - wp[WP_Y_IDX]
    # Projected on the nearest waypoint's normal
    lateral = -jnp.sin(psi) * dx + jnp.cos(psi) * dy
    heading_error = jnp.arctan2(jnp.sin(state[POSE_THETA_IDX] - psi), jnp.cos(state[POSE_THETA_IDX] - psi))
    return wp, lateral, heading_error

@jax.jit
def cost_function_jax(state, control, waypoints, w, wall_map):
    wp, lateral, heading_error = raceline_errors_jax(state, waypoints)

    # Distance of the car centre to the closer track edge (the simulator truncates the episode a bit beyond it)
    edge_distance = jnp.minimum(wp[WP_D_LEFT_IDX] - lateral, wp[WP_D_RIGHT_IDX] + lateral)
    border_cost = w['border'] * jax.nn.relu(w['border_margin'] - edge_distance) ** 2

    wall_distance = wall_distance_jax(state, wall_map)
    wall_cost = w['wall'] * jax.nn.relu(w['wall_margin'] - wall_distance) ** 2
    crash_cost = w['crash'] * (wall_distance < w['crash_margin'])

    v = state[LINEAR_VEL_X_IDX]
    target_speed = w['speed_factor'] * wp[WP_VX_IDX]
    speed_cost = w['speed'] * (v - target_speed) ** 2
    progress_cost = -w['progress'] * v * jnp.cos(heading_error)

    tracking_cost = w['lateral'] * lateral ** 2 + w['heading'] * heading_error ** 2
    control_cost = w['angular_control'] * jnp.abs(control[0]) + w['translational_control'] * jnp.abs(control[1])

    return tracking_cost + speed_cost + progress_cost + border_cost + wall_cost + crash_cost + control_cost

@jax.jit
def terminal_cost_jax(state, waypoints, w):
    """Soft version of the acados MPC terminal set: near the raceline, heading within terminal_max_heading
    and not faster than terminal_speed_factor * waypoint speed, so the car can still follow the raceline after the horizon."""
    wp, lateral, heading_error = raceline_errors_jax(state, waypoints)
    tracking_cost = w['terminal_lateral'] * lateral ** 2 + w['terminal_heading'] * heading_error ** 2
    heading_bound_cost = w['terminal_heading_bound'] * jax.nn.relu(jnp.abs(heading_error) - w['terminal_max_heading']) ** 2
    speed_cost = w['terminal_speed'] * jax.nn.relu(state[LINEAR_VEL_X_IDX] - w['terminal_speed_factor'] * wp[WP_VX_IDX]) ** 2
    return tracking_cost + heading_bound_cost + speed_cost

@jax.jit
def cost_function_sequence_jax(state_sequence, control_sequence, waypoints, w, wall_map):
    cost_fn = lambda s, u: cost_function_jax(s, u, waypoints, w, wall_map)
    standard_costs = jax.vmap(cost_fn)(state_sequence, control_sequence)

    # Intra-horizon smoothness: penalize sudden control changes within the sequence
    control_diff = control_sequence[1:] - control_sequence[:-1]
    smoothness_costs = (w['angular_smoothness'] * control_diff[:, 0] ** 2
                        + w['translational_smoothness'] * control_diff[:, 1] ** 2)
    costs = standard_costs.at[1:].add(smoothness_costs)
    return costs.at[-1].add(terminal_cost_jax(state_sequence[-1], waypoints, w))

@jax.jit
def cost_function_batch_sequence_jax(state_batch_sequence, Q_batch_sequence, waypoints, w, wall_map):
    return jax.vmap(lambda states, controls: cost_function_sequence_jax(states, controls, waypoints, w, wall_map))(
        state_batch_sequence, Q_batch_sequence)
