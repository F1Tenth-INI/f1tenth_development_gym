"""Grip-limited speed along a closed waypoint line.

Shared by the HWM planner and the midline waypoint loader so both apply the same profile.
"""

from __future__ import annotations

import torch


def grip_speed_profile(xy: torch.Tensor, accel: float, max_speed: float) -> torch.Tensor:
    """``(N,)`` fastest speeds along the closed line ``xy`` ``(N, 2)`` whose cornering, braking and
    accelerating together stay within a friction circle of radius ``accel``."""
    points = xy.detach().cpu().double()
    before, after = torch.roll(points, 1, dims=0), torch.roll(points, -1, dims=0)
    a, b, c = (points - before).norm(dim=1), (after - points).norm(dim=1), (after - before).norm(dim=1)
    cross = ((points - before)[:, 0] * (after - points)[:, 1] - (points - before)[:, 1] * (after - points)[:, 0]).abs()
    curvature = 2.0 * cross / (a * b * c).clamp(min=1e-9)  # through the point and its two neighbours
    curvature = torch.maximum(curvature, torch.maximum(torch.roll(curvature, 1), torch.roll(curvature, -1)))
    speed = torch.sqrt(accel / curvature.clamp(min=1e-6)).clamp(max=max_speed).tolist()
    step, kappa, n = b.tolist(), curvature.tolist(), len(speed)

    def longitudinal(v: float, k: float) -> float:
        lateral = min(v * v * k, accel)
        return (accel * accel - lateral * lateral) ** 0.5

    # Twice around in each direction, so the limit carries over the closing segment.
    for i in range(2 * n):
        j, nxt = i % n, (i + 1) % n
        speed[nxt] = min(speed[nxt], (speed[j] ** 2 + 2.0 * longitudinal(speed[j], kappa[j]) * step[j]) ** 0.5)
    for i in range(2 * n, 0, -1):
        j, prev = i % n, (i - 1) % n
        speed[prev] = min(speed[prev], (speed[j] ** 2 + 2.0 * longitudinal(speed[j], kappa[j]) * step[prev]) ** 0.5)
    return torch.tensor(speed, dtype=torch.float32)
