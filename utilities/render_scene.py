"""Backend-agnostic render scene: write geometry once, adapt to pygame/web/ROS.

Producers (planners, CarSystem) should prefer::

    scene.set_static_points("waypoints", xy, color=(180, 180, 180))
    scene.set_dynamic_points("lidar", xy, color=(255, 0, 255))
    scene.set_trajectories("mpc.rollouts", trajs, color=(250, 25, 30))

Adapters turn ``snapshot()`` into web_overlay / pygame draws / ROS markers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import numpy as np

Lifetime = Literal["static", "dynamic"]
Kind = Literal["points", "polyline", "trajectories", "poses", "sprite", "labels"]


@dataclass
class RenderLayer:
    name: str
    kind: Kind
    data: Any
    lifetime: Lifetime = "dynamic"
    color: Optional[Tuple[int, int, int]] = None
    style: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "data": self.data,
            "lifetime": self.lifetime,
            "color": list(self.color) if self.color is not None else None,
            "style": dict(self.style),
        }


class RenderScene:
    """Named geometry layers shared by all render backends."""

    def __init__(self) -> None:
        self._layers: Dict[str, RenderLayer] = {}
        self.labels: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}

    def clear(self, lifetime: Optional[Lifetime] = None) -> None:
        if lifetime is None:
            self._layers.clear()
            self.labels.clear()
            self.meta.clear()
            return
        self._layers = {
            name: layer
            for name, layer in self._layers.items()
            if layer.lifetime != lifetime
        }
        if lifetime == "dynamic":
            # Labels are treated as dynamic telemetry.
            self.labels.clear()

    def clear_dynamic(self) -> None:
        self.clear(lifetime="dynamic")

    def remove(self, name: str) -> None:
        self._layers.pop(str(name), None)

    def get(self, name: str) -> Optional[RenderLayer]:
        return self._layers.get(str(name))

    def set(
        self,
        name: str,
        kind: Kind,
        data: Any,
        *,
        lifetime: Lifetime = "dynamic",
        color: Optional[Iterable[int]] = None,
        **style: Any,
    ) -> None:
        key = str(name)
        color_t: Optional[Tuple[int, int, int]] = None
        if color is not None:
            c = [int(v) for v in list(color)[:3]]
            if len(c) == 3:
                color_t = (c[0], c[1], c[2])
        if data is None:
            self._layers.pop(key, None)
            return
        self._layers[key] = RenderLayer(
            name=key,
            kind=kind,
            data=data,
            lifetime=lifetime,
            color=color_t,
            style=dict(style),
        )

    def set_static_points(
        self,
        name: str,
        xy: Any,
        *,
        color: Optional[Iterable[int]] = None,
        **style: Any,
    ) -> None:
        self.set(name, "points", xy, lifetime="static", color=color, **style)

    def set_dynamic_points(
        self,
        name: str,
        xy: Any,
        *,
        color: Optional[Iterable[int]] = None,
        **style: Any,
    ) -> None:
        self.set(name, "points", xy, lifetime="dynamic", color=color, **style)

    def set_polylines(
        self,
        name: str,
        lines: Any,
        *,
        lifetime: Lifetime = "dynamic",
        color: Optional[Iterable[int]] = None,
        **style: Any,
    ) -> None:
        self.set(name, "polyline", lines, lifetime=lifetime, color=color, **style)

    def set_trajectories(
        self,
        name: str,
        trajs: Any,
        *,
        lifetime: Lifetime = "dynamic",
        color: Optional[Iterable[int]] = None,
        **style: Any,
    ) -> None:
        self.set(name, "trajectories", trajs, lifetime=lifetime, color=color, **style)

    def set_poses(
        self,
        name: str,
        poses: Any,
        *,
        lifetime: Lifetime = "dynamic",
        color: Optional[Iterable[int]] = None,
        **style: Any,
    ) -> None:
        self.set(name, "poses", poses, lifetime=lifetime, color=color, **style)

    def set_sprite(
        self,
        name: str,
        data: Any,
        *,
        lifetime: Lifetime = "dynamic",
        color: Optional[Iterable[int]] = None,
        **style: Any,
    ) -> None:
        self.set(name, "sprite", data, lifetime=lifetime, color=color, **style)

    def set_labels(self, labels: Mapping[str, Any], *, replace: bool = False) -> None:
        if replace:
            self.labels = {}
        for key, value in labels.items():
            self.labels[str(key)] = value

    def layers(
        self,
        *,
        lifetime: Optional[Lifetime] = None,
        kind: Optional[Kind] = None,
    ) -> List[RenderLayer]:
        out = list(self._layers.values())
        if lifetime is not None:
            out = [layer for layer in out if layer.lifetime == lifetime]
        if kind is not None:
            out = [layer for layer in out if layer.kind == kind]
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Plain dict for adapters (layers keyed by name)."""
        return {
            "layers": {name: layer.to_dict() for name, layer in self._layers.items()},
            "labels": dict(self.labels),
            "meta": deepcopy(self.meta),
        }
