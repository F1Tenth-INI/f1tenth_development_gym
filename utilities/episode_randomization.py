"""
Per-episode domain randomization from a YAML spec file.

Set Settings.EPISODE_RANDOMIZATION_FILE to a YAML path. On each episode reset,
matching Settings attributes are sampled and applied.

YAML examples::

    NUMBER_OF_VIRTUAL_OPPONENTS: [0, 1, 2]
    GLOBAL_WAYPOINT_VEL_FACTOR: uniform(0.5, 0.7)
    SURFACE_FRICTION: uniform(0.5, 1.0)
    REVERSE_DIRECTION: [true, false]

- A list samples a random element.
- ``uniform(low, high)`` samples a float (or int when both bounds are integers).
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Optional

import numpy as np
import yaml

from utilities.Settings import Settings

_UNIFORM_RE = re.compile(
    r"^\s*uniform\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*$",
    re.IGNORECASE,
)

_SPEC_CACHE: dict[str, dict[str, Any]] = {}


def _load_spec(path: str) -> dict[str, Any]:
    abs_path = os.path.abspath(os.path.expanduser(path))
    if abs_path not in _SPEC_CACHE:
        with open(abs_path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Episode randomization file must be a YAML mapping: {abs_path}")
        _SPEC_CACHE[abs_path] = loaded
    return _SPEC_CACHE[abs_path]


def clear_spec_cache() -> None:
    """Drop cached YAML specs (useful in tests)."""
    _SPEC_CACHE.clear()


def _parse_bound(raw: str) -> int | float:
    text = raw.strip()
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return float(text)


def _infer_type_from_spec(spec: Any) -> type:
    if isinstance(spec, str):
        match = _UNIFORM_RE.match(spec)
        if match:
            low = _parse_bound(match.group(1))
            high = _parse_bound(match.group(2))
            if isinstance(low, int) and isinstance(high, int):
                return int
            return float
    if isinstance(spec, (list, tuple)) and spec:
        return type(spec[0])
    if isinstance(spec, bool):
        return bool
    if isinstance(spec, int) and not isinstance(spec, bool):
        return int
    if isinstance(spec, float):
        return float
    return str


def _cast_value(value: Any, target_type: type) -> Any:
    if target_type is bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def _sample_from_spec(spec: Any, target_type: type, rng: np.random.Generator) -> Any:
    if isinstance(spec, str):
        match = _UNIFORM_RE.match(spec)
        if match:
            low = _parse_bound(match.group(1))
            high = _parse_bound(match.group(2))
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                if low > high:
                    low, high = high, low
                if target_type is int or (isinstance(low, int) and isinstance(high, int)):
                    return int(rng.integers(int(low), int(high) + 1))
                return float(rng.uniform(float(low), float(high)))
        return _cast_value(spec, target_type)

    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("Empty choice list in episode randomization spec")
        return _cast_value(rng.choice(spec), target_type)

    return _cast_value(spec, target_type)


def _apply_surface_friction_hook(simulation: Any, mu: float) -> None:
    simulation.vehicle_parameters_instance.mu = float(mu)
    if getattr(simulation, "world_sim", None) is not None:
        simulation.world_sim.update_params({})


def _apply_setting_hooks(setting_name: str, value: Any, simulation: Any) -> None:
    if setting_name == "SURFACE_FRICTION" and simulation is not None:
        _apply_surface_friction_hook(simulation, value)
    if setting_name == "MAP_NAME":
        Settings.recalculate_paths()


def apply_episode_randomization(
    simulation: Any = None,
    rng: Optional[np.random.Generator] = None,
    *,
    spec_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Sample values from the YAML spec and write them onto Settings.

    Returns ``{setting_name: sampled_value}``. Empty when no file is configured.
    """
    path = spec_path or getattr(Settings, "EPISODE_RANDOMIZATION_FILE", None)
    if not path:
        return {}

    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Episode randomization file not found: {abs_path}")

    rng = rng or np.random.default_rng()
    spec = _load_spec(abs_path)
    sampled: dict[str, Any] = {}

    for setting_name, entry in spec.items():
        if not hasattr(Settings, setting_name):
            continue
        original = getattr(Settings, setting_name)
        if callable(original):
            continue

        target_type = type(original)
        if target_type is type(None):
            target_type = _infer_type_from_spec(entry)

        value = _sample_from_spec(entry, target_type, rng)
        setattr(Settings, setting_name, value)
        _apply_setting_hooks(setting_name, value, simulation)
        sampled[setting_name] = value

    if sampled and bool(getattr(Settings, "EPISODE_RANDOMIZATION_VERBOSE", False)):
        formatted = ", ".join(f"{key}={value!r}" for key, value in sampled.items())
        print(f"[episode_randomization] {formatted}")

    return sampled


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample episode randomization values once and apply them to Settings.",
    )
    parser.add_argument(
        "--file",
        dest="spec_path",
        default=None,
        help="YAML spec path (default: Settings.EPISODE_RANDOMIZATION_FILE)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sampled values without modifying Settings.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    spec_path = args.spec_path or getattr(Settings, "EPISODE_RANDOMIZATION_FILE", None)
    if not spec_path:
        raise SystemExit("No spec file: set Settings.EPISODE_RANDOMIZATION_FILE or pass --file")

    if args.dry_run:
        abs_path = os.path.abspath(os.path.expanduser(spec_path))
        spec = _load_spec(abs_path)
        rng = np.random.default_rng()
        for setting_name, entry in spec.items():
            if not hasattr(Settings, setting_name):
                continue
            original = getattr(Settings, setting_name)
            if callable(original):
                continue
            target_type = type(original)
            if target_type is type(None):
                target_type = _infer_type_from_spec(entry)
            value = _sample_from_spec(entry, target_type, rng)
            print(f"{setting_name}: {value!r}")
        return 0

    sampled = apply_episode_randomization(spec_path=spec_path)
    for setting_name, value in sampled.items():
        print(f"{setting_name}: {value!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
