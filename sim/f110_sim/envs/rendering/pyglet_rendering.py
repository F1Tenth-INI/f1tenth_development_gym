"""
Adapter for the legacy pyglet/OpenGL renderer.

The pyglet renderer source lives in `legacy_pyglet.py` (this folder) and is
based on the original f1tenth_gym implementation. It is exposed here under
the name expected by `run_simulation.py` (`EnvRenderer`).

If pyglet is not installed (or the legacy module fails to import for any
other reason), we fall back transparently to the pygame renderer so that
`Settings.RENDER_BACKEND = "pyglet"` does not hard-crash the simulator.
Callers that need the pyglet-specific overlay extras (e.g. the
`render_callback` in `run_simulation.py`) should gate on
`hasattr(renderer, "zoomed_height")` to detect the active backend.
"""

import warnings

try:
    from sim.f110_sim.envs.rendering.legacy_pyglet import EnvRenderer  # noqa: F401
    RENDERER_KIND = "pyglet"
except Exception as exc:  # pyglet missing, OpenGL context unavailable, etc.
    warnings.warn(
        f"Pyglet renderer unavailable ({type(exc).__name__}: {exc}). "
        "Falling back to pygame renderer.",
        RuntimeWarning,
        stacklevel=2,
    )
    from sim.f110_sim.envs.rendering.pygame_rendering import EnvRenderer  # noqa: F401
    RENDERER_KIND = "pygame"
