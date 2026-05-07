import importlib.util
from pathlib import Path
import warnings


_LEGACY_PYGLET_RENDERER_PATH = Path(__file__).resolve().parents[1] / "rendering.py"


def _load_legacy_env_renderer():
    if not _LEGACY_PYGLET_RENDERER_PATH.exists():
        from sim.f110_sim.envs.rendering.pygame_rendering import EnvRenderer as PygameRendererAdapter
        warnings.warn(
            f"Pyglet renderer source not found at {_LEGACY_PYGLET_RENDERER_PATH}. "
            "Falling back to pygame renderer adapter.",
            RuntimeWarning,
            stacklevel=2,
        )
        return PygameRendererAdapter
    spec = importlib.util.spec_from_file_location("legacy_pyglet_rendering", _LEGACY_PYGLET_RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pyglet renderer from {_LEGACY_PYGLET_RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnvRenderer


class EnvRenderer:
    def __init__(self, *args, **kwargs):
        impl_cls = _load_legacy_env_renderer()
        self._impl = impl_cls(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._impl, item)

