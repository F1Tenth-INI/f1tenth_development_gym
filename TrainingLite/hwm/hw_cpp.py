"""C++ library of the HW planner. Sources in ``cpp/``.

* ``HwPlannerCore``: the whole planner (``Settings.HWM_BACKEND = "cpp"``, see ``hirarchical_planner_cpp.py``).
* ``HwCpp``: only its kernels (Pacejka rollouts, late-crash freeze, drive costs, correlated samples) under the
  Python planner (``HWM_BACKEND = "cpp_kernels"``), to check the C++ code against the Python one.

Everything runs on one persistent thread pool (``HWM_ROLLOUT_THREADS`` threads, the caller included).

The library is compiled on first use with the system ``g++`` for the machine it runs on
(``-march=native`` / ``-mcpu=native``) and cached under ``cpp/build/<hash of sources and flags>/``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import threading
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent / "cpp"
_LOCK = threading.Lock()
_LIB = None

_f32p = ctypes.POINTER(ctypes.c_float)
_f64p = ctypes.POINTER(ctypes.c_double)
_u8p = ctypes.POINTER(ctypes.c_uint8)
_i32p = ctypes.POINTER(ctypes.c_int32)


def _arch_flags() -> list[str]:
    """Code for this CPU. ``HWM_CPP_ARCH`` overrides it (e.g. ``-march=x86-64-v2``: 128-bit vectors only)."""
    if os.environ.get("HWM_CPP_ARCH"):
        return os.environ["HWM_CPP_ARCH"].split()
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return ["-mcpu=native"]
    return ["-march=native"]


def _build() -> Path:
    cxx = os.environ.get("CXX", "g++")
    common = ["-O3", "-std=c++17", "-fPIC", "-pthread", *_arch_flags()]
    # No -ffast-math anywhere: hw_math.h range reductions depend on operation order, the costs on inf / NaN.
    units = {
        "hw_rollout.cpp": common + ["-fno-math-errno", "-fno-trapping-math", "-fopenmp-simd"],
        "hw_costs.cpp": common + ["-ffp-contract=off"],
        "hw_planner.cpp": common + ["-ffp-contract=off"],
    }
    digest = hashlib.sha1()
    for name in sorted(os.listdir(_SRC)):
        if name.endswith((".cpp", ".h")):
            digest.update(name.encode())
            digest.update((_SRC / name).read_bytes())
    digest.update(repr((cxx, units)).encode())
    out_dir = _SRC / "build" / digest.hexdigest()[:16]
    lib = out_dir / "libhwcore.so"
    if lib.is_file():
        return lib
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    for name, flags in units.items():
        obj = out_dir / (name + ".o")
        subprocess.run([cxx, *flags, "-c", str(_SRC / name), "-o", str(obj)], check=True)
        objects.append(str(obj))
    tmp = out_dir / f"libhwcore.so.{os.getpid()}"
    # Plain link: nothing that changes floating point modes process wide.
    subprocess.run([cxx, "-shared", "-pthread", *objects, "-o", str(tmp), "-lm"], check=True)
    os.replace(tmp, lib)
    return lib


def _lib():
    global _LIB
    with _LOCK:
        if _LIB is None:
            lib = ctypes.CDLL(str(_build()))
            i, f, d = ctypes.c_int, ctypes.c_float, ctypes.c_double
            lib.hw_set_threads.argtypes = [i]
            lib.hw_threads.restype = i
            lib.hw_set_map.argtypes = [_f32p, i, i, d, d, d, d, d, _f64p, _f64p]
            lib.hw_rollout.argtypes = [_f32p, i, f, i, f, f, f, i, i, i, _f32p, _f32p, _f32p, _f32p]
            lib.hw_freeze_late.argtypes = [_f32p, i, i, i, f]
            lib.hw_drive_costs.argtypes = [_f32p, i, i, _f32p, _f32p, i, _f32p, i, _f32p, _u8p, _i32p, _f32p, _f32p]
            lib.hw_correlated_samples.argtypes = [_f32p, _f32p, i, i, f, f, f, f, f, _f32p]
            lib.hw_set_blank_map.argtypes = []
            vp = ctypes.c_void_p
            lib.hwp_error.restype = ctypes.c_char_p
            lib.hwp_create.restype = vp
            lib.hwp_create.argtypes = [ctypes.c_char_p, _f32p, i, f, i, i, _f32p, _f32p, _f32p, _f32p, _f32p, _i32p,
                                       _f32p, i, i]
            lib.hwp_destroy.argtypes = [vp]
            lib.hwp_step.argtypes = [vp, _f32p, _f32p]
            lib.hwp_step.restype = i
            lib.hwp_lap.argtypes = [vp, f]
            lib.hwp_episode_done.argtypes = [vp, i]
            lib.hwp_reset.argtypes = [vp]
            for name in ("hwp_table", "hwp_actions", "hwp_off_track_counts"):
                getattr(lib, name).argtypes = [vp]
                getattr(lib, name).restype = _f32p
            lib.hwp_known.argtypes = [vp]
            lib.hwp_known.restype = _u8p
            lib.hwp_frozen.argtypes = [vp]
            lib.hwp_frozen.restype = i
            lib.hwp_recover_mode.argtypes = [vp]
            lib.hwp_recover_mode.restype = i
            lib.hwp_stats.argtypes = [vp, _f64p]
            lib.hwp_clear_stats.argtypes = [vp]
            _LIB = lib
        return _LIB


def set_threads(threads: int) -> int:
    """Threads of the pool shared by everything in the library, the caller included. Returns the count."""
    lib = _lib()
    lib.hw_set_threads(int(threads))
    return int(lib.hw_threads())


def set_map(geometry) -> None:
    """Wall distance grid and body corners of a ``WallGeometry`` (process wide)."""
    grid = _f32(geometry.wall_distance.numpy())
    cx = np.ascontiguousarray(geometry._corner_x.numpy(), dtype=np.float64)
    cy = np.ascontiguousarray(geometry._corner_y.numpy(), dtype=np.float64)
    _lib().hw_set_map(
        _ptr(grid), int(grid.shape[0]), int(grid.shape[1]), float(geometry.origin_x), float(geometry.origin_y),
        float(geometry.origin_cos), float(geometry.origin_sin), float(geometry.resolution),
        _ptr(cx, _f64p), _ptr(cy, _f64p),
    )


def set_blank_map() -> None:
    """No walls: nothing crashes (``Settings.BLANK_MAP``)."""
    _lib().hw_set_blank_map()


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _ptr(a: np.ndarray, kind=_f32p):
    return a.ctypes.data_as(kind)


_MODES = {"ODE:pacejka": 0, "ODE:ks": 1, "ODE:ks_pacejka": 2}


class HwCpp:
    """The planner's numeric kernels. One per planner; the thread pool and the map are process wide."""

    def __init__(self, threads: int, car_params: np.ndarray, ode_model: str, dt: float, substeps: int,
                 denorm, cap: float):
        self.lib = _lib()
        self.lib.hw_set_threads(int(threads))
        if ode_model not in _MODES:
            raise ValueError(f"HWM_BACKEND 'cpp' supports {sorted(_MODES)}, not {ode_model!r}")
        self.car = _f32(car_params).reshape(-1)
        if self.car.size != 32:
            raise ValueError(f"car parameters must have 32 entries, got {self.car.size}")
        self.mode = _MODES[ode_model]
        self.dt, self.substeps = float(dt), int(substeps)
        self.denorm = [float(x) for x in np.asarray(denorm).reshape(-1)]
        self.cap = float(cap)
        self._map = None

    @property
    def threads(self) -> int:
        return int(self.lib.hw_threads())

    def set_map(self, geometry) -> None:
        """Wall distance grid and body corners of a ``WallGeometry``. Once per map."""
        if self._map is geometry.wall_distance:
            return
        set_map(geometry)
        self._map = geometry.wall_distance

    def rollout(self, s0: np.ndarray, actions: np.ndarray, pending: np.ndarray) -> np.ndarray:
        """States ``(B, H + 1, D)`` from ``s0`` ``(B, D)``, ``actions`` ``(B, H, 2)`` and delay queues ``(B, delay, 2)``."""
        s0, actions, pending = _f32(s0), _f32(actions), _f32(pending)
        batch, horizon = int(actions.shape[0]), int(actions.shape[1])
        delay = int(pending.shape[1]) if pending.ndim == 3 else 0
        out = np.empty((batch, horizon + 1, int(s0.shape[1])), dtype=np.float32)
        self.lib.hw_rollout(
            _ptr(self.car), self.mode, self.dt, self.substeps, self.denorm[0], self.denorm[1], self.cap,
            batch, horizon, delay, _ptr(s0), _ptr(actions), _ptr(pending), _ptr(out),
        )
        return out

    def freeze_late(self, trajectories: np.ndarray, end: int, border: float) -> None:
        """``HirarchicalPlanner._freeze_late_crashes`` in place on a C-contiguous float32 ``(B, T, D)`` array."""
        assert trajectories.dtype == np.float32 and trajectories.flags.c_contiguous
        batch, steps = int(trajectories.shape[0]), int(trajectories.shape[1])
        self.lib.hw_freeze_late(_ptr(trajectories), batch, steps, int(end), float(border))

    def drive_costs(self, states: np.ndarray, match_xy: np.ndarray, rows: np.ndarray, count: int,
                    consts, crash_end: int):
        """``(drive, active, late, closeness, depth)`` per rollout, as ``drive_costs_jax.drive_costs``."""
        states, match_xy, rows = _f32(states), _f32(match_xy), _f32(rows)
        batch, steps = int(states.shape[0]), int(states.shape[1])
        c = _f32(consts)
        drive = np.empty(batch, np.float32)
        active = np.empty(batch, np.uint8)
        late = np.empty(batch, np.int32)
        closeness = np.empty(batch, np.float32)
        depth = np.empty(batch, np.float32)
        self.lib.hw_drive_costs(
            _ptr(states), batch, steps, _ptr(match_xy), _ptr(rows), int(count), _ptr(c), int(crash_end),
            _ptr(drive), _ptr(active, _u8p), _ptr(late, _i32p), _ptr(closeness), _ptr(depth),
        )
        return drive, active.astype(bool), late, closeness, depth

    def correlated_samples(self, nominal: np.ndarray, eps: np.ndarray, std, rho: float, lo: float, hi: float):
        """``(B, H, 2)`` samples ``HirarchicalPlanner._sample_mpc_actions`` makes from the noise ``eps``."""
        nominal, eps = _f32(nominal), _f32(eps)
        batch, horizon = int(eps.shape[0]), int(eps.shape[1])
        out = np.empty((batch, horizon, 2), dtype=np.float32)
        self.lib.hw_correlated_samples(
            _ptr(nominal), _ptr(eps), batch, horizon, float(std[0]), float(std[1]), float(rho), float(lo), float(hi),
            _ptr(out),
        )
        return out


STAT_NAMES = ("recovery", "samples", "rollout", "costs", "plan", "window_update", "table_and_seed", "total")


class HwPlannerCore:
    """The C++ planner. ``table`` ``(n, 14)``, ``actions`` ``(n, H, 2)``, ``known`` ``(n,)`` and ``off_track_counts``
    are zero-copy views of its waypoint table."""

    def __init__(self, config: dict, threads: int, car_params, ode_model: str, dt: float, substeps: int,
                 table: np.ndarray, anchor: np.ndarray, tangent: np.ndarray, ortho: np.ndarray, track_field=None):
        self.lib = _lib()
        self.lib.hw_set_threads(int(threads))
        if ode_model not in _MODES:
            raise ValueError(f"HWM_BACKEND 'cpp' supports {sorted(_MODES)}, not {ode_model!r}")
        text = "\n".join(
            f"{k} " + " ".join(repr(float(v)) for v in (val if isinstance(val, (list, tuple)) else [val]))
            for k, val in config.items()
        ).encode()
        self._keep = [_f32(car_params).reshape(-1), _f32(table), _f32(anchor), _f32(tangent), _f32(ortho)]
        n = int(table.shape[0])
        if track_field is not None:
            length, label, direction = track_field
            grids = [_f32(length), np.ascontiguousarray(label, dtype=np.int32), _f32(direction)]
            grid_h, grid_w = int(length.shape[0]), int(length.shape[1])
            tf = (_ptr(grids[0]), _ptr(grids[1], _i32p), _ptr(grids[2]))
        else:
            grids, grid_h, grid_w, tf = [], 0, 0, (None, None, None)
        car, tab, anc, tan, ort = self._keep
        self.handle = self.lib.hwp_create(
            text, _ptr(car), _MODES[ode_model], float(dt), int(substeps), n, _ptr(tab), _ptr(anc), _ptr(tan),
            _ptr(ort), *tf, grid_h, grid_w,
        )
        if not self.handle:
            raise RuntimeError(f"hw_planner: {self.lib.hwp_error().decode()}")
        horizon = int(config["H"])
        self.table = np.ctypeslib.as_array(self.lib.hwp_table(self.handle), shape=(n, 14))
        self.actions = np.ctypeslib.as_array(self.lib.hwp_actions(self.handle), shape=(n, horizon, 2))
        self.known = np.ctypeslib.as_array(self.lib.hwp_known(self.handle), shape=(n,))
        self.off_track_counts = np.ctypeslib.as_array(self.lib.hwp_off_track_counts(self.handle), shape=(n,))
        self._out = np.zeros(4, np.float32)

    def refresh_views(self) -> None:
        """The table arrays are reallocated when a snapshot is restored: point the views at the new ones."""
        n, horizon = self.table.shape[0], self.actions.shape[1]
        self.table = np.ctypeslib.as_array(self.lib.hwp_table(self.handle), shape=(n, 14))
        self.actions = np.ctypeslib.as_array(self.lib.hwp_actions(self.handle), shape=(n, horizon, 2))
        self.known = np.ctypeslib.as_array(self.lib.hwp_known(self.handle), shape=(n,))
        self.off_track_counts = np.ctypeslib.as_array(self.lib.hwp_off_track_counts(self.handle), shape=(n,))

    def step(self, state: np.ndarray) -> np.ndarray:
        """``[angular control, translational control, network steer, network translational]``."""
        s = _f32(state).reshape(-1)
        if self.lib.hwp_step(self.handle, _ptr(s), _ptr(self._out)) != 0:
            raise RuntimeError(f"hw_planner: {self.lib.hwp_error().decode()}")
        return self._out.copy()

    def lap(self, lap_time: float) -> None:
        self.lib.hwp_lap(self.handle, float(lap_time))
        self.refresh_views()

    def episode_done(self, crash: bool) -> None:
        self.lib.hwp_episode_done(self.handle, int(bool(crash)))

    def reset(self) -> None:
        self.lib.hwp_reset(self.handle)

    @property
    def frozen(self) -> bool:
        return bool(self.lib.hwp_frozen(self.handle))

    def stats(self) -> dict:
        out = np.zeros(len(STAT_NAMES) + 1, np.float64)
        self.lib.hwp_stats(self.handle, _ptr(out, _f64p))
        return {**dict(zip(STAT_NAMES, out[:-1].tolist())), "steps": int(out[-1])}

    def clear_stats(self) -> None:
        self.lib.hwp_clear_stats(self.handle)

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.hwp_destroy(handle)
            self.handle = None
