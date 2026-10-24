"""Nested wall-clock profiler for the HWM control step (``Settings.HWM_PROFILE``).

``PROF.section("name")`` is a context manager, ``PROF.instrument(obj, names)`` wraps methods of an
object so every call is a section. Sections nest: the path (``a > b > c``) is the key, so the same
function called from two places is reported separately. A section's *self* time is its total minus
its children, so the report shows where time really goes.

Everything is a no-op unless ``PROF.enabled`` is set, so the instrumentation can stay in the code.
"""

from __future__ import annotations

import atexit
import functools
import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional

import numpy as np

_SEP = " > "


class _Stat:
    __slots__ = ("samples",)

    def __init__(self) -> None:
        self.samples: list[float] = []


class Profiler:
    def __init__(self) -> None:
        self.enabled = False
        self.stats: dict[str, _Stat] = {}
        self.order: list[str] = []
        self._stack: list[str] = []
        self._starts: list[float] = []
        self.step_times: list[float] = []  # time spent in the root section of each control step
        self.loop_times: list[float] = []  # wall time between two control steps (controller + simulator + render)
        self._last_start: Optional[float] = None
        self.meta: dict[str, Any] = {}
        self.output_path: Optional[str] = None
        self._registered = False
        self.skip_first = 0
        self._seen_roots = 0
        self._owner: Optional[int] = None  # thread of the control loop; sections from other threads are ignored

    # ------------------------------------------------------------------ setup
    def enable(self, output_path: Optional[str] = None, skip_first: int = 0, **meta: Any) -> None:
        self.enabled = True
        self.output_path = output_path
        self.skip_first = int(skip_first)
        self.meta.update(meta)
        if not self._registered:
            atexit.register(self.report_at_exit)
            self._registered = True

    def reset(self) -> None:
        self.stats.clear()
        self.order.clear()
        self.step_times.clear()
        self.loop_times.clear()
        self._last_start = None
        self._seen_roots = 0

    # ------------------------------------------------------------------ recording
    def _record(self, path: str, dt: float) -> None:
        if self._seen_roots <= self.skip_first:
            return
        stat = self.stats.get(path)
        if stat is None:
            stat = self.stats[path] = _Stat()
            self.order.append(path)
        stat.samples.append(dt)

    def _mine(self) -> bool:
        return self._owner is None or self._owner == threading.get_ident()

    def begin(self, name: str) -> None:
        """Explicit start of a section, closed by ``end()``. For code that is awkward to indent."""
        if not self.enabled or not self._mine():
            return
        path = (self._stack[-1] + _SEP + name) if self._stack else name
        self._stack.append(path)
        self._starts.append(time.perf_counter())

    def end(self) -> None:
        if not self.enabled or not self._mine() or not self._stack or not self._starts:
            return
        dt = time.perf_counter() - self._starts.pop()
        self._record(self._stack.pop(), dt)

    @contextmanager
    def section(self, name: str):
        if not self.enabled or not self._mine():
            yield
            return
        path = (self._stack[-1] + _SEP + name) if self._stack else name
        self._stack.append(path)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._stack.pop()
            self._record(path, dt)

    def begin_step(self) -> None:
        """Call at the start of each control step (marks the loop period)."""
        if not self.enabled:
            return
        if self._owner is None:
            self._owner = threading.get_ident()
        if not self._mine():
            return
        now = time.perf_counter()
        if self._last_start is not None and self._seen_roots > self.skip_first:
            self.loop_times.append(now - self._last_start)
        self._last_start = now
        self._seen_roots += 1

    def end_step(self, dt: float) -> None:
        if self.enabled and self._seen_roots > self.skip_first:
            self.step_times.append(dt)

    def wrap(self, fn, name: Optional[str] = None):
        """``fn`` as a timed section (identity when disabled at call time)."""
        label = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "fn")).split(".")[-1]

        @functools.wraps(fn)
        def timed(*args, **kwargs):
            if not self.enabled or not self._mine():
                return fn(*args, **kwargs)
            with self.section(label):
                return fn(*args, **kwargs)

        return timed

    def instrument(self, obj: Any, names: Iterable[str], prefix: str = "") -> None:
        """Replace the methods ``names`` of ``obj`` by timed versions (missing names are skipped)."""
        for name in names:
            fn = getattr(obj, name, None)
            if fn is None or getattr(fn, "_profiled", False):
                continue
            timed = self.wrap(fn, prefix + name)
            timed._profiled = True  # type: ignore[attr-defined]
            setattr(obj, name, timed)

    def instrument_module(self, module: Any, names: Iterable[str], prefix: str = "") -> None:
        self.instrument(module, names, prefix)

    # ------------------------------------------------------------------ reporting
    @staticmethod
    def _summary(samples: list[float]) -> dict[str, float]:
        a = np.asarray(samples, dtype=np.float64) * 1e3
        return {
            "calls": int(a.size),
            "total_ms": float(a.sum()),
            "mean_ms": float(a.mean()),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max()),
            "min_ms": float(a.min()),
        }

    def table(self) -> list[dict[str, Any]]:
        steps = max(len(self.step_times), 1)
        rows = []
        children_total: dict[str, float] = {}
        for path, stat in self.stats.items():
            if _SEP in path:
                parent = path.rsplit(_SEP, 1)[0]
                children_total[parent] = children_total.get(parent, 0.0) + float(np.sum(stat.samples)) * 1e3
        step_total = float(np.sum(self.step_times)) * 1e3 if self.step_times else 0.0
        for path in self.order:
            s = self._summary(self.stats[path].samples)
            s["path"] = path
            s["depth"] = path.count(_SEP)
            s["self_total_ms"] = s["total_ms"] - children_total.get(path, 0.0)
            s["per_step_ms"] = s["total_ms"] / steps
            s["self_per_step_ms"] = s["self_total_ms"] / steps
            s["calls_per_step"] = s["calls"] / steps
            s["pct_of_step"] = 100.0 * s["total_ms"] / step_total if step_total else 0.0
            s["self_pct_of_step"] = 100.0 * s["self_total_ms"] / step_total if step_total else 0.0
            rows.append(s)
        return rows

    def format_report(self) -> str:
        rows = self.table()
        steps = len(self.step_times)
        lines = []
        head = ", ".join(f"{k}={v}" for k, v in self.meta.items())
        lines.append(f"HWM control-step profile ({steps} steps measured; {head})")
        if self.step_times:
            st = self._summary(self.step_times)
            lines.append(
                f"controller step: mean {st['mean_ms']:.3f} ms, median {st['median_ms']:.3f}, "
                f"p95 {st['p95_ms']:.3f}, max {st['max_ms']:.3f}, min {st['min_ms']:.3f}"
            )
        if self.loop_times:
            lt = self._summary(self.loop_times)
            lines.append(
                f"full loop period (controller + simulator + render + anything else): mean {lt['mean_ms']:.3f} ms, "
                f"median {lt['median_ms']:.3f}, p95 {lt['p95_ms']:.3f}, max {lt['max_ms']:.3f}  "
                f"-> {1e3 / lt['mean_ms']:.1f} control steps/s"
            )
            if self.step_times:
                outside = lt["mean_ms"] - self._summary(self.step_times)["mean_ms"]
                lines.append(f"outside the controller (simulator, rendering, glue): {outside:.3f} ms/step")
        lines.append("")
        lines.append(
            f"{'section':<70} {'calls/stp':>9} {'mean ms':>9} {'med ms':>9} {'p95 ms':>9} {'max ms':>9} "
            f"{'ms/step':>9} {'self/step':>9} {'%step':>6} {'self%':>6}"
        )
        for r in rows:
            name = "  " * r["depth"] + r["path"].rsplit(_SEP, 1)[-1]
            lines.append(
                f"{name:<70.70} {r['calls_per_step']:>9.2f} {r['mean_ms']:>9.3f} {r['median_ms']:>9.3f} "
                f"{r['p95_ms']:>9.3f} {r['max_ms']:>9.3f} {r['per_step_ms']:>9.3f} {r['self_per_step_ms']:>9.3f} "
                f"{r['pct_of_step']:>6.1f} {r['self_pct_of_step']:>6.1f}"
            )
        lines.append("")
        lines.append("Self time ranking (ms per control step):")
        for r in sorted(rows, key=lambda r: -r["self_per_step_ms"])[:25]:
            lines.append(f"  {r['self_per_step_ms']:>9.3f} ms  {r['self_pct_of_step']:>5.1f}%  {r['path']}")
        return "\n".join(lines)

    def dump(self, path: Optional[str] = None) -> Optional[str]:
        path = path or self.output_path
        if not path or not self.stats:
            return None
        text = self.format_report()
        with open(path + ".txt", "w") as f:
            f.write(text + "\n")
        payload = {
            "meta": self.meta,
            "steps": len(self.step_times),
            "step_ms": [t * 1e3 for t in self.step_times],
            "loop_ms": [t * 1e3 for t in self.loop_times],
            "sections": self.table(),
        }
        with open(path + ".json", "w") as f:
            json.dump(payload, f)
        return text

    def report_at_exit(self) -> None:
        if self.enabled and self.stats:
            try:
                text = self.dump()
                if text:
                    print(text)
            except Exception as error:  # never break shutdown
                print(f"[profiler] report failed: {error}")


PROF = Profiler()
