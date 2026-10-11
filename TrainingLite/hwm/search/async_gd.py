"""Background coefficient optimization for the planner (``HWM_GD_ASYNC``).

The control loop never waits for gradient descent. Every control step it submits the
newest state and history, then drives the feedback law with the latest finished
coefficients. One worker thread runs plans back to back, each from the newest submitted
state; requests that arrive while a plan runs replace each other, so only the newest is
planned next.

The worker optimizes with its own copy of the actor and the LLD, so weight broadcasts
never change a model in the middle of a plan. New weights are loaded into the copies
before the next plan starts. Memory retrieval goes through ``MemoryManager``, which
locks its own state.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, NamedTuple, Optional

import torch

from TrainingLite.hwm.models.bundle import ModelBundle
from TrainingLite.hwm.search.coefficients import COEFF_DIM
from TrainingLite.hwm.search.gradient_planner import GradientPlanner
from TrainingLite.hwm.shared.memory_manager import MemoryManager
from TrainingLite.hwm.shared.rewards import LowLevelRewards
from TrainingLite.hwm.shared.wall_geometry import WallGeometry
from utilities.Settings import Settings


class AsyncPlan(NamedTuple):
    coefficients: torch.Tensor
    """``(1, COEFF_DIM)`` optimized feedback law."""
    generation: int
    """Episode generation the plan's start state belongs to."""
    control_index: int
    """Control step of the plan's start state."""
    steps: int
    """Gradient steps taken."""
    seconds: float
    """Wall time of the plan."""
    path: Optional[torch.Tensor]
    """``(HWM_DISPLAY_PATH_STEPS + 1, state_dim)`` expected LLD path of the coefficients, with ``HWM_GD_SHOW_PLAN``."""
    path_control_index: int
    """Control step the path starts at: the newest submitted state when the plan finished."""


class AsyncCoefficientOptimizer:
    def __init__(self, models: ModelBundle, memory: MemoryManager, wall_geometry: WallGeometry):
        self.models = ModelBundle(
            {name: copy.deepcopy(models[name]).eval() for name in ("actor", "LLD")}
        )
        self.planner = GradientPlanner(
            self.models["actor"],
            self.models["LLD"],
            LowLevelRewards(wall_geometry, self.models["LLD"]),
            memory,
            wall_geometry,
        )
        self.plans = 0
        self._condition = threading.Condition()
        self._request: Optional[tuple[int, int, torch.Tensor, torch.Tensor]] = None
        self._result: Optional[AsyncPlan] = None
        self._weights: Optional[dict[str, dict[str, Any]]] = None
        self._warm: Optional[tuple[int, torch.Tensor]] = None
        self._error: Optional[BaseException] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="hwm-gd", daemon=True)
        self._thread.start()

    def submit(self, generation: int, control_index: int, state: torch.Tensor, history: torch.Tensor) -> None:
        """Plan from raw ``state`` ``(1, state_dim)`` after ``history`` ``(1, S-1, row_dim)`` next."""
        with self._condition:
            self._request = (int(generation), int(control_index), state.detach().clone(), history.detach().clone())
            self._condition.notify()

    def latest(self, generation: int) -> Optional[AsyncPlan]:
        """Newest finished plan of ``generation``, None before the first one. Re-raises worker errors."""
        with self._condition:
            if self._error is not None:
                raise RuntimeError("asynchronous coefficient optimization failed") from self._error
            if self._result is None or self._result.generation != int(generation):
                return None
            return self._result

    def set_weights(self, sds: dict[str, dict[str, Any]]) -> None:
        """Module state dicts to load before the next plan. Later calls overwrite modules given earlier."""
        with self._condition:
            pending = {} if self._weights is None else self._weights
            pending.update({name: sd for name, sd in sds.items() if name in self.models})
            self._weights = pending or None

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        torch.set_num_threads(int(Settings.HWM_GD_ASYNC_THREADS))
        while True:
            with self._condition:
                while self._request is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                generation, control_index, state, history = self._request
                self._request = None
                weights, self._weights = self._weights, None
            try:
                if weights is not None:
                    self.models.load_state_dicts(weights, strict=True)
                    self.models.eval()
                plan = self._plan(generation, control_index, state, history)
            except BaseException as error:
                with self._condition:
                    self._error = error
                return
            with self._condition:
                self._result = plan
                self.plans += 1

    def _plan(self, generation: int, control_index: int, state: torch.Tensor, history: torch.Tensor) -> AsyncPlan:
        """One plan, warm started from this generation's last plan (zeros first) with ``HWM_GD_WARM_START``."""
        initial = None
        if Settings.HWM_GD_WARM_START:
            if self._warm is not None and self._warm[0] == generation:
                initial = self._warm[1]
            else:
                initial = torch.zeros(1, COEFF_DIM)
        budget = Settings.HWM_GD_TIME_BUDGET_S
        start = time.perf_counter()
        deadline = None if budget is None else start + float(budget)
        result = self.planner.optimize(
            state,
            history,
            deadline=deadline,
            max_steps=None if deadline is not None else int(Settings.HWM_GD_STEPS),
            initial_coefficients=initial,
            num_samples=int(Settings.HWM_GD_INFERENCE_SAMPLES),
        )
        coefficients = result.coefficients.detach()
        self._warm = (generation, coefficients)
        seconds = time.perf_counter() - start
        path, path_index = None, control_index
        if Settings.HWM_GD_SHOW_PLAN:
            # The car has moved on during the plan; draw from where it is now.
            with self._condition:
                newest = self._request
            if newest is not None and newest[0] == generation:
                path_index, path_state, path_history = newest[1], newest[2], newest[3]
            else:
                path_state, path_history = state, history
            path = self.planner.mean_trajectory(
                path_state, path_history, coefficients, steps=int(Settings.HWM_DISPLAY_PATH_STEPS)
            )[0]
        return AsyncPlan(coefficients, generation, control_index, result.steps, seconds, path, path_index)
