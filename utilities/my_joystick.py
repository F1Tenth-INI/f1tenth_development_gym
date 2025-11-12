#!/usr/bin/env python3
# universal_joystick.py
import time
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pygame

LOG_LEVEL = logging.INFO
logger = logging.getLogger("universal_joystick")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)


class UniversalJoystick:
    """
    A robust pygame joystick wrapper that:
      - Detects Xbox / PS4 (DualShock 4) / generic pads by name heuristics
      - Assigns Steering / Throttle / Brake axes with sensible defaults
      - Calibrates idle offsets and applies deadzone
      - Returns (steering, throttle) for control loops

    Steering: in [-1, 1], right positive (you can invert)
    Throttle: in [0, 1] from triggers or stick (you can invert or use [-1,1] if desired)
    """

    # Heuristic substrings seen on Linux/macOS/Windows for common pads
    SONY_HINTS = (
        "Sony", "Wireless Controller", "Sony Interactive",
        "Sony Computer Entertainment", "PS4"
    )
    XBOX_HINTS = ("Xbox", "X-Box", "Microsoft Controller")

    def __init__(
        self,
        index: int = 0,
        deadzone: float = 0.08,
        steering_invert: bool = False,
        throttle_invert: bool = False,
        auto_calibrate: bool = True,
        calibration_retries: int = 5,
        prefer_detected_axis3_for_sony: bool = True,
    ):
        """
        Args:
            index: which joystick to open (0-based)
            deadzone: absolute threshold to zero small drift
            steering_invert: flip steering sign
            throttle_invert: invert throttle after normalization
            auto_calibrate: measure idle offsets at init
            calibration_retries: retries for calibration if axes moving
            prefer_detected_axis3_for_sony: a practical fix for many DS4s on Linux
        """
        self.index = index
        self.deadzone = float(deadzone)
        self.steering_invert = bool(steering_invert)
        self.throttle_invert = bool(throttle_invert)
        self.prefer_axis3_for_sony = bool(prefer_detected_axis3_for_sony)

        # pygame setup
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() < 1:
            raise RuntimeError("No joysticks detected by pygame.")

        self.joy = pygame.joystick.Joystick(self.index)
        self.joy.init()

        self.name = self.joy.get_name() or "Unknown"
        self.num_axes = self.joy.get_numaxes()
        self.num_buttons = self.joy.get_numbuttons()

        logger.info(
            f'Joystick: "{self.name}" with {self.num_axes} axes, {self.num_buttons} buttons'
        )

        # buffers
        self.axes = np.zeros(self.num_axes, dtype=float)
        self.offsets = np.zeros(self.num_axes, dtype=float)

        # axis indices for steering/throttle/brake
        self.Steering: int = 0
        self.Throttle: int = 1
        self.Brake: Optional[int] = None  # optional

        self._assign_axes_by_name()

        # sanity clamp to valid range
        self._clamp_axis_index("Steering")
        self._clamp_axis_index("Throttle")
        if self.Brake is not None:
            self._clamp_axis_index("Brake")

        logger.info(
            f"Axis mapping => Steering:{self.Steering}  Throttle:{self.Throttle}  "
            + (f"Brake:{self.Brake}" if self.Brake is not None else "Brake:None")
        )

        if auto_calibrate:
            self.calibrate(retries=calibration_retries)

    # ---------- Mapping & Calibration ----------

    def _is_sony(self) -> bool:
        n = self.name
        return any(hint in n for hint in self.SONY_HINTS)

    def _is_xbox(self) -> bool:
        n = self.name
        return any(hint in n for hint in self.XBOX_HINTS)

    def _assign_axes_by_name(self) -> None:
        """
        Heuristic axis assignment per vendor + robust fallbacks.
        For Sony DS4 on Linux, we've frequently seen:
            - Steering on axis 3 (your case)
            - Triggers on 3/4/5 depending on driver/SDL mapping
        For Xbox on Linux, typical:
            - Steering: 0
            - R2: 5, L2: 4 (or 2 on some 360 receivers)
        """
        n = self.name

        if self._is_xbox():
            # Common Xbox mapping on Linux (SDL):
            self.Steering = 0
            # Prefer R2 as throttle (often 5)
            self.Throttle = 5 if self.num_axes > 5 else (2 if self.num_axes > 2 else 1)
            # Brake often 4 or 2
            self.Brake = 4 if self.num_axes > 4 else (2 if self.num_axes > 2 else None)

        elif self._is_sony():
            # Steering on axis 3 (from your test)
            self.Steering = 3 if (self.prefer_axis3_for_sony and self.num_axes > 3) else 0
            # Throttle on left stick Y (axis 1), neutral = 0
            self.Throttle = 1
            self.Brake = None  # ignore triggers if you don’t need them
        else:
            # Generic fallback: left stick X for steering, rightmost axis for throttle
            self.Steering = 0 if self.num_axes > 0 else 0
            self.Throttle = min(self.num_axes - 1, 5) if self.num_axes > 1 else 0
            self.Brake = min(self.num_axes - 2, 4) if self.num_axes > 2 else None

    def _first_valid_axis(self, candidates: List[int], default: Optional[int]) -> Optional[int]:
        for c in candidates:
            if 0 <= c < self.num_axes:
                return c
        return default

    def _clamp_axis_index(self, attr: str) -> None:
        idx = getattr(self, attr)
        if idx is None:
            return
        if not (0 <= idx < self.num_axes):
            logger.warning(f"{attr} index {idx} out of range; falling back to 0")
            setattr(self, attr, 0)

    def calibrate(self, seconds: float = 0.3, retries: int = 3) -> None:
        """
        Capture idle offsets (averaged). If axes are moving beyond deadzone, retry.
        """
        for attempt in range(retries):
            pygame.event.pump()
            samples = []
            t0 = time.time()
            while time.time() - t0 < seconds:
                pygame.event.pump()
                vals = [self.joy.get_axis(i) for i in range(self.num_axes)]
                samples.append(vals)
                time.sleep(0.01)
            arr = np.array(samples, dtype=float)
            mean = arr.mean(axis=0)
            max_abs = np.max(np.abs(mean))
            if max_abs < max(self.deadzone, 0.12):  # tolerate small drift
                self.offsets[:] = mean
                logger.info(f"Calibration complete. Offsets: {np.round(self.offsets, 3)}")
                return
            logger.info("Axes moving during calibration; retrying...")
            time.sleep(0.4)
        logger.warning("Calibration skipped (axes moving). Offsets left as zeros.")

    # ---------- Reading ----------

    def _apply_deadzone(self, x: float) -> float:
        return 0.0 if abs(x) < self.deadzone else x

    def _normalize_trigger(self, v: float) -> float:
        """
        Convert trigger [-1,1] to [0,1] if requested.
        Many drivers give -1 (released) to +1 (fully pressed).
        """
        
        return -v if self.throttle_invert else v

    def read(self) -> Tuple[float, float]:
        pygame.event.pump()
        for i in range(self.num_axes):
            raw = self.joy.get_axis(i)
            self.axes[i] = raw - self.offsets[i]

        # steering
        s = float(self.axes[self.Steering])
        s = self._apply_deadzone(s)
        if self.steering_invert:
            s = -s

        # throttle from left stick Y
        t_raw = float(self.axes[self.Throttle])
        t_raw = self._apply_deadzone(t_raw)
        # map: up = −1 => throttle = 1, neutral = 0, down = +1 => throttle = 0
        t = self._normalize_trigger(t_raw)

        return s, t

        # ---------- Diagnostics ----------

    def read_full(self) -> Dict[str, Any]:
        """
        Return a snapshot of all axes/buttons (for debugging UIs).
        """
        pygame.event.pump()
        axes = [self.joy.get_axis(i) - self.offsets[i] for i in range(self.num_axes)]
        btns = {i: self.joy.get_button(i) for i in range(self.num_buttons)}
        return {
            "name": self.name,
            "axes": axes,
            "buttons": btns,
            "mapping": {
                "Steering": self.Steering,
                "Throttle": self.Throttle,
                "Brake": self.Brake,
            },
        }

    def run_axis_probe(self, print_only_changes: bool = True, interval: float = 0.05) -> None:
        """
        Live print all axes. Use this to discover your device’s exact mapping.
        """
        last = [0.0] * self.num_axes
        print(f'Probing axes for "{self.name}"...')
        while True:
            pygame.event.pump()
            changed = False
            cur = []
            for i in range(self.num_axes):
                v = round(self.joy.get_axis(i), 3)
                cur.append(v)
                if v != last[i]:
                    changed = True
            if changed or not print_only_changes:
                print("axes:", cur)
                last = cur
            time.sleep(interval)
