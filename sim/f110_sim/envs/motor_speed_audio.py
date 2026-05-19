"""Continuous motor-speed tone for pygame rendering.

The renderer feeds motor shaft speed from ``update_obs``; PortAudio synthesis
runs in a sounddevice callback thread so the sim/render loop is never blocked.
"""

import threading
from typing import Optional

import numpy as np

from utilities.state_utilities import MOTOR_ANGULAR_VEL_IDX


class MotorSpeedAudio:
    """Smooth sine tone; pitch and volume follow |motor_angular_vel| [rad/s]."""

    SAMPLE_RATE = 44100
    BLOCKSIZE = 1024

    def __init__(
        self,
        f_min_hz: float = 55.0,
        f_max_hz: float = 650.0,
        omega_for_max_hz: float = 2000.0,
        min_omega_rad_s: float = 3.0,
        volume: float = 0.12,
        smooth_tau_s: float = 0.08,
    ):
        self._f_min_hz = float(f_min_hz)
        self._f_max_hz = float(f_max_hz)
        self._omega_for_max_hz = max(float(omega_for_max_hz), 1.0)
        self._min_omega_rad_s = float(min_omega_rad_s)
        self._volume = float(volume)
        self._smooth_tau_s = max(float(smooth_tau_s), 1e-3)

        self._target_omega = 0.0
        self._smoothed_omega = 0.0
        self._smoothed_f_hz = 0.0
        self._smoothed_amp = 0.0
        self._phase = 0.0
        self._lock = threading.Lock()
        self._stream = None
        self._sd = None
        self._available = False

        try:
            import sounddevice as sd

            self._sd = sd
            self._available = True
        except ImportError:
            print(
                "Motor speed audio disabled: install sounddevice "
                "(pip install sounddevice)"
            )
        except Exception as exc:
            print(f"Motor speed audio disabled: {exc}")

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if not self._available or self._stream is not None:
            return
        self._stream = self._sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=2,
            dtype=np.float32,
            blocksize=self.BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def set_target_omega(self, omega_rad_s: float) -> None:
        with self._lock:
            self._target_omega = float(omega_rad_s)

    # Backwards-compatible alias
    set_motor_angular_vel = set_target_omega

    def _audio_callback(self, outdata, frames, _time_info, _status) -> None:
        with self._lock:
            target_omega = abs(self._target_omega)

        dt = frames / self.SAMPLE_RATE
        alpha = 1.0 - np.exp(-dt / self._smooth_tau_s)

        self._smoothed_omega += alpha * (target_omega - self._smoothed_omega)

        if self._smoothed_omega < self._min_omega_rad_s:
            self._smoothed_f_hz *= 1.0 - alpha
            self._smoothed_amp *= 1.0 - alpha
            if self._smoothed_amp < 1e-5:
                outdata.fill(0)
                return
        else:
            t_norm = min(1.0, self._smoothed_omega / self._omega_for_max_hz)
            target_f = self._f_min_hz + t_norm * (self._f_max_hz - self._f_min_hz)
            target_amp = self._volume * (0.15 + 0.85 * t_norm)
            self._smoothed_f_hz += alpha * (target_f - self._smoothed_f_hz)
            self._smoothed_amp += alpha * (target_amp - self._smoothed_amp)

        f_hz = self._smoothed_f_hz
        amp = self._smoothed_amp
        t = np.arange(frames, dtype=np.float64) / self.SAMPLE_RATE
        phase = self._phase + 2.0 * np.pi * f_hz * t
        wave = amp * np.sin(phase)
        self._phase = float(phase[-1] % (2.0 * np.pi))

        mono = wave.astype(np.float32, copy=False)
        outdata[:, 0] = mono
        outdata[:, 1] = mono

    @classmethod
    def from_settings(cls):
        from utilities.Settings import Settings

        return cls(
            f_min_hz=float(getattr(Settings, "PYGAME_MOTOR_AUDIO_F_MIN_HZ", 55.0)),
            f_max_hz=float(getattr(Settings, "PYGAME_MOTOR_AUDIO_F_MAX_HZ", 650.0)),
            omega_for_max_hz=float(
                getattr(Settings, "PYGAME_MOTOR_AUDIO_OMEGA_MAX_RAD_S", 2000.0)
            ),
            min_omega_rad_s=float(
                getattr(Settings, "PYGAME_MOTOR_AUDIO_MIN_OMEGA_RAD_S", 3.0)
            ),
            volume=float(getattr(Settings, "PYGAME_MOTOR_AUDIO_VOLUME", 0.12)),
            smooth_tau_s=float(
                getattr(Settings, "PYGAME_MOTOR_AUDIO_SMOOTH_TAU_S", 0.08)
            ),
        )


def motor_angular_vel_from_render_obs(obs) -> Optional[float]:
    """Read ego motor shaft speed [rad/s] from a render/sim observation dict."""
    if obs is None:
        return None
    ego_idx = int(obs.get("ego_idx", 0))
    car_states = obs.get("car_states")
    if car_states is None:
        return None
    states = np.asarray(car_states)
    if states.ndim != 2 or states.shape[0] == 0:
        return None
    ego_idx = min(ego_idx, states.shape[0] - 1)
    if states.shape[1] <= MOTOR_ANGULAR_VEL_IDX:
        return None
    value = float(states[ego_idx, MOTOR_ANGULAR_VEL_IDX])
    return value if np.isfinite(value) else 0.0


def motor_erpm_from_omega(omega_rad_s: float) -> float:
    """VESC-style ERPM from motor shaft angular velocity [rad/s]."""
    return float(omega_rad_s) * 60.0 / (2.0 * np.pi)
