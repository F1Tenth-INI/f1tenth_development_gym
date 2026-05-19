"""
SDL environment setup (must run before the first pygame / SDL import).

On macOS, Bluetooth DualShock 4 / PS4 controllers disappear from pygame after
pygame.display is initialized unless the HIDAPI joystick driver is enabled.
"""
import os
import sys


def configure_macos_bluetooth_gamepad() -> None:
    if sys.platform != "darwin":
        return
    os.environ.setdefault("SDL_JOYSTICK_HIDAPI", "1")
    # Keep receiving axis updates when the pygame window is not focused.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
