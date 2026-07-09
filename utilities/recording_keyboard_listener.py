"""Keyboard shortcuts for toggling CSV recording during simulation."""

from utilities.Settings import Settings


def on_recording_key_press(car_system, key) -> None:
    try:
        if key.char != "r":
            return
        if car_system.recorder is None:
            print("No recorder available to toggle")
            return
        recording_started = car_system.recorder.toggle_recording()
        if recording_started is True:
            print("Recording STARTED (r key pressed)")
        elif recording_started is False:
            print("Recording STOPPED (r key pressed)")
        else:
            print("Recording toggle requested but recorder is starting up...")
    except AttributeError:
        pass


def start_recording_keyboard_listener(car_system) -> None:
    if Settings.RENDER_MODE is None:
        car_system.start_recorder()
        return

    try:
        from pynput import keyboard

        listener = keyboard.Listener(on_press=lambda key: on_recording_key_press(car_system, key))
        listener.start()
    except ImportError:
        car_system.start_recorder()
