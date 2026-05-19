"""
Screen utility functions for cross-platform screen size detection.

Avoid pygame.display here: on macOS, initializing the SDL video subsystem
before opening a Bluetooth gamepad makes pygame.joystick.get_count() return 0
(DualShock 4 / PS4 Controller).
"""

import sys


class ScreenUtils:
    """Utility class for screen size detection and window sizing."""

    @staticmethod
    def get_screen_size():
        """
        Get the primary screen size without touching pygame.display.

        Returns:
            tuple: (width, height) of the primary screen in pixels
        """
        try:
            if sys.platform == "darwin":
                width, height = ScreenUtils._get_screen_size_macos()
            else:
                width, height = ScreenUtils._get_screen_size_tkinter()
            print(f"Detected primary screen size: {width}x{height}")
            return width, height
        except Exception as e:
            print(f"Error detecting screen size: {e}")
            print("Using default screen size: 1920x1080")
            return 1920, 1080  # Default fallback

    @staticmethod
    def _get_screen_size_tkinter():
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        return width, height

    @staticmethod
    def _get_screen_size_macos():
        try:
            from AppKit import NSScreen

            frame = NSScreen.mainScreen().frame()
            return int(frame.size.width), int(frame.size.height)
        except ImportError:
            return ScreenUtils._get_screen_size_tkinter()
    
    @staticmethod
    def get_scaled_window_size(scale_factor=0.4):
        """
        Get a scaled window size based on the primary screen size
        
        Args:
            scale_factor (float): Scaling factor (0.0 to 1.0)
            
        Returns:
            tuple: (width, height) of the scaled window
        """
        width, height = ScreenUtils.get_screen_size()
        window_width = int(width * scale_factor)
        window_height = int(height * scale_factor)
        print(f"Setting window size to: {window_width}x{window_height} ({scale_factor*100:.0f}%)")
        return window_width, window_height
    
    @staticmethod
    def get_primary_monitor_size():
        """
        Get the primary monitor size (same as get_screen_size for pygame)
        
        Returns:
            tuple: (width, height) of the primary monitor in pixels
        """
        return ScreenUtils.get_screen_size()
