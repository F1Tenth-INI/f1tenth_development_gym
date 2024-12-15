# TunerSettings.py
import os

USE_REMOTE_FILES = True

MAP_NAME = "RCA1"  # Replace as needed
LOCAL_MAP_DIR = "utilities/maps"
REMOTE_MAP_DIR = "catkin_ws/src/f1tenth_system/racecar/racecar/maps"
REMOTE_AT_LOCAL_DIR = "./maps/"

REVERSE_DIRECTION = False

REMOTE_CONFIG = {
    "host": os.getenv("REMOTE_HOST", "ini-xavier.local"),
    "port": int(os.getenv("REMOTE_PORT", "22")),
    "username": os.getenv("REMOTE_USERNAME", "racecar"),
    "password": os.getenv("REMOTE_PASSWORD", "Inivincible"),
    "remotePath":  os.path.join(REMOTE_MAP_DIR, MAP_NAME)
}

MAP_LIMITS_X = [-6.7, 6.7]
MAP_LIMITS_Y = [-7.6, 2.0]