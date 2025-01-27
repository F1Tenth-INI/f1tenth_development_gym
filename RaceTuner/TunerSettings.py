# TunerSettings.py
import os

USE_REMOTE_FILES = False

REMOTE_MAP_DIR = "catkin_ws/src/f1tenth_system/gym_bridge/f1tenth_development_gym/utilities/maps"
REMOTE_SETTINGS_DIR = "catkin_ws/src/f1tenth_system/gym_bridge/f1tenth_development_gym/utilities"
REMOTE_AT_LOCAL_DIR = "./maps/"

REVERSE_DIRECTION = False

REMOTE_CONFIG = {
    "host": os.getenv("REMOTE_HOST", "ini-nuc.local"),
    "port": int(os.getenv("REMOTE_PORT", "22")),
    "username": os.getenv("REMOTE_USERNAME", "racecar"),
    "password": os.getenv("REMOTE_PASSWORD", "Inivincible"),
}

MAP_LIMITS_X = [-15, 8]
MAP_LIMITS_Y = [-5, 15]

AUTO_SCALE_MAP = True