"""
Utility to capture and log training run metadata including:
- Full command line arguments that were passed
- Settings values after being overridden by CLI args
- Timestamp and environment info
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def save_run_metadata(
    csv_filepath: str,
    cli_args: Dict[str, Any],
    settings_obj: Any,
) -> Optional[str]:
    """
    Save command line arguments and Settings state to a JSON file alongside experiment data.

    Args:
        csv_filepath: Path to the CSV file being saved (e.g., ExperimentRecordings/2026-02-26_..._data/metrics.csv)
        cli_args: Dictionary of parsed command line arguments (e.g., from argparse.Namespace.__dict__)
        settings_obj: Settings object with all current configuration values

    Returns:
        Path to saved metadata JSON file, or None if save failed
    """
    try:
        # Determine the experiment directory (the _data folder)
        experiment_dir = csv_filepath.rsplit(".", 1)[0] + "_data"
        
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)

        metadata_path = os.path.join(experiment_dir, "run_metadata.json")

        # Build metadata dictionary
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "command_line_args": _serialize_args(cli_args),
            "settings": _serialize_settings(settings_obj),
        }

        # Write to JSON file
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✓ Saved run metadata to: {metadata_path}")
        return metadata_path

    except Exception as e:
        print(f"⚠ Warning: Failed to save run metadata: {e}", file=sys.stderr)
        return None


def _serialize_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert argument dictionary to JSON-serializable format."""
    serialized = {}
    for key, value in args_dict.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            serialized[key] = value
        elif isinstance(value, (list, tuple)):
            serialized[key] = list(value)
        elif isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = str(value)
    return serialized


def _serialize_settings(settings_obj: Any) -> Dict[str, Any]:
    """Extract all public attributes from Settings object into JSON-serializable dictionary."""
    settings_dict = {}
    
    if hasattr(settings_obj, "__dict__"):
        for key, value in settings_obj.__dict__.items():
            # Skip private/dunder attributes and methods
            if key.startswith("_"):
                continue
            if callable(value):
                continue
                
            # Serialize value
            if value is None or isinstance(value, (str, int, float, bool)):
                settings_dict[key] = value
            elif isinstance(value, (list, tuple)):
                settings_dict[key] = list(value)
            elif isinstance(value, Path):
                settings_dict[key] = str(value)
            else:
                settings_dict[key] = str(value)
    
    return settings_dict


def print_run_summary(cli_args: Dict[str, Any], settings_obj: Any) -> None:
    """
    Print a human-readable summary of the run configuration to console.

    Args:
        cli_args: Dictionary of parsed command line arguments
        settings_obj: Settings object
    """
    print("\n" + "=" * 80)
    print("RUN CONFIGURATION SUMMARY")
    print("=" * 80)

    print("\nCommand Line Arguments:")
    for key, value in sorted(cli_args.items()):
        if key not in ["help"]:
            print(f"  {key:30s} = {value}")

    print("\nKey Settings:")
    key_settings = [
        "MAP_NAME",
        "SIMULATION_LENGTH",
        "RENDER_MODE",
        "DEVICE",
        "SAC_LEARNING_RATE",
        "SAC_BATCH_SIZE",
        "SAC_BUFFER_SIZE",
        "SAC_TAU",
        "SAC_GAMMA",
        "USE_TRAIN_MODE_EPISODES",
        "SAVE_VIDEOS",
    ]

    for key in key_settings:
        if hasattr(settings_obj, key):
            value = getattr(settings_obj, key)
            print(f"  {key:30s} = {value}")

    print("=" * 80 + "\n")
