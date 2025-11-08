"""
Utility module for parsing command-line arguments and overriding Settings attributes.

Usage:
    from utilities.parser_utilities import parse_settings_args, save_settings_snapshot
    
    # Parse arguments and override Settings
    args = parse_settings_args()
    
    # Or with custom description
    args = parse_settings_args(description='Custom description')
    
    # Save settings snapshot to file
    save_settings_snapshot(output_path='./settings_snapshot.yml')
"""
import argparse
import ast
import inspect
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def convert_value(value_str: str, target_type: type, original_value) -> Any:
    """Convert string value to appropriate type based on target type."""
    if target_type == bool:
        # Handle boolean values
        if value_str.lower() in ['true', '1', 'yes', 'on']:
            return True
        elif value_str.lower() in ['false', '0', 'no', 'off']:
            return False
        else:
            raise ValueError(f"Cannot convert '{value_str}' to bool")
    elif target_type == int:
        return int(value_str)
    elif target_type == float:
        return float(value_str)
    elif target_type == str:
        return value_str
    elif target_type in (list, tuple):
        # Try to parse as Python literal (list, tuple, etc.)
        try:
            parsed = ast.literal_eval(value_str)
            # If original was tuple, convert list to tuple
            if target_type == tuple and isinstance(parsed, list):
                return tuple(parsed)
            return parsed
        except (ValueError, SyntaxError):
            # If parsing fails, treat as string
            return value_str
    else:
        # For unknown types, try ast.literal_eval first, then fall back to string
        try:
            parsed = ast.literal_eval(value_str)
            # Preserve tuple type if original was tuple
            if isinstance(original_value, tuple) and isinstance(parsed, list):
                return tuple(parsed)
            return parsed
        except (ValueError, SyntaxError):
            return value_str


def get_settings_attributes(Settings):
    """Get all public attributes from Settings class."""
    # Get all attributes that are not private (don't start with _)
    attributes = {}
    for name, value in inspect.getmembers(Settings):
        if not name.startswith('_') and not inspect.ismethod(value) and not inspect.isfunction(value):
            attributes[name] = value
    return attributes


def serialize_value(value: Any) -> Any:
    """Convert value to a JSON/YAML serializable format."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
        return value.tolist()
    else:
        # For other types, convert to string
        return str(value)


def save_settings_snapshot(output_path: Optional[str] = None, format: str = 'yaml') -> str:
    """
    Save a snapshot of all Settings attributes to a file.
    
    Args:
        output_path: Path where to save the settings file. If None, saves to 
                     Settings.RECORDING_FOLDER with a timestamped filename.
        format: File format ('yaml' or 'json'). Defaults to 'yaml'.
    
    Returns:
        str: Path to the saved settings file.
    
    Example:
        from utilities.parser_utilities import save_settings_snapshot
        
        # Save to default location (recording folder)
        path = save_settings_snapshot()
        
        # Save to custom location
        path = save_settings_snapshot('./my_settings.yml')
        
        # Save as JSON
        path = save_settings_snapshot('./my_settings.json', format='json')
    """
    from utilities.Settings import Settings
    
    # Get all Settings attributes
    settings_attrs = get_settings_attributes(Settings)
    
    # Prepare settings dictionary with metadata
    settings_dict = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
        },
        'settings': {}
    }
    
    # Serialize all settings
    for attr_name, attr_value in sorted(settings_attrs.items()):
        try:
            settings_dict['settings'][attr_name] = serialize_value(attr_value)
        except Exception as e:
            # If serialization fails, save as string
            settings_dict['settings'][attr_name] = str(attr_value)
    
    # Determine output path
    if output_path is None:
        output_dir = Settings.RECORDING_FOLDER
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if format.lower() == 'yaml':
            output_path = os.path.join(output_dir, f'settings_snapshot_{timestamp}.yml')
        else:
            output_path = os.path.join(output_dir, f'settings_snapshot_{timestamp}.json')
    else:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save to file
    try:
        if format.lower() == 'yaml':
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is not installed. Install it with: pip install pyyaml")
            
            with open(output_path, 'w') as f:
                yaml.dump(settings_dict, f, default_flow_style=False, sort_keys=False)
        else:  # JSON
            with open(output_path, 'w') as f:
                json.dump(settings_dict, f, indent=2, sort_keys=False)
        
        print(f"Settings snapshot saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Warning: Could not save settings snapshot to {output_path}: {e}")
        raise


def parse_settings_args(description: Optional[str] = None, verbose: bool = True, 
                       save_snapshot: bool = False, snapshot_path: Optional[str] = None) -> argparse.Namespace:
    """
    Parse command-line arguments and override Settings attributes.
    
    Args:
        description: Optional description for the argument parser. 
                     Defaults to 'Run with configurable settings'.
        verbose: If True, print messages when overriding settings. Defaults to True.
        save_snapshot: If True, save a snapshot of all Settings to a file after parsing.
                      Defaults to True.
        snapshot_path: Path where to save the settings snapshot. If None, saves to 
                       Settings.RECORDING_FOLDER with a timestamped filename.
    
    Returns:
        argparse.Namespace: Parsed arguments namespace.
    
    Example:
        from utilities.parser_utilities import parse_settings_args
        from utilities.Settings import Settings
        
        args = parse_settings_args()
        # Settings attributes are now overridden based on command-line arguments
        # Settings snapshot is automatically saved
    """
    # Import Settings to get attributes
    from utilities.Settings import Settings
    
    # Get all Settings attributes
    settings_attrs = get_settings_attributes(Settings)
    
    # Create argument parser
    if description is None:
        description = 'Run with configurable settings'
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dynamically add arguments for all Settings attributes
    for attr_name, attr_value in sorted(settings_attrs.items()):
        # Determine the type of the attribute
        attr_type = type(attr_value)
        
        # Create argument name (lowercase with underscores)
        arg_name = f'--{attr_name}'
        
        # Add help text with current value
        help_text = f'Override Settings.{attr_name} (default: {attr_value}, type: {attr_type.__name__})'
        
        # Add argument
        parser.add_argument(arg_name, type=str, default=None, help=help_text)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Override settings with command line arguments
    for attr_name in settings_attrs.keys():
        arg_value = getattr(args, attr_name, None)
        if arg_value is not None:
            # Get the original value to determine type
            original_value = getattr(Settings, attr_name)
            target_type = type(original_value)
            
            # Convert and set the value
            try:
                converted_value = convert_value(arg_value, target_type, original_value)
                setattr(Settings, attr_name, converted_value)
                if verbose:
                    print(f"Overriding Settings.{attr_name} = {converted_value} (was {original_value})")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not convert '{arg_value}' to {target_type.__name__} for Settings.{attr_name}: {e}")
                    print(f"  Keeping original value: {original_value}")
    
    # Recalculate dependent paths/settings after overrides
    if hasattr(Settings, "recalculate_paths"):
        Settings.recalculate_paths()

    # Save settings snapshot after parsing
    if save_snapshot:
        try:
            save_settings_snapshot(output_path=snapshot_path)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not save settings snapshot: {e}")
    
    return args


def save_settings_with_recording(csv_filepath: str) -> str:
    """
    Save a settings snapshot alongside a CSV recording file.
    
    This function extracts the base name and directory from the CSV filepath
    and saves a settings snapshot with the same base name.
    
    Args:
        csv_filepath: Path to the CSV recording file.
    
    Returns:
        str: Path to the saved settings file.
    
    Example:
        from utilities.parser_utilities import save_settings_with_recording
        
        # After creating a CSV recording
        csv_path = recorder.csv_filepath
        settings_path = save_settings_with_recording(csv_path)
    """
    from utilities.Settings import Settings
    
    # Extract base name and directory from CSV path
    csv_dir = os.path.dirname(csv_filepath)
    csv_basename = os.path.basename(csv_filepath)
    csv_name_without_ext = os.path.splitext(csv_basename)[0]
    
    # Create settings file path with same base name
    settings_path = os.path.join(csv_dir, f'{csv_name_without_ext}_settings.yml')
    
    # Save settings snapshot
    return save_settings_snapshot(output_path=settings_path, format='yaml')

