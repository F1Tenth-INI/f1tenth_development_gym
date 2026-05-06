#!/usr/bin/env python3
"""
State Comparison Visualization Tool

This script provides a UI for comparing recorded vehicle states with car model predictions.
It loads CSV recordings and allows comparison with different vehicle dynamics models.

Features:
- Load and visualize vehicle state recordings from CSV files
- Compare with physics-based car models (Pacejka tire model)
- Control delay simulation: specify number of timesteps for control delay
  (e.g., delay=4 means the control executed at time t is actually from time t-4)
- Interactive visualization with zooming, panning, and range selection
- Export comparison plots

Author: GitHub Copilot
Date: August 13, 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple

# Configure matplotlib for readable plot fonts on high-DPI displays
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
# Font sizes will be set responsively based on screen resolution
# Base font sizes (will be scaled based on DPI and multiplier)
matplotlib.rcParams['font.size'] = 12  # Base plot font size (will be adjusted)
matplotlib.rcParams['axes.titlesize'] = 14  # Plot title size
matplotlib.rcParams['axes.labelsize'] = 12  # Plot label size
matplotlib.rcParams['xtick.labelsize'] = 10  # Tick label size
matplotlib.rcParams['ytick.labelsize'] = 10  # Tick label size
matplotlib.rcParams['legend.fontsize'] = 11  # Legend size
matplotlib.rcParams['figure.titlesize'] = 16  # Figure title size

# Add the parent directories to path to import simulation modules
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'sim/f110_sim/envs'))
sys.path.append(os.path.join(parent_dir, 'utilities'))
sys.path.append(os.path.join(parent_dir, 'TrainingLite', 'Visualization'))

from utilities.state_utilities import STATE_VARIABLES, STATE_INDICES
from visualization_common import (
    VisualizationCommon, AVAILABLE_MODELS,
    STEERING_CONTROL_COLUMN, ACCELERATION_CONTROL_COLUMN
)


class StateComparisonVisualizer:
    """Main class for the state comparison visualization tool."""
    
    def __init__(self):
        # Initialize common utilities
        self.common = VisualizationCommon()
        
        # Config file path - use the directory where this script is located
        vis_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file_path = os.path.join(vis_dir, 'visualization_config.json')
        
        # Initialize tkinter UI
        self.root = tk.Tk()
        self.root.title("State Comparison Visualizer")
        
        # Initialize font scaling variables
        self.base_font_scale = 1.0  # Will be set by setup_dpi_scaling
        self.font_multiplier = 1.0  # User-adjustable multiplier
        
        # Detect and handle high DPI displays
        self.setup_dpi_scaling()
        
        # Setup larger UI fonts for control panel
        self.setup_ui_fonts()
        
        self.root.geometry("1800x1100")  # Extra large window for much larger fonts
        
        # Data storage - delegate to common
        self.data = self.common.data
        self.csv_file_path = self.common.csv_file_path or ''  # Ensure it's a string, not None
        self.time_column = self.common.time_column
        self.state_columns = self.common.state_columns
        self.control_columns = self.common.control_columns
        self.state_indices = self.common.state_indices
        
        # Visualization parameters
        self.start_index = 0
        self.end_index = None
        self.comparison_start_index = 0
        self.comparison_data_dict = {}  # Store predictions for all start indices
        
        # Slider optimization parameters
        self.slider_update_timer = None
        self.slider_update_delay = 300  # milliseconds to wait after slider stops moving
        
        # Available car models and parameters
        self.available_models = AVAILABLE_MODELS
        self.available_car_params = self.common.available_car_params
        
        # Preload models
        self.common.reload_car_models()
        self.car_models = self.common.car_models  # Keep reference for compatibility
        
        # Wider selectors so long option text remains visible in the dropdown
        self.selector_width_chars = 32
        
        # List of selected additional data columns to plot
        self.selected_other_data = []  # List of column names
        
        # Reference to twin axis for additional data (for clearing)
        self.ax_other = None
        
        # Sync scales for state and additional data
        self.sync_scales = tk.BooleanVar(value=False)
        
        # Color cycle for additional data plots (distinct colors)
        self.other_data_colors = ['#008000', '#FF00FF', '#0000FF', '#FFA500', '#00CED1', 
                                  '#9400D3', '#FFD700', '#FF1493', '#00FF00', '#8A2BE2',
                                  '#DC143C', '#20B2AA', '#FF6347', '#7B68EE', '#00FA9A']
        
        # Setup UI
        self.setup_ui()
        
        # Update font size info display (after UI is created)
        if hasattr(self, 'update_font_size_info'):
            self.update_font_size_info()
        
        # Bind keyboard shortcuts
        self.root.bind('<Command-s>', lambda event: self.save_plot())
        self.root.bind('<Control-s>', lambda event: self.save_plot())  # For non-Mac systems
        self.root.focus_set()  # Make sure the window can receive key events
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Load config file if it exists
        self.load_config()

        # Initialize car parameter file watcher
        self._init_param_file_watch()
    
    def _reload_car_models(self):
        """Force reload modules and recreate car model instances."""
        self.common.reload_car_models()
        self.car_models = self.common.car_models
        
    def setup_dpi_scaling(self):
        """Setup DPI scaling for responsive font sizing.
        
        Fonts are scaled inversely with DPI - smaller screens (lower DPI) get smaller fonts,
        larger screens (higher DPI) get larger fonts. This ensures readability on all displays.
        """
        try:
            # Get screen DPI
            dpi = self.root.winfo_fpixels('1i')
            
            # Calculate responsive font scale
            # Baseline: 96 DPI (standard desktop)
            # Lower DPI (smaller screens) -> smaller fonts
            # Higher DPI (larger/high-res screens) -> larger fonts
            baseline_dpi = 96.0
            font_scale = dpi / baseline_dpi
            
            # Clamp font scale to reasonable range (0.5x to 2.0x)
            font_scale = max(0.5, min(2.0, font_scale))
            
            # Base font sizes (for 96 DPI baseline)
            base_font_size = 12
            base_title_size = 14
            base_label_size = 12
            base_tick_size = 10
            base_legend_size = 11
            base_figure_title_size = 16
            
            # Apply responsive scaling (will be multiplied by user multiplier later)
            self.base_font_scale = font_scale
            self._apply_font_sizes()
            
            # Set figure DPI based on screen DPI (for better rendering)
            plot_dpi = max(72, min(150, int(100 * font_scale)))
            matplotlib.rcParams['figure.dpi'] = plot_dpi
            
            print(f"Responsive font scaling applied (DPI: {dpi:.1f}, Font scale: {font_scale:.2f}x)")
            
            # Set tkinter scaling for better text rendering
            try:
                # Use moderate scaling for UI elements
                ui_scale = max(1.0, min(1.5, font_scale))
                self.root.tk.call('tk', 'scaling', ui_scale)
            except:
                pass  # Some systems don't support tk scaling
                
        except Exception as e:
            print(f"Could not detect DPI scaling: {e}")
            # Use default baseline scaling
            self.base_font_scale = 1.0
            self._apply_font_sizes()
    
    def _apply_font_sizes(self):
        """Apply font sizes using base scale and user multiplier."""
        # Base font sizes (for 96 DPI baseline)
        base_font_size = 12
        base_title_size = 14
        base_label_size = 12
        base_tick_size = 10
        base_legend_size = 11
        base_figure_title_size = 16
        
        # Calculate final font sizes with both responsive scaling and user multiplier
        total_scale = self.base_font_scale * self.font_multiplier
        
        matplotlib.rcParams['font.size'] = int(base_font_size * total_scale)
        matplotlib.rcParams['axes.titlesize'] = int(base_title_size * total_scale)
        matplotlib.rcParams['axes.labelsize'] = int(base_label_size * total_scale)
        matplotlib.rcParams['xtick.labelsize'] = int(base_tick_size * total_scale)
        matplotlib.rcParams['ytick.labelsize'] = int(base_tick_size * total_scale)
        matplotlib.rcParams['legend.fontsize'] = int(base_legend_size * total_scale)
        matplotlib.rcParams['figure.titlesize'] = int(base_figure_title_size * total_scale)
        
        # Update info label if it exists
        if hasattr(self, 'font_size_info_label'):
            self.update_font_size_info()
    
    def setup_ui_fonts(self):
        """Setup responsive fonts for the UI control panel based on screen DPI and font multiplier."""
        self._apply_ui_fonts()
    
    def _apply_ui_fonts(self):
        """Apply UI fonts using base scale and user multiplier."""
        try:
            import tkinter.font as tkFont
            
            # Base UI font size (for 96 DPI baseline) - reasonable default
            base_ui_font_size = 10
            
            # Calculate responsive UI font size based on DPI scaling and user multiplier
            total_scale = self.base_font_scale * self.font_multiplier
            ui_font_size = int(base_ui_font_size * total_scale)
            
            # Clamp to reasonable range (6pt to 20pt to allow for multiplier adjustments)
            ui_font_size = max(6, min(20, ui_font_size))
            
            # Calculate responsive padding (scales with font size)
            padding_size = max(5, int(ui_font_size * 0.4))
            
            # Create responsive fonts for all UI elements
            self.large_font = tkFont.Font(family="Arial", size=ui_font_size, weight="normal")
            self.large_bold_font = tkFont.Font(family="Arial", size=ui_font_size, weight="bold")
            self.button_font = tkFont.Font(family="Arial", size=ui_font_size, weight="normal")
            self.metrics_font = tkFont.Font(family="Arial", size=ui_font_size, weight="bold")
            
            # Configure ttk styles with responsive fonts
            style = ttk.Style()
            
            # Configure all styles with responsive fonts and padding
            style.configure('Large.TLabel', font=self.large_font)
            style.configure('Bold.TLabel', font=self.large_bold_font)
            style.configure('Large.TButton', font=self.button_font, padding=(padding_size, padding_size))
            style.configure('Large.TEntry', font=self.large_font, padding=padding_size)
            style.configure('Large.TCombobox', font=self.large_font, padding=padding_size)
            style.configure('Large.TCheckbutton', font=self.large_font)
            
            # Set default fonts and padding for all ttk widgets
            style.configure('TLabel', font=self.large_font)
            style.configure('TButton', font=self.button_font, padding=(padding_size, padding_size))
            style.configure('TEntry', font=self.large_font, padding=padding_size)
            style.configure('TCombobox', font=self.large_font, padding=padding_size)
            style.configure('TCheckbutton', font=self.large_font)
            style.configure('TLabelFrame', font=self.large_bold_font)
            
            # Configure combobox dropdown list to have responsive font
            self.root.option_add('*TCombobox*Listbox.font', self.large_font)
            # Also configure the dropdown height
            style.map('TCombobox', fieldbackground=[('readonly', 'white')])
            style.map('Large.TCombobox', fieldbackground=[('readonly', 'white')])
            
            # Update slider font if it exists
            if hasattr(self, 'comparison_slider'):
                self.comparison_slider.config(font=self.large_font)
            
            print(f"Responsive UI fonts configured ({ui_font_size}pt for all elements, DPI scale: {self.base_font_scale:.2f}x, Multiplier: {self.font_multiplier:.2f}x)")
            
        except Exception as e:
            print(f"Could not configure UI fonts: {e}")
            # Fallback to system fonts
            self.large_font = None
            self.large_bold_font = None
            self.button_font = None
            self.metrics_font = None
        
    def _on_closing(self):
        """Clean up resources when the window is closed."""
        # Cancel any pending slider updates
        if self.slider_update_timer is not None:
            self.root.after_cancel(self.slider_update_timer)

        # Cancel parameter file watcher
        if hasattr(self, '_param_watch_timer') and self._param_watch_timer is not None:
            try:
                self.root.after_cancel(self._param_watch_timer)
            except Exception:
                pass
        
        # Save config before closing
        self.save_config()
        
        # Close the window
        self.root.destroy()
    
    def load_config(self):
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_file_path):
            return
        try:
            with open(self.config_file_path, 'r') as f:
                config = json.load(f)
            
            # Load CSV file
            csv_path = config.get('csv_file_path', '')
            
            if csv_path:
                # Convert to absolute path for consistency
                if not os.path.isabs(csv_path):
                    csv_path = os.path.abspath(csv_path)
                
                if os.path.exists(csv_path):
                    self.data = pd.read_csv(csv_path, comment='#')
                    self.csv_file_path = csv_path
                    self.common.data = self.data
                    self.common.csv_file_path = csv_path
                    
                    self.file_label.config(text=f"Loaded: {os.path.basename(csv_path)}")
                    print(f"Loaded CSV from config: {csv_path}")
                    
                    available_states = [col for col in self.state_columns if col in self.data.columns]
                    self.state_combo['values'] = available_states
                    # Restore selected state if present and valid
                    cfg_state = config.get('state_name')
                    if cfg_state and cfg_state in available_states:
                        self.state_var.set(cfg_state)
                    elif available_states:
                        self.state_var.set(available_states[0])
                    
                    # Update other data combo box with all CSV columns
                    all_columns = list(self.data.columns)
                    if hasattr(self, 'other_data_combo'):
                        self.other_data_combo['values'] = all_columns
                        self.other_data_var.set("")  # Reset to empty (default)
                    
                    # Enable reload button
                    if hasattr(self, 'reload_button'):
                        self.reload_button.config(state='normal')
                else:
                    print(f"Warning: CSV file in config does not exist: {csv_path}")
            else:
                print("No CSV file path in config")
            
            # Load settings
            self.start_index = config.get('start_index', 0)
            self.end_index = config.get('end_index', None)
            self.start_index_var.set(str(self.start_index))
            self.end_index_var.set(str(self.end_index) if self.end_index is not None else "")
            self.horizon_var.set(str(config.get('horizon_steps', 50)))
            self.steering_delay_var.set(str(config.get('steering_delay_steps', 2)))
            self.acceleration_delay_var.set(str(config.get('acceleration_delay_steps', 2)))
            
            # Load font multiplier
            if 'font_multiplier' in config:
                self.font_multiplier = float(config.get('font_multiplier', 1.0))
                # Clamp to valid range
                self.font_multiplier = max(0.5, min(3.0, self.font_multiplier))
                if hasattr(self, 'font_multiplier_var'):
                    self.font_multiplier_var.set(f"{self.font_multiplier:.2f}")
                self._apply_font_sizes()
                # Also update UI fonts when loading config
                if hasattr(self, '_apply_ui_fonts'):
                    self._apply_ui_fonts()
                if hasattr(self, 'update_font_size_info'):
                    self.update_font_size_info()
            
            # Load default car model and parameters
            if 'default_car_model' in config:
                default_model = config.get('default_car_model')
                if default_model in self.available_models.values() and hasattr(self, 'model_var'):
                    self.model_var.set(default_model)
            elif hasattr(self, 'model_var') and not self.model_var.get():
                # Set default if not already set
                if self.available_models:
                    self.model_var.set(list(self.available_models.values())[0])
            
            if 'default_car_parameters' in config:
                default_params = config.get('default_car_parameters')
                if default_params in self.available_car_params.values() and hasattr(self, 'params_var'):
                    self.params_var.set(default_params)
            elif hasattr(self, 'params_var') and not self.params_var.get():
                # Set default if not already set
                if self.available_car_params:
                    self.params_var.set(list(self.available_car_params.values())[0])

            # Restore UI toggles and selections
            try:
                if hasattr(self, 'enable_comparison'):
                    self.enable_comparison.set(bool(config.get('enable_comparison', True)))
                if hasattr(self, 'show_controls'):
                    self.show_controls.set(bool(config.get('show_controls', False)))
                if hasattr(self, 'show_delta_state'):
                    self.show_delta_state.set(bool(config.get('show_delta_state', False)))
                if hasattr(self, 'show_all_comparisons'):
                    self.show_all_comparisons.set(bool(config.get('show_all_comparisons', False)))
                if hasattr(self, 'sync_scales'):
                    self.sync_scales.set(bool(config.get('sync_scales', False)))
                if hasattr(self, 'show_metrics'):
                    self.show_metrics.set(bool(config.get('show_metrics', True)))
                # Restore selected other data columns that still exist
                if self.data is not None and hasattr(self, 'other_data_combo'):
                    all_columns = list(self.data.columns)
                    sel_other = config.get('selected_other_data', []) or []
                    self.selected_other_data = [c for c in sel_other if c in all_columns]
                    if hasattr(self, 'other_data_listbox'):
                        self.update_other_data_listbox()
                # Restore comparison slider position (validated later)
                if hasattr(self, 'comparison_start_var'):
                    self.comparison_start_var.set(int(config.get('comparison_start_index', 0)))
            except Exception:
                pass
            # Apply UI layout updates based on restored toggles
            try:
                if hasattr(self, '_update_plot_layout'):
                    self._update_plot_layout()
                if hasattr(self, 'on_show_all_comparisons_toggled'):
                    self.on_show_all_comparisons_toggled()
            except Exception:
                pass
            
            if self.data is not None and hasattr(self, 'comparison_slider'):
                self.update_comparison_slider_range()
                self.plot_state()
        except Exception as e:
            print(f"Could not load config: {e}")
    
    def save_config(self):
        """Save current configuration to JSON file."""
        try:
            def safe_int(var, default):
                if hasattr(var, 'get'):
                    val = var.get().strip()
                    return int(val) if val and val.isdigit() else default
                return default
            
            # Get csv_file_path - use self.csv_file_path directly
            csv_path = str(self.csv_file_path) if self.csv_file_path else ''
            
            # Ensure config directory exists
            config_dir = os.path.dirname(self.config_file_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            config_dict = {
                'csv_file_path': csv_path,
                'start_index': self.start_index,
                'end_index': self.end_index,
                'horizon_steps': safe_int(self.horizon_var, 50) if hasattr(self, 'horizon_var') else 50,
                'steering_delay_steps': safe_int(self.steering_delay_var, 2) if hasattr(self, 'steering_delay_var') else 2,
                'acceleration_delay_steps': safe_int(self.acceleration_delay_var, 2) if hasattr(self, 'acceleration_delay_var') else 2,
                'font_multiplier': self.font_multiplier,
                'enable_comparison': bool(self.enable_comparison.get()) if hasattr(self, 'enable_comparison') else True,
                'show_controls': bool(self.show_controls.get()) if hasattr(self, 'show_controls') else False,
                'show_delta_state': bool(self.show_delta_state.get()) if hasattr(self, 'show_delta_state') else False,
                'show_all_comparisons': bool(self.show_all_comparisons.get()) if hasattr(self, 'show_all_comparisons') else False,
                'sync_scales': bool(self.sync_scales.get()) if hasattr(self, 'sync_scales') else False,
                'show_metrics': bool(self.show_metrics.get()) if hasattr(self, 'show_metrics') else True,
                'state_name': self.state_var.get() if hasattr(self, 'state_var') and self.state_var.get() else '',
                'selected_other_data': list(self.selected_other_data) if hasattr(self, 'selected_other_data') else [],
                'comparison_start_index': int(self.comparison_start_var.get()) if hasattr(self, 'comparison_start_var') else 0,
            }
            
            # Add default car model and parameters if available
            if hasattr(self, 'model_var') and self.model_var.get():
                config_dict['default_car_model'] = self.model_var.get()
            if hasattr(self, 'params_var') and self.params_var.get():
                config_dict['default_car_parameters'] = self.params_var.get()
            
            # Write config file directly - this MUST work
            abs_config_path = os.path.abspath(self.config_file_path)
            with open(abs_config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure it's written
            
            print(f"Config saved to: {abs_config_path}")
            print(f"  csv_file_path = {csv_path}")
        except Exception as e:
            print(f"ERROR saving config: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Control panel
        self.setup_control_panel(control_frame)
        
        # Plot area
        self.setup_plot_area(plot_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel with file loading and options."""
        # Collapse button for all control panels - make it smaller
        collapse_button_frame = ttk.Frame(parent)
        collapse_button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        self.control_panels_collapsed = tk.BooleanVar(value=False)
        self.collapse_all_button = ttk.Button(collapse_button_frame, text="▼", width=2,
                                              command=self.toggle_all_control_panels)
        self.collapse_all_button.pack(pady=2)
        
        # Container frame for all control panels - this is what we'll hide/show
        self.control_panels_container = ttk.Frame(parent)
        self.control_panels_container.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # File loading section
        file_frame = ttk.LabelFrame(self.control_panels_container, text="Data Loading")
        file_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Button frame for load and reload buttons
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(pady=2)
        
        ttk.Button(button_frame, text="Load CSV File", 
                  command=self.load_csv_file, style='Large.TButton').pack(side=tk.LEFT, padx=2)
        
        self.reload_button = ttk.Button(button_frame, text="Reload", 
                                        command=self.reload_csv_file, style='Large.TButton')
        self.reload_button.pack(side=tk.LEFT, padx=2)
        self.reload_button.config(state='disabled')  # Disabled until a file is loaded
        
        self.file_label = ttk.Label(file_frame, text="No file loaded", 
                                   wraplength=200, style='Large.TLabel')
        self.file_label.pack(pady=2)
        
        # Data range controls
        ttk.Label(file_frame, text="Start Index:", style='Bold.TLabel').pack()
        self.start_index_var = tk.StringVar(value="0")
        ttk.Entry(file_frame, textvariable=self.start_index_var, width=10, 
                 style='Large.TEntry').pack()
        
        ttk.Label(file_frame, text="End Index:", style='Bold.TLabel').pack()
        self.end_index_var = tk.StringVar(value="")
        ttk.Entry(file_frame, textvariable=self.end_index_var, width=10, 
                 style='Large.TEntry').pack()
        
        ttk.Button(file_frame, text="Update Range", 
                  command=self.update_data_range, style='Large.TButton').pack(pady=2)
        
        # State selection
        state_frame = ttk.LabelFrame(self.control_panels_container, text="State Selection")
        state_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(state_frame, text="Select State:", style='Bold.TLabel').pack()
        self.state_var = tk.StringVar()
        self.state_combo = ttk.Combobox(state_frame, textvariable=self.state_var,
                                       values=self.state_columns, state='readonly',
                                       width=self.selector_width_chars,
                                       style='Large.TCombobox')
        self.state_combo.pack(pady=2)
        self.state_combo.bind('<<ComboboxSelected>>', self.on_state_changed)
        
        ttk.Label(state_frame, text="Select Other Data:", style='Bold.TLabel').pack()
        self.other_data_var = tk.StringVar()
        self.other_data_combo = ttk.Combobox(state_frame, textvariable=self.other_data_var,
                                            values=[], state='readonly',
                                            width=self.selector_width_chars,
                                            style='Large.TCombobox')
        self.other_data_combo.pack(pady=2)
        self.other_data_combo.bind('<<ComboboxSelected>>', self.on_other_data_changed)
        
        # Frame for selected items list and clear button
        other_data_list_frame = ttk.Frame(state_frame)
        other_data_list_frame.pack(pady=2, fill=tk.BOTH, expand=True)
        
        ttk.Label(other_data_list_frame, text="Selected:", style='Bold.TLabel').pack(anchor='w')
        
        # Listbox to show selected items (with scrollbar if needed)
        listbox_frame = ttk.Frame(other_data_list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.other_data_listbox = tk.Listbox(listbox_frame, height=4, yscrollcommand=scrollbar.set,
                                             font=self.large_font if hasattr(self, 'large_font') and self.large_font else None)
        self.other_data_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.other_data_listbox.yview)
        
        # Bind double-click to remove item
        self.other_data_listbox.bind('<Double-Button-1>', self.on_other_data_remove)
        
        # Clear button
        ttk.Button(other_data_list_frame, text="Clear All", 
                  command=self.clear_other_data, style='Large.TButton').pack(pady=2)
        
        # Sync scales checkbox
        ttk.Checkbutton(other_data_list_frame, text="Sync Scales",
                       variable=self.sync_scales,
                       command=self.on_sync_scales_toggled, style='Large.TCheckbutton').pack(pady=2)
        
        self.show_controls = tk.BooleanVar(value=False)
        ttk.Checkbutton(state_frame, text="Show Control Plots",
                       variable=self.show_controls,
                       command=self.on_show_controls_toggled, style='Large.TCheckbutton').pack(pady=2)
        
        self.show_delta_state = tk.BooleanVar(value=False)
        ttk.Checkbutton(state_frame, text="Show Delta State Plot",
                       variable=self.show_delta_state,
                       command=self.on_show_delta_state_toggled, style='Large.TCheckbutton').pack(pady=2)
        
        # Model Comparison options
        comparison_frame = ttk.LabelFrame(self.control_panels_container, text="Model Comparison")
        comparison_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.enable_comparison = tk.BooleanVar(value=True)  # Default enabled
        ttk.Checkbutton(comparison_frame, text="Enable Comparison",
                       variable=self.enable_comparison,
                       command=self.on_comparison_toggled, style='Large.TCheckbutton').pack()
        
        ttk.Label(comparison_frame, text="Car Model:", style='Bold.TLabel').pack()
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(comparison_frame, textvariable=self.model_var,
                                       values=list(self.available_models.values()),
                                       state='readonly', width=self.selector_width_chars,
                                       style='Large.TCombobox')
        self.model_combo.pack(pady=2)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)
        
        ttk.Label(comparison_frame, text="Car Parameters:", style='Bold.TLabel').pack()
        self.params_var = tk.StringVar()
        self.params_combo = ttk.Combobox(comparison_frame, textvariable=self.params_var,
                                        values=list(self.available_car_params.values()),
                                        state='readonly', width=self.selector_width_chars,
                                        style='Large.TCombobox')
        self.params_combo.pack(pady=2)
        self.params_combo.bind('<<ComboboxSelected>>', self.on_params_changed)

        # Status label to show currently loaded parameter summary
        self.param_status_label = ttk.Label(comparison_frame, text="", style='Large.TLabel', foreground='gray')
        self.param_status_label.pack(anchor='w', pady=(0, 4))
        
        ttk.Label(comparison_frame, text="Horizon Steps:", style='Bold.TLabel').pack()
        self.horizon_var = tk.StringVar(value="50")
        horizon_entry = ttk.Entry(comparison_frame, textvariable=self.horizon_var, width=10,
                                 style='Large.TEntry')
        horizon_entry.pack()
        horizon_entry.bind('<KeyRelease>', self.on_horizon_changed)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.control_panels_container, text="Settings")
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(settings_frame, text="Steering Delay Steps:", style='Bold.TLabel').pack()
        self.steering_delay_var = tk.StringVar(value="2")
        steering_delay_entry = ttk.Entry(settings_frame, textvariable=self.steering_delay_var, width=10,
                                        style='Large.TEntry')
        steering_delay_entry.pack()
        steering_delay_entry.bind('<KeyRelease>', self.on_steering_delay_changed)
        
        ttk.Label(settings_frame, text="Acceleration Delay Steps:", style='Bold.TLabel').pack()
        self.acceleration_delay_var = tk.StringVar(value="2")
        acceleration_delay_entry = ttk.Entry(settings_frame, textvariable=self.acceleration_delay_var, width=10,
                                            style='Large.TEntry')
        acceleration_delay_entry.pack()
        acceleration_delay_entry.bind('<KeyRelease>', self.on_acceleration_delay_changed)
        
        # Add explanatory label for control delays
        delay_help_label = ttk.Label(settings_frame, 
                                   text="(0 = no delay, 4 = use control\nfrom 4 timesteps ago)",
                                   style='Large.TLabel', foreground='gray')
        delay_help_label.pack(pady=(0, 5))
        
        ttk.Button(settings_frame, text="Run Full Comparison",
                  command=self.run_full_comparison, style='Large.TButton').pack(pady=5)
        
        ttk.Button(settings_frame, text="Save Plot",
                  command=self.save_plot, style='Large.TButton').pack(pady=2)
        
        self.show_all_comparisons = tk.BooleanVar()
        ttk.Checkbutton(settings_frame, text="Show All Comparisons",
                       variable=self.show_all_comparisons,
                       command=self.on_show_all_comparisons_toggled, style='Large.TCheckbutton').pack(pady=2)
        
        # Metrics display section
        metrics_frame = ttk.LabelFrame(self.control_panels_container, text="Error Metrics")
        metrics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Create labels for metrics display with larger fonts
        self.metrics_labels = {}
        metrics_list = ['Mean Error', 'Max Error', 'Error Std', 'RMSE']
        for metric in metrics_list:
            ttk.Label(metrics_frame, text=f"{metric}:", style='Bold.TLabel').pack(anchor='w')
            self.metrics_labels[metric.lower().replace(' ', '_')] = ttk.Label(
                metrics_frame, text="N/A", style='Large.TLabel'
            )
            self.metrics_labels[metric.lower().replace(' ', '_')].pack(anchor='w', pady=(0, 5))
        
        # Show metrics for current comparison checkbox
        self.show_metrics = tk.BooleanVar(value=True)
        ttk.Checkbutton(metrics_frame, text="Show Metrics",
                       variable=self.show_metrics,
                       command=self.on_show_metrics_toggled, style='Large.TCheckbutton').pack(pady=2)
        
        # Font multiplier control (for plot fonts)
        ttk.Label(metrics_frame, text="Font Size Multiplier:", style='Bold.TLabel').pack(anchor='w')
        font_mult_frame = ttk.Frame(metrics_frame)
        font_mult_frame.pack(anchor='w', pady=2, fill=tk.X)
        
        self.font_multiplier_var = tk.StringVar(value="1.0")  # Default multiplier
        font_mult_entry = ttk.Entry(font_mult_frame, textvariable=self.font_multiplier_var, width=8,
                                   style='Large.TEntry')
        font_mult_entry.pack(side=tk.LEFT, padx=(0, 5))
        font_mult_entry.bind('<KeyRelease>', self.on_font_multiplier_changed)
        font_mult_entry.bind('<Return>', self.on_font_multiplier_changed)
        
        # Add quick adjustment buttons
        ttk.Button(font_mult_frame, text="-", width=3,
                  command=lambda: self.adjust_font_multiplier(-0.1), style='Large.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(font_mult_frame, text="+", width=3,
                  command=lambda: self.adjust_font_multiplier(0.1), style='Large.TButton').pack(side=tk.LEFT, padx=2)
        
        # Info label showing current effective font size
        self.font_size_info_label = ttk.Label(metrics_frame, text="", style='Large.TLabel', foreground='gray')
        self.font_size_info_label.pack(anchor='w', pady=(0, 2))
        self.update_font_size_info()
    
    def toggle_all_control_panels(self):
        """Toggle the collapse state of all control panel frames."""
        if self.control_panels_collapsed.get():
            # Expand: show all control panels
            self.control_panels_container.pack(side=tk.LEFT, fill=tk.Y, padx=5)
            self.collapse_all_button.config(text="▼")
            self.control_panels_collapsed.set(False)
        else:
            # Collapse: hide all control panels
            self.control_panels_container.pack_forget()
            self.collapse_all_button.config(text="▶")
            self.control_panels_collapsed.set(True)
        
    def setup_plot_area(self, parent):
        """Setup the plotting area."""
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.backends._backend_tk import NavigationToolbar2Tk
            from matplotlib.figure import Figure
            
            # Create slider frame
            slider_frame = ttk.Frame(parent)
            slider_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            
            ttk.Label(slider_frame, text="Comparison Start Index:", style='Large.TLabel').pack(side=tk.LEFT)
            self.comparison_start_var = tk.IntVar(value=0)
            # Use responsive font for slider (fallback to reasonable default)
            slider_font = self.large_font if hasattr(self, 'large_font') and self.large_font else ('Arial', 10)
            self.comparison_slider = tk.Scale(slider_frame, from_=0, to=0, 
                                            variable=self.comparison_start_var,
                                            orient=tk.HORIZONTAL, length=400,
                                            font=slider_font,
                                            width=30,  # Make slider wider/taller
                                            sliderlength=40,  # Make slider thumb larger
                                            command=self.on_comparison_slider_changed)
            self.comparison_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            self.comparison_index_label = ttk.Label(slider_frame, text="Index: 0", style='Large.TLabel')
            self.comparison_index_label.pack(side=tk.RIGHT, padx=5)
            
            # Create figure with zoom/pan capabilities and proper DPI
            current_dpi = matplotlib.rcParams['figure.dpi']
            self.fig = Figure(figsize=(12, 10), dpi=current_dpi)
            
            # Initially create just the main plot (single subplot)
            self.ax = self.fig.add_subplot(111)  # Full-size single plot
            self.ax_controls = None  # Will be created when needed
            self.ax_delta = None  # Will be created when needed
            
            # Adjust layout
            self.fig.tight_layout(pad=2.0)
            
            self.canvas = FigureCanvasTkAgg(self.fig, parent)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Navigation toolbar for zooming and panning
            toolbar = NavigationToolbar2Tk(self.canvas, parent)
            toolbar.update()
            
            # Enable interactive navigation
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
            
            # Initialize pan variables
            self.pan_start = None
            self.is_panning = False
            
        except ImportError:
            ttk.Label(parent, text="Matplotlib not available for plotting").pack()
            self.fig = None
            self.ax = None
            self.canvas = None
            
    def load_csv_file(self):
        """Load a CSV file containing vehicle state data."""
        file_path = filedialog.askopenfilename(
            title="Select CSV Recording File",
            initialdir=os.path.join(os.path.dirname(__file__), '..', 'ExperimentRecordings'),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convert to absolute path to ensure consistency
                file_path = os.path.abspath(file_path)
                
                # Read CSV, skipping comment lines
                self.data = pd.read_csv(file_path, comment='#')
                
                # Store the file path - THIS MUST BE SET BEFORE save_config()
                self.csv_file_path = file_path
                self.common.csv_file_path = file_path
                self.common.data = self.data
                
                # Update file label
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"Loaded: {filename}")
                
                # Update state combo box with available columns
                available_states = [col for col in self.state_columns if col in self.data.columns]
                self.state_combo['values'] = available_states
                
                if available_states:
                    self.state_var.set(available_states[0])
                
                # Update other data combo box with all CSV columns
                all_columns = list(self.data.columns)
                self.other_data_combo['values'] = all_columns
                # Reset to empty (default)
                self.other_data_var.set("")
                # Clear selected additional data list when loading new CSV
                self.selected_other_data = []
                if hasattr(self, 'other_data_listbox'):
                    self.update_other_data_listbox()
                
                # Reset data range with default values
                self.start_index = 0
                self.end_index = min(500, len(self.data))  # Default range 0-500
                self.start_index_var.set("0")
                self.end_index_var.set(str(self.end_index))
                
                # Update comparison slider range
                if hasattr(self, 'comparison_slider'):
                    self.update_comparison_slider_range()
                    
                # Enable reload button
                if hasattr(self, 'reload_button'):
                    self.reload_button.config(state='normal')
                
                # Plot initial state
                self.plot_state()
                
                # Save config IMMEDIATELY after loading CSV - this is critical
                self.save_config()
                
                messagebox.showinfo("Success", 
                                  f"Loaded {len(self.data)} data points from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
    
    def reload_csv_file(self):
        """Reload the currently loaded CSV file."""
        if not self.csv_file_path or not os.path.exists(self.csv_file_path):
            messagebox.showerror("Error", "No file loaded or file no longer exists.")
            return
        
        try:
            # Read CSV, skipping comment lines
            self.data = pd.read_csv(self.csv_file_path, comment='#')
            # Sync to common object
            self.common.data = self.data
            self.common.csv_file_path = self.csv_file_path
            
            # Update file label
            filename = os.path.basename(self.csv_file_path)
            self.file_label.config(text=f"Reloaded: {filename}")
            
            # Update state combo box with available columns
            available_states = [col for col in self.state_columns if col in self.data.columns]
            self.state_combo['values'] = available_states
            
            if available_states:
                # Try to keep current selection if it still exists, otherwise use first available
                current_state = self.state_var.get()
                if current_state in available_states:
                    self.state_var.set(current_state)
                else:
                    self.state_var.set(available_states[0])
            
            # Update other data combo box with all CSV columns
            all_columns = list(self.data.columns)
            self.other_data_combo['values'] = all_columns
            
            # Keep selected other data items if they still exist in the new data
            self.selected_other_data = [item for item in self.selected_other_data 
                                       if item in all_columns]
            if hasattr(self, 'other_data_listbox'):
                self.update_other_data_listbox()
            
            # Update comparison slider range
            if hasattr(self, 'comparison_slider'):
                self.update_comparison_slider_range()
            
            # Plot current state
            self.plot_state()
            
            # Save config after reloading CSV
            self.save_config()
            
            messagebox.showinfo("Success", 
                              f"Reloaded {len(self.data)} data points from {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload CSV file: {str(e)}")
                
    
    def on_state_changed(self, event=None):
        """Handle state selection change."""
        self.plot_state()
        self.save_config()
    
    def on_other_data_changed(self, event=None):
        """Handle other data selection change - add to list if not already present."""
        selected = self.other_data_var.get()
        if selected and selected.strip() and selected not in self.selected_other_data:
            # Check if column exists in data
            if hasattr(self, 'data') and self.data is not None and selected in self.data.columns:
                self.selected_other_data.append(selected)
                self.update_other_data_listbox()
                # Reset dropdown selection
                self.other_data_var.set("")
                self.plot_state()
                self.save_config()
    
    def update_other_data_listbox(self):
        """Update the listbox to show currently selected additional data items."""
        if hasattr(self, 'other_data_listbox'):
            self.other_data_listbox.delete(0, tk.END)
            for item in self.selected_other_data:
                self.other_data_listbox.insert(tk.END, item)
    
    def on_other_data_remove(self, event=None):
        """Remove the selected item from the list when double-clicked."""
        if hasattr(self, 'other_data_listbox'):
            selection = self.other_data_listbox.curselection()
            if selection:
                idx = selection[0]
                if 0 <= idx < len(self.selected_other_data):
                    self.selected_other_data.pop(idx)
                    self.update_other_data_listbox()
                    self.plot_state()
    
    def clear_other_data(self):
        """Clear all selected additional data items."""
        self.selected_other_data = []
        self.update_other_data_listbox()
        self.save_config()
    
    def on_sync_scales_toggled(self):
        """Handle sync scales checkbox toggle."""
        self.plot_state()
        self.plot_state()
        self.save_config()
        
    def _update_plot_layout(self):
        """Update the plot layout based on which plots are enabled."""
        if self.fig is None or self.canvas is None:
            return
        
        # Store current axis limits to preserve zoom/pan state
        xlim = self.ax.get_xlim() if self.ax is not None else None
        ylim = self.ax.get_ylim() if self.ax is not None else None
        
        # Clear the entire figure
        self.fig.clear()
        
        show_controls = self.show_controls.get()
        show_delta = self.show_delta_state.get()
        
        if show_controls and show_delta:
            # Three plots: main, delta, controls
            self.ax = self.fig.add_subplot(311)
            self.ax_delta = self.fig.add_subplot(312, sharex=self.ax)
            self.ax_controls = self.fig.add_subplot(313, sharex=self.ax)
        elif show_controls:
            # Two plots: main, controls
            self.ax = self.fig.add_subplot(211)
            self.ax_controls = self.fig.add_subplot(212, sharex=self.ax)
            self.ax_delta = None
        elif show_delta:
            # Two plots: main, delta
            self.ax = self.fig.add_subplot(211)
            self.ax_delta = self.fig.add_subplot(212, sharex=self.ax)
            self.ax_controls = None
        else:
            # Single plot: main only
            self.ax = self.fig.add_subplot(111)
            self.ax_controls = None
            self.ax_delta = None
        
        # Restore axis limits if they existed
        if xlim is not None and self.ax is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None and self.ax is not None:
            self.ax.set_ylim(ylim)
        
        # Reconnect navigation events to new axes
        if hasattr(self, 'fig') and self.fig.canvas is not None:
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        
        # Adjust layout and redraw
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()
    
    def on_show_controls_toggled(self):
        """Handle show controls checkbox toggle."""
        self._update_plot_layout()
        self.plot_state()
        self.save_config()
    
    def on_show_delta_state_toggled(self):
        """Handle show delta state checkbox toggle."""
        self._update_plot_layout()
        self.plot_state()
        self.save_config()
        
    def on_show_all_comparisons_toggled(self):
        """Handle show all comparisons checkbox toggle."""
        # Enable/disable slider based on show all comparisons setting
        if hasattr(self, 'comparison_slider'):
            if self.show_all_comparisons.get():
                # Disable slider when showing all comparisons
                self.comparison_slider.config(state='disabled')
            else:
                # Enable slider when showing single comparison
                self.comparison_slider.config(state='normal')
        
        self.plot_state()
        self.save_config()
        
    def on_show_metrics_toggled(self):
        """Handle show metrics checkbox toggle."""
        self.update_metrics_display()
        self.save_config()
        
    def on_font_multiplier_changed(self, event=None):
        """Handle font multiplier change."""
        try:
            multiplier = float(self.font_multiplier_var.get())
            
            # Clamp multiplier to reasonable range
            multiplier = max(0.5, min(3.0, multiplier))
            
            self.font_multiplier = multiplier
            self.font_multiplier_var.set(f"{multiplier:.2f}")
            
            # Apply new font sizes for plots
            self._apply_font_sizes()
            
            # Apply new font sizes for UI
            self._apply_ui_fonts()
            
            # Update font size info display
            self.update_font_size_info()
            
            # Update existing plot
            if self.fig is not None:
                # Redraw with new font sizes
                self.canvas.draw()
            
            # Force UI to refresh with new font sizes
            self.root.update_idletasks()
                
        except (ValueError, AttributeError):
            # Invalid multiplier, revert to current value
            self.font_multiplier_var.set(f"{self.font_multiplier:.2f}")
    
    def adjust_font_multiplier(self, delta):
        """Adjust font multiplier by a delta amount."""
        new_multiplier = max(0.5, min(3.0, self.font_multiplier + delta))
        self.font_multiplier = new_multiplier
        self.font_multiplier_var.set(f"{new_multiplier:.2f}")
        # Apply changes (this will update both plot and UI fonts)
        self.on_font_multiplier_changed()
    
    def update_font_size_info(self):
        """Update the info label showing current effective font size."""
        if hasattr(self, 'font_size_info_label'):
            effective_size = int(12 * self.base_font_scale * self.font_multiplier)
            self.font_size_info_label.config(
                text=f"Effective base size: {effective_size}pt (DPI scale: {self.base_font_scale:.2f}x, Multiplier: {self.font_multiplier:.2f}x)"
            )
        
    def update_metrics_display(self):
        """Update the metrics display based on current comparison."""
        if not hasattr(self, 'metrics_labels') or not self.show_metrics.get():
            # Hide metrics if checkbox is unchecked
            for label in self.metrics_labels.values():
                label.config(text="N/A")
            return
        
        selected_state = self.state_var.get()
        if not selected_state or not self.enable_comparison.get():
            for label in self.metrics_labels.values():
                label.config(text="N/A")
            return
        
        # Calculate metrics based on what's actually displayed
        metrics = self.calculate_metrics_for_displayed_predictions(selected_state)
        
        if metrics:
            # Update display labels
            self.metrics_labels['mean_error'].config(text=f"{metrics['mean_error']:.4f}")
            self.metrics_labels['max_error'].config(text=f"{metrics['max_error']:.4f}")
            self.metrics_labels['error_std'].config(text=f"{metrics['error_std']:.4f}")
            self.metrics_labels['rmse'].config(text=f"{metrics['rmse']:.4f}")
        else:
            # If we get here, no valid metrics could be calculated
            for label in self.metrics_labels.values():
                label.config(text="N/A")
        
    def on_comparison_toggled(self):
        """Handle comparison checkbox toggle."""
        if self.enable_comparison.get() and not self.model_var.get():
            # Set defaults when enabling comparison
            self.model_var.set(list(self.available_models.values())[0])
            self.params_var.set(list(self.available_car_params.values())[0])
        self.plot_state()
        
    def on_model_changed(self, event=None):
        """Handle model selection change."""
        self._reload_car_models()
        if self.enable_comparison.get():
            self.plot_state()
        self.save_config()
            
    def on_params_changed(self, event=None):
        """Handle car parameters selection change."""
        # Reset watcher state to new file
        self._reset_param_file_watch()

        # Clear cached comparisons when parameters change
        if hasattr(self, 'comparison_data_dict'):
            self.comparison_data_dict = {}

        # If comparison is enabled, recompute based on current mode
        if self.enable_comparison.get():
            try:
                if hasattr(self, 'show_all_comparisons') and self.show_all_comparisons.get():
                    self.run_full_comparison()
                else:
                    comp_idx = self.comparison_start_var.get() if hasattr(self, 'comparison_start_var') else 0
                    if self.run_single_comparison(comp_idx):
                        self.plot_state()
            except Exception:
                pass
        self.save_config()
            
    def on_horizon_changed(self, event=None):
        """Handle horizon value change."""
        # Update comparison slider range when horizon changes
        self.update_comparison_slider_range()
        # Save config after horizon changes
        self.save_config()
    
    def on_steering_delay_changed(self, event=None):
        """Handle steering delay value change."""
        try:
            delay_str = self.steering_delay_var.get()
            if delay_str.strip():  # Only validate if not empty
                delay_val = int(delay_str)
                if delay_val < 0:
                    messagebox.showwarning("Warning", "Steering delay should be non-negative (>= 0)")
                    self.steering_delay_var.set("0")
                    self.save_config()
                    return
                elif delay_val > 100:  # Reasonable upper limit
                    messagebox.showwarning("Warning", "Steering delay seems very large. Are you sure?")
                # Save config after valid delay change
                self.save_config()
        except ValueError:
            if self.steering_delay_var.get().strip():  # Only show error for non-empty invalid input
                messagebox.showerror("Error", "Steering delay must be a non-negative integer")
                self.steering_delay_var.set("0")
                self.save_config()
                return
    
    def on_acceleration_delay_changed(self, event=None):
        """Handle acceleration delay value change."""
        try:
            delay_str = self.acceleration_delay_var.get()
            if delay_str.strip():  # Only validate if not empty
                delay_val = int(delay_str)
                if delay_val < 0:
                    messagebox.showwarning("Warning", "Acceleration delay should be non-negative (>= 0)")
                    self.acceleration_delay_var.set("0")
                    self.save_config()
                    return
                elif delay_val > 100:  # Reasonable upper limit
                    messagebox.showwarning("Warning", "Acceleration delay seems very large. Are you sure?")
                # Save config after valid delay change
                self.save_config()
        except ValueError:
            if self.acceleration_delay_var.get().strip():  # Only show error for non-empty invalid input
                messagebox.showerror("Error", "Acceleration delay must be a non-negative integer")
                self.acceleration_delay_var.set("0")
                self.save_config()
                return
        
        # Clear existing comparison data when control delay changes
        if hasattr(self, 'comparison_data_dict'):
            self.comparison_data_dict = {}
        # Update comparison slider range when control delay changes
        self.update_comparison_slider_range()
    
    def on_slider_delay_changed(self, event=None):
        """Handle slider update delay value change."""
        try:
            delay_str = self.slider_delay_var.get()
            if delay_str.strip():  # Only validate if not empty
                delay_val = int(delay_str)
                if delay_val < 50:
                    messagebox.showwarning("Warning", "Slider delay should be at least 50ms for stability")
                    self.slider_delay_var.set("50")
                    return
                elif delay_val > 2000:
                    messagebox.showwarning("Warning", "Slider delay seems very large. Are you sure?")
                
                # Update the actual delay value
                self.slider_update_delay = delay_val
        except ValueError:
            if self.slider_delay_var.get().strip():  # Only show error for non-empty invalid input
                messagebox.showerror("Error", "Slider delay must be a positive integer (milliseconds)")
                self.slider_delay_var.set("300")
                return
    
    
    def plot_state(self):
        """Plot the selected state variable."""
        if self.fig is None or self.ax is None or self.canvas is None or self.data is None:
            return
            
        selected_state = self.state_var.get()
        if not selected_state or selected_state not in self.data.columns:
            return
            
        self.ax.clear()
        
        # Clear and remove any existing twin axis for additional data
        if hasattr(self, 'ax_other') and self.ax_other is not None:
            try:
                self.ax_other.clear()
                self.ax_other.remove()
            except:
                pass
            self.ax_other = None
        
        # Get data range
        start_idx = self.start_index
        end_idx = self.end_index if self.end_index is not None else len(self.data)
        
        # Plot ground truth data
        if self.time_column in self.data.columns:
            time_data = self.data[self.time_column].iloc[start_idx:end_idx]
        else:
            time_data = np.arange(start_idx, end_idx)
        
        state_data = self.data[selected_state].iloc[start_idx:end_idx]
        
        self.ax.plot(time_data, state_data, 'k-', label='Ground Truth', linewidth=2)
        
        # Plot model prediction if comparison is enabled
        if self.enable_comparison.get() and hasattr(self, 'comparison_data_dict') and self.comparison_data_dict:
            
            if hasattr(self, 'show_all_comparisons') and self.show_all_comparisons.get():
                # Show all full horizon predictions starting from every timestep within current range
                print(f"Plotting all full horizon predictions for {selected_state}")
                comparison_count = 0
                
                for comp_start_idx, comparison_data in self.comparison_data_dict.items():
                    # Only show predictions that start within the current visible range
                    if comp_start_idx < start_idx or comp_start_idx >= end_idx:
                        continue
                        
                    if selected_state in comparison_data:
                        # Get full horizon prediction data
                        full_horizon = len(comparison_data[selected_state])
                        
                        # Generate time data for the full prediction horizon
                        if self.time_column in self.data.columns:
                            # Get the actual time values from data, extending beyond current range if needed
                            if comp_start_idx + full_horizon <= len(self.data):
                                comp_time = self.data[self.time_column].iloc[comp_start_idx:comp_start_idx + full_horizon]
                            else:
                                # If prediction extends beyond data, extrapolate time
                                available_time = self.data[self.time_column].iloc[comp_start_idx:]
                                dt = self.get_timestep()
                                missing_steps = full_horizon - len(available_time)
                                if len(available_time) > 0:
                                    last_time = available_time.iloc[-1]
                                    extra_time = np.arange(1, missing_steps + 1) * dt + last_time
                                    comp_time = np.concatenate([available_time.to_numpy(), extra_time])
                                else:
                                    comp_time = np.arange(comp_start_idx, comp_start_idx + full_horizon) * dt
                        else:
                            dt = self.get_timestep()
                            comp_time = np.arange(comp_start_idx, comp_start_idx + full_horizon) * dt
                        
                        # Show full prediction data with color gradient over horizon
                        full_prediction = np.array(comparison_data[selected_state])
                        
                        # Plot with color gradient over horizon
                        self.plot_prediction_with_gradient(comp_time, full_prediction, 
                                                          comparison_count, full_horizon, 
                                                          label='Model Predictions (Full Horizon)' if comparison_count == 0 else None)
                        comparison_count += 1
                        
                if comparison_count > 0:
                    print(f"Plotted {comparison_count} full horizon predictions starting within range [{start_idx}, {end_idx})")
                else:
                    print(f"No comparison data found for state {selected_state} within current range [{start_idx}, {end_idx})")
                    
            else:
                # Show single comparison
                comp_start_idx = self.comparison_start_var.get() if hasattr(self, 'comparison_start_var') else 0
                
                # Debug information
                print(f"Plotting single comparison: enabled, looking for index {comp_start_idx}")
                print(f"Available comparison indices: {list(self.comparison_data_dict.keys())}")
                
                if comp_start_idx in self.comparison_data_dict:
                    comparison_data = self.comparison_data_dict[comp_start_idx]
                    print(f"Found comparison data with states: {list(comparison_data.keys())}")
                    
                    if selected_state in comparison_data:
                        # Get time data for comparison
                        horizon = len(comparison_data[selected_state])
                        if self.time_column in self.data.columns:
                            comp_time = self.data[self.time_column].iloc[comp_start_idx:comp_start_idx + horizon]
                        else:
                            dt = self.get_timestep()
                            comp_time = np.arange(comp_start_idx, comp_start_idx + horizon) * dt
                        
                        print(f"Plotting {selected_state} comparison with {horizon} points")
                        
                        # Plot with color gradient over horizon
                        self.plot_prediction_with_gradient(comp_time, np.array(comparison_data[selected_state]), 
                                                          0, horizon, label='Model Prediction')
                    else:
                        print(f"Selected state {selected_state} not in comparison data")
                else:
                    print(f"No comparison data for index {comp_start_idx}")
        else:
            print(f"Comparison not enabled or no data: enabled={self.enable_comparison.get()}, has_dict={hasattr(self, 'comparison_data_dict')}, dict_empty={not self.comparison_data_dict if hasattr(self, 'comparison_data_dict') else 'N/A'}")
        
        # Plot other data items from the list (on twin y-axis with different colors)
        self.ax_other = None
        if hasattr(self, 'selected_other_data') and self.selected_other_data:
            # Create a single twin y-axis for all additional data items
            self.ax_other = self.ax.twinx()
            
            # Plot all selected items with different colors
            ylabels = []
            for idx, selected_other_data in enumerate(self.selected_other_data):
                if selected_other_data and selected_other_data.strip() and selected_other_data in self.data.columns:
                    other_data_values = self.data[selected_other_data].iloc[start_idx:end_idx]
                    
                    # Get color from color cycle
                    color = self.other_data_colors[idx % len(self.other_data_colors)]
                    
                    # Plot with unique color
                    self.ax_other.plot(time_data, other_data_values, '-', color=color, 
                                label=f'{selected_other_data}', linewidth=2, alpha=0.7)
                    ylabels.append(selected_other_data.replace('_', ' ').title())
            
            # Set ylabel to show all selected items (comma-separated)
            if ylabels:
                self.ax_other.set_ylabel(' / '.join(ylabels), color='gray')
                self.ax_other.tick_params(axis='y', labelcolor='gray')
        
        self.ax.set_xlabel('Time (s)' if self.time_column in self.data.columns else 'Step')
        self.ax.set_ylabel(selected_state.replace('_', ' ').title())
        self.ax.set_title(f'State Comparison: {selected_state}')
        
        # Clear any existing legend before creating a new one
        if self.ax.get_legend() is not None:
            self.ax.get_legend().remove()
        
        # Create combined legend if we have other data
        if self.ax_other is not None:
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = self.ax_other.get_legend_handles_labels()
            # Combine all lines and labels
            all_lines = lines1 + lines2
            all_labels = labels1 + labels2
            # Remove duplicates while preserving order (in case of duplicates)
            seen_labels = set()
            unique_lines = []
            unique_labels = []
            for line, label in zip(all_lines, all_labels):
                if label not in seen_labels:
                    seen_labels.add(label)
                    unique_lines.append(line)
                    unique_labels.append(label)
            # Create legend with all items
            self.ax.legend(unique_lines, unique_labels, loc='upper left', frameon=True, fancybox=True, shadow=True)
        else:
            # Create legend with just the main plot items
            self.ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        self.ax.grid(True, alpha=0.3)
        
        # Set axis limits to respect the current data range
        if self.time_column in self.data.columns:
            x_min = self.data[self.time_column].iloc[start_idx]
            x_max = self.data[self.time_column].iloc[end_idx - 1] if end_idx <= len(self.data) else self.data[self.time_column].iloc[-1]
            self.ax.set_xlim(x_min, x_max)
            # Also set x-axis limits for twin axis if it exists
            if self.ax_other is not None:
                self.ax_other.set_xlim(x_min, x_max)
        else:
            self.ax.set_xlim(start_idx, end_idx - 1)
            # Also set x-axis limits for twin axis if it exists
            if self.ax_other is not None:
                self.ax_other.set_xlim(start_idx, end_idx - 1)
        
        # Sync y-axis scales if enabled
        if self.sync_scales.get() and self.ax_other is not None:
            # Get the y-limits from both axes
            ylim_main = self.ax.get_ylim()
            ylim_other = self.ax_other.get_ylim()
            # Use the wider range to ensure all data is visible
            y_min = min(ylim_main[0], ylim_other[0])
            y_max = max(ylim_main[1], ylim_other[1])
            # Apply the same limits to both axes
            self.ax.set_ylim(y_min, y_max)
            self.ax_other.set_ylim(y_min, y_max)
        
        # Plot delta state if enabled
        if (hasattr(self, 'ax_delta') and self.ax_delta is not None and 
            self.show_delta_state.get()):
            self.plot_delta_state(start_idx, end_idx, time_data, selected_state)
        
        # Plot controls if enabled
        if (hasattr(self, 'ax_controls') and self.ax_controls is not None and 
            self.show_controls.get()):
            self.plot_controls(start_idx, end_idx, time_data)
        
        self.canvas.draw()
        
        # Update metrics display after plotting
        self.update_metrics_display()
        
    def plot_prediction_with_gradient(self, time_data, prediction_data, trajectory_index, horizon, label=None):
        """Plot prediction with color gradient over horizon."""
        if self.ax is None:
            return
        # Delegate to common utilities
        self.common.plot_prediction_with_gradient(self.ax, time_data, prediction_data, horizon, label, trajectory_index)
    
    def plot_delta_state(self, start_idx, end_idx, time_data, selected_state):
        """Plot delta (difference between consecutive states) for ground truth and predictions."""
        if self.ax_delta is None or self.data is None:
            return
        
        self.ax_delta.clear()
        
        # Check if we have a pre-computed delta column
        delta_col = f'delta_state_{selected_state}'
        if delta_col in self.data.columns:
            # Use pre-computed delta column
            delta_gt = self.data[delta_col].iloc[start_idx:end_idx-1].values
            delta_time = time_data.iloc[:-1].values if hasattr(time_data, 'iloc') else time_data[:-1]
        else:
            # Calculate delta for ground truth: delta[i] = state[i+1] - state[i]
            state_data = self.data[selected_state].iloc[start_idx:end_idx].values
            if len(state_data) > 1:
                if selected_state == 'pose_theta':
                    # Handle angular wrap-around for pose_theta (radians)
                    delta_raw = np.diff(state_data)
                    delta_gt = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
                else:
                    delta_gt = np.diff(state_data)
                delta_time = time_data.iloc[:-1].values if hasattr(time_data, 'iloc') else time_data[:-1]
            else:
                delta_gt = np.array([])
                delta_time = np.array([])
        
        if len(delta_gt) > 0:
            self.ax_delta.plot(delta_time, delta_gt, 'k-', label='Ground Truth Delta', linewidth=2)
        
        # Plot delta for predictions if comparison is enabled
        if self.enable_comparison.get() and hasattr(self, 'comparison_data_dict') and self.comparison_data_dict:
            # Get comparison indices to plot
            if hasattr(self, 'show_all_comparisons') and self.show_all_comparisons.get():
                comp_indices = [(idx, data) for idx, data in self.comparison_data_dict.items() 
                               if start_idx <= idx < end_idx and selected_state in data]
            else:
                comp_start_idx = self.comparison_start_var.get() if hasattr(self, 'comparison_start_var') else 0
                if comp_start_idx in self.comparison_data_dict and selected_state in self.comparison_data_dict[comp_start_idx]:
                    comp_indices = [(comp_start_idx, self.comparison_data_dict[comp_start_idx])]
                else:
                    comp_indices = []
            
            # Plot each comparison
            for comp_count, (comp_start_idx, comparison_data) in enumerate(comp_indices):
                prediction = np.array(comparison_data[selected_state])
                if len(prediction) > 1:
                    # Handle angular wrap-around for pose_theta (radians)
                    if selected_state == 'pose_theta':
                        delta_raw = np.diff(prediction)
                        delta_pred = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
                    else:
                        delta_pred = np.diff(prediction)
                    horizon = len(prediction)
                    
                    # Generate time data
                    if self.time_column in self.data.columns:
                        if comp_start_idx + horizon <= len(self.data):
                            comp_time = self.data[self.time_column].iloc[comp_start_idx:comp_start_idx + horizon]
                        else:
                            available_time = self.data[self.time_column].iloc[comp_start_idx:]
                            dt = self.get_timestep()
                            missing_steps = horizon - len(available_time)
                            if len(available_time) > 0:
                                last_time = available_time.iloc[-1]
                                extra_time = np.arange(1, missing_steps + 1) * dt + last_time
                                comp_time = np.concatenate([available_time.to_numpy(), extra_time])
                            else:
                                comp_time = np.arange(comp_start_idx, comp_start_idx + horizon) * dt
                    else:
                        dt = self.get_timestep()
                        comp_time = np.arange(comp_start_idx, comp_start_idx + horizon) * dt
                    
                    # Time for delta (one less point)
                    delta_pred_time = comp_time.iloc[:-1].values if hasattr(comp_time, 'iloc') else comp_time[:-1]
                    
                    # Plot with gradient
                    label = 'Model Prediction Delta (Full Horizon)' if comp_count == 0 and len(comp_indices) > 1 else 'Model Prediction Delta'
                    self.plot_delta_prediction_with_gradient(delta_pred_time, delta_pred, comp_count, len(delta_pred), label if comp_count == 0 else None)
        
        self.ax_delta.set_xlabel('Time (s)' if self.time_column in self.data.columns else 'Step')
        self.ax_delta.set_ylabel(f'Δ {selected_state.replace("_", " ").title()}')
        self.ax_delta.set_title(f'Delta State: {selected_state}')
        self.ax_delta.legend()
        self.ax_delta.grid(True, alpha=0.3)
        
        # Set x-axis limits to match main plot
        if self.time_column in self.data.columns:
            x_min = self.data[self.time_column].iloc[start_idx]
            x_max = self.data[self.time_column].iloc[end_idx - 1] if end_idx <= len(self.data) else self.data[self.time_column].iloc[-1]
            self.ax_delta.set_xlim(x_min, x_max)
        else:
            self.ax_delta.set_xlim(start_idx, end_idx - 1)
    
    def plot_delta_prediction_with_gradient(self, time_data, delta_data, trajectory_index, horizon, label=None):
        """Plot delta prediction with color gradient over horizon."""
        if self.ax_delta is None:
            return
        
        try:
            from matplotlib.colors import LinearSegmentedColormap
            colors = ["#5200F5", "#FF00BF", "#FF0000"]
            alpha_base = 0.4
            cmap = LinearSegmentedColormap.from_list("gradient", colors, N=100)
        except ImportError:
            if trajectory_index == 0:
                start_color = np.array([1.0, 0.4, 0.4])
                end_color = np.array([0.5, 0.0, 0.0])
                alpha_base = 0.8
            else:
                start_color = np.array([1.0, 0.6, 0.0])
                end_color = np.array([0.5, 0.3, 0.1])
                alpha_base = 0.4
            cmap = None
        
        # Plot delta as segments with gradient
        for i in range(len(delta_data) - 1):
            color_position = i / max(1, len(delta_data) - 1)
            
            if cmap is not None:
                color = cmap(color_position)
            else:
                color = start_color * (1 - color_position) + end_color * color_position
            
            alpha = alpha_base * (1.0 - 0.3 * color_position)
            
            # Plot segment as dots
            self.ax_delta.plot(time_data[i:i+2], delta_data[i:i+2], 
                             'o', color=color, markersize=3, alpha=alpha,
                             label=label if i == 0 else None)
        
    def plot_controls(self, start_idx, end_idx, time_data):
        """Plot control inputs (steering and acceleration) in the control subplot."""
        # Early return if no control subplot exists
        if self.ax_controls is None or self.data is None:
            return
            
        # Clear the control axes and any twin axes
        self.ax_controls.clear()
        
        # Check if control columns exist in data
        available_control_cols = [col for col in self.control_columns if col in self.data.columns]
        
        if not available_control_cols:
            # If control columns don't exist, disable the checkbox
            if hasattr(self, 'show_controls'):
                self.show_controls.set(False)
            return
        
        # Get control data for the specified range
        control_data = {}
        for col in available_control_cols:
            control_data[col] = self.data[col].iloc[start_idx:end_idx]
        
        # Create time data if not provided or if it's wrong length
        if time_data is None or len(time_data) != (end_idx - start_idx):
            if self.time_column in self.data.columns:
                time_data = self.data[self.time_column].iloc[start_idx:end_idx]
            else:
                time_data = np.arange(start_idx, end_idx)
        
        # Plot steering angle if available
        ax_controls2 = None
        lines_legend = []
        labels_legend = []
        
        # Plot steering control
        if STEERING_CONTROL_COLUMN in control_data:
            line1 = self.ax_controls.plot(time_data, control_data[STEERING_CONTROL_COLUMN], 
                                        'r-', label='Steering Angle', linewidth=2)
            self.ax_controls.set_ylabel('Steering Angle (rad)', color='r')
            self.ax_controls.tick_params(axis='y', labelcolor='r')
            # Add horizontal line at zero for steering angle
            self.ax_controls.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1)
            lines_legend.extend(line1)
            labels_legend.append('Steering Angle')
        
        # Create second y-axis for acceleration if available
        if ACCELERATION_CONTROL_COLUMN in control_data:
            ax_controls2 = self.ax_controls.twinx()
            line2 = ax_controls2.plot(time_data, control_data[ACCELERATION_CONTROL_COLUMN], 
                                    'b-', label='Acceleration', linewidth=2)
            ax_controls2.set_ylabel('Acceleration (m/s²)', color='b')
            ax_controls2.tick_params(axis='y', labelcolor='b')
            # Add horizontal line at zero for acceleration
            ax_controls2.axhline(y=0, color='b', linestyle='--', alpha=0.7, linewidth=1)
            lines_legend.extend(line2)
            labels_legend.append('Acceleration')
        
        # Create combined legend if we have lines
        if lines_legend:
            self.ax_controls.legend(lines_legend, labels_legend, loc='upper right')
        
        # Set title and grid
        self.ax_controls.set_title('Control Inputs')
        self.ax_controls.grid(True, alpha=0.3)
        
        # Set x-axis label and limits (shared with main plot)
        self.ax_controls.set_xlabel('Time (s)' if self.time_column in self.data.columns else 'Step')
        
        # Set x-axis limits to match main plot
        if self.time_column in self.data.columns:
            x_min = self.data[self.time_column].iloc[start_idx]
            x_max = self.data[self.time_column].iloc[end_idx - 1] if end_idx <= len(self.data) else self.data[self.time_column].iloc[-1]
            self.ax_controls.set_xlim(x_min, x_max)
            if ax_controls2 is not None:
                ax_controls2.set_xlim(x_min, x_max)
        else:
            self.ax_controls.set_xlim(start_idx, end_idx - 1)
            if ax_controls2 is not None:
                ax_controls2.set_xlim(start_idx, end_idx - 1)
        
    def update_data_range(self):
        """Update the data range for visualization."""
        if self.data is None:
            return
            
        try:
            start_idx = int(self.start_index_var.get()) if self.start_index_var.get() else 0
            end_idx = int(self.end_index_var.get()) if self.end_index_var.get() else None
            
            # Validate indices
            if start_idx < 0:
                start_idx = 0
            if end_idx is not None and end_idx > len(self.data):
                end_idx = len(self.data)
            if end_idx is not None and start_idx >= end_idx:
                messagebox.showerror("Error", "Start index must be less than end index")
                return
                
            self.start_index = start_idx
            self.end_index = end_idx
            
            # Update comparison slider range based on current data range and horizon
            self.update_comparison_slider_range()
            
            # Save config after updating data range
            self.save_config()
            
            self.plot_state()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid index values. Please enter integers.")
            
    def update_comparison_slider_range(self):
        """Update the comparison slider range based on current data range and horizon."""
        if not hasattr(self, 'comparison_slider') or self.data is None:
            return
            
        try:
            horizon = int(self.horizon_var.get()) if self.horizon_var.get().isdigit() else 50
        except:
            horizon = 50
            
        # Slider should go from start_index to (end_index - horizon)
        effective_end = (self.end_index if self.end_index is not None else len(self.data))
        max_start = max(self.start_index, effective_end - horizon)
        
        if max_start > self.start_index:
            self.comparison_slider.config(from_=self.start_index, to=max_start)
            # Ensure current value is within new range
            current_val = self.comparison_start_var.get()
            if current_val < self.start_index:
                self.comparison_start_var.set(self.start_index)
            elif current_val > max_start:
                self.comparison_start_var.set(max_start)
        else:
            # Not enough data for comparison - set a minimal range
            self.comparison_slider.config(from_=self.start_index, to=self.start_index)
            self.comparison_start_var.set(self.start_index)
            
    def on_comparison_slider_changed(self, value):
        """Handle comparison slider change with delayed plotting for better performance."""
        # Immediately update the index and label for responsive UI feedback
        self.comparison_start_index = int(float(value))
        if hasattr(self, 'comparison_index_label'):
            self.comparison_index_label.config(text=f"Index: {self.comparison_start_index}")
        
        # Cancel any pending update
        if self.slider_update_timer is not None:
            self.root.after_cancel(self.slider_update_timer)
        
        # Schedule a delayed update - only triggers if user stops moving slider
        self.slider_update_timer = self.root.after(
            self.slider_update_delay, 
            self._delayed_slider_update
        )
    
    def _delayed_slider_update(self):
        """Perform the actual heavy update after slider stops moving."""
        self.slider_update_timer = None  # Clear the timer reference
        
        # If comparison is enabled and we don't have prediction for this index, compute it
        if self.enable_comparison.get() and self.comparison_start_index not in self.comparison_data_dict:
            if self.run_single_comparison(self.comparison_start_index):
                self.plot_state()
                self.save_config()
        elif self.enable_comparison.get():
            # We have the data, just replot
            self.plot_state()
            self.save_config()
            
    def on_scroll(self, event):
        """Handle mouse scroll for zooming."""
        if self.ax is None or event.inaxes != self.ax:
            return
            
        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Scale factor for zoom
        scale_factor = 1.1 if event.button == 'down' else 1/1.1
        
        # Calculate new limits centered on mouse position
        xdata, ydata = event.xdata, event.ydata
        if xdata is not None and ydata is not None:
            new_width = (xlim[1] - xlim[0]) * scale_factor
            new_height = (ylim[1] - ylim[0]) * scale_factor
            
            relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
            rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])
            
            new_xlim = (xdata - new_width * relx, xdata + new_width * (1-relx))
            new_ylim = (ydata - new_height * rely, ydata + new_height * (1-rely))
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            if self.canvas is not None:
                self.canvas.draw()
    
    def on_button_press(self, event):
        """Handle mouse button press for panning."""
        if self.ax is None or event.inaxes != self.ax or event.button != 1:  # Only left mouse button
            return
        self.pan_start = (event.xdata, event.ydata)
        self.is_panning = True
        
    def on_mouse_move(self, event):
        """Handle mouse move for panning."""
        if self.ax is None or not self.is_panning or self.pan_start is None or event.inaxes != self.ax:
            return
            
        if event.xdata is not None and event.ydata is not None:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            self.ax.set_xlim((xlim[0] - dx, xlim[1] - dx))
            self.ax.set_ylim((ylim[0] - dy, ylim[1] - dy))
            
            if self.canvas is not None:
                self.canvas.draw()
    
    def on_button_release(self, event):
        """Handle mouse button release."""
        self.is_panning = False
        self.pan_start = None
        
    def save_plot(self):
        """Save the current plot to a file."""
        if self.fig is None:
            messagebox.showerror("Error", "No plot to save.")
            return
            
        # Get the current state name for default filename
        selected_state = self.state_var.get() if hasattr(self, 'state_var') else "plot"
        default_filename = f"state_comparison_{selected_state}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Open file dialog to choose save location
        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("EPS files", "*.eps"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Save the figure with current DPI settings for consistent quality
                current_dpi = matplotlib.rcParams['savefig.dpi']
                self.fig.savefig(file_path, dpi=current_dpi, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                messagebox.showinfo("Success", f"Plot saved to: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
        
    def get_car_parameters_filename(self, display_name):
        """Get the actual filename from display name."""
        return self.common.get_car_parameters_filename(display_name)

    def _get_selected_param_file_path(self):
        """Resolve absolute path of currently selected car parameter file."""
        try:
            filename = self.get_car_parameters_filename(self.params_var.get()) if hasattr(self, 'params_var') else None
            if not filename:
                return None
            car_dir = os.path.join(parent_dir, 'utilities', 'car_files')
            return os.path.join(car_dir, filename)
        except Exception:
            return None

    def _update_param_status_label(self):
        """Update UI label with a concise summary of current car parameters."""
        try:
            path = self._get_selected_param_file_path()
            if not path or not os.path.exists(path):
                if hasattr(self, 'param_status_label'):
                    self.param_status_label.config(text="Parameters: (none)")
                return
            # Load raw params for readable names
            from utilities.car_files.vehicle_parameters import VehicleParameters
            vp = VehicleParameters(os.path.basename(path))
            d = vp.to_dict()
            # Build concise summary
            summary = (
                f"{os.path.basename(path)} | mu={d.get('mu'):.3f}, lf={d.get('lf'):.3f}, lr={d.get('lr'):.3f}, m={d.get('m'):.1f}, "
                f"Bf={d.get('C_Pf')[0]:.3f}, Cf={d.get('C_Pf')[1]:.2f}, Dr={d.get('C_Pr')[2]:.2f}, c_rr={d.get('c_rr'):.4f}, brake_mult={d.get('brake_multiplier'):.2f}"
            )
            self.param_status_label.config(text=summary)
        except Exception:
            if hasattr(self, 'param_status_label'):
                self.param_status_label.config(text="Parameters: (unavailable)")
        
    def get_model_key(self, display_name):
        """Get the model key from display name."""
        return self.common.get_model_key(display_name)
        
    def load_car_parameters(self, param_file):
        """Load car parameters from YAML file."""
        result = self.common.load_car_parameters(param_file)
        if result is None:
            messagebox.showerror("Error", f"Error loading car parameters from {param_file}")
        return result

    def _init_param_file_watch(self):
        """Start periodic watch for changes to the selected car parameter file."""
        self._param_watch_timer = None
        self._param_file_path = None
        self._param_file_mtime = None
        self._reset_param_file_watch()
        self._update_param_status_label()
        self._schedule_param_watch()

    def _reset_param_file_watch(self):
        """Reset watch state to the currently selected parameter file."""
        path = self._get_selected_param_file_path()
        self._param_file_path = path
        try:
            self._param_file_mtime = os.path.getmtime(path) if path and os.path.exists(path) else None
        except Exception:
            self._param_file_mtime = None
        self._update_param_status_label()

    def _schedule_param_watch(self):
        """Schedule next parameter file check."""
        try:
            # Check every 1s; lightweight and avoids external deps
            self._param_watch_timer = self.root.after(1000, self._check_param_file_changed)
        except Exception:
            self._param_watch_timer = None

    def _check_param_file_changed(self):
        """Detect parameter file changes and refresh predictions if needed."""
        try:
            current_path = self._get_selected_param_file_path()
            if current_path != self._param_file_path:
                # Selection changed; reset base and continue
                self._reset_param_file_watch()
            else:
                if current_path and os.path.exists(current_path):
                    mtime = os.path.getmtime(current_path)
                    if self._param_file_mtime is not None and mtime > self._param_file_mtime:
                        self._param_file_mtime = mtime
                        self._on_param_file_changed()
        finally:
            self._schedule_param_watch()

    def _on_param_file_changed(self):
        """Handle car parameter file updates by recomputing predictions."""
        # Clear cached comparisons
        if hasattr(self, 'comparison_data_dict'):
            self.comparison_data_dict = {}

        # If comparison is enabled, recompute
        if self.enable_comparison.get():
            if hasattr(self, 'show_all_comparisons') and self.show_all_comparisons.get():
                self.run_full_comparison()
                return
            comp_idx = self.comparison_start_var.get() if hasattr(self, 'comparison_start_var') else 0
            if self.run_single_comparison(comp_idx):
                self.plot_state()
        # Update status label
        self._update_param_status_label()
            
    def _validate_comparison_requirements(self):
        """Validate that comparison can be run."""
        if self.data is None:
            messagebox.showerror("Error", "No data loaded. Please load a CSV file first.")
            return False
            
        try:
            horizon = int(self.horizon_var.get())
            if horizon <= 0:
                raise ValueError("Horizon must be positive")
        except (ValueError, AttributeError):
            messagebox.showerror("Error", "Invalid horizon value. Please enter a positive integer.")
            return False
            
        return horizon
            
    def run_single_comparison(self, start_index):
        """Run model comparison for a single start index with control delay support."""
        horizon = self._validate_comparison_requirements()
        if not horizon:
            return False
            
        try:
            # Get selected model and parameters
            model_name = self.get_model_key(self.model_var.get())
            param_file = self.get_car_parameters_filename(self.params_var.get())
            
            # Get delays from UI
            try:
                steering_delay = int(self.steering_delay_var.get()) if hasattr(self, 'steering_delay_var') and self.steering_delay_var.get().isdigit() else 0
            except:
                steering_delay = 0
            
            try:
                acceleration_delay = int(self.acceleration_delay_var.get()) if hasattr(self, 'acceleration_delay_var') and self.acceleration_delay_var.get().isdigit() else 0
            except:
                acceleration_delay = 0
            
            # Sync data to common
            self.common.data = self.data
            self.common._current_start_index = start_index
            
            # Run comparison using common utilities
            result = self.common.run_single_comparison(start_index, model_name, param_file, horizon, steering_delay, acceleration_delay)
            if result is None:
                return False
            
            # Store comparison data for this start index
            self.comparison_data_dict[start_index] = result
            return True
            
        except Exception as e:
            print(f"Single comparison failed for index {start_index}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _run_model_prediction(self, model_name, initial_state, control_sequence, car_params, dt, horizon):
        """Run model prediction based on model name (delegates to common)."""
        self.common.data = self.data
        return self.common.run_model_prediction(model_name, initial_state, control_sequence, car_params, dt, horizon, self.data)
            
    def run_full_comparison(self):
        """Run model comparison for multiple start indices to enable sliding comparison."""
        import importlib
        import sys
        # Force reload car_model_jax and dynamic_model_pacejka_jax using importlib.reload
        if 'sim.f110_sim.envs.dynamic_model_pacejka_jax' in sys.modules:
            importlib.reload(sys.modules['sim.f110_sim.envs.dynamic_model_pacejka_jax'])
        else:
            import sim.f110_sim.envs.dynamic_model_pacejka_jax
        # Reload modules before running comparison
        self._reload_car_models()

        horizon = self._validate_comparison_requirements()
        if not horizon:
            return

        # Force reload car parameters at the start to ensure we have the latest values
        param_file = self.get_car_parameters_filename(self.params_var.get())
        test_params = self.load_car_parameters(param_file)
        if test_params is None:
            messagebox.showerror("Error", "Failed to load car parameters.")
            return

        print(f"Car parameters reloaded from {param_file}")

        # Clear previous comparison data
        self.comparison_data_dict = {}

        # Determine range of start indices based on current data range settings
        effective_start = self.start_index if self.start_index is not None else 0
        effective_end = self.end_index if self.end_index is not None else len(self.data) if self.data is not None else 0
        max_start_idx = effective_end - horizon

        if max_start_idx <= effective_start:
            messagebox.showerror("Error", "Horizon is larger than available data in current range.")
            return

        # Use a reasonable step size for performance within the current data range
        range_size = max_start_idx - effective_start
        step_size = max(1, range_size // 100)  # Compute ~100 predictions max
        start_indices = list(range(effective_start, max_start_idx, step_size))

        # Show progress
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Computing Predictions...")
        progress_window.geometry("400x100")
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(start_indices))
        progress_bar.pack(pady=20, padx=20, fill=tk.X)
        progress_label = ttk.Label(progress_window, text="Starting...")
        progress_label.pack()
        progress_window.update()

        try:
            successful_comparisons = 0
            for i, start_idx in enumerate(start_indices):
                # Update progress
                progress_var.set(i)
                progress_label.config(text=f"Computing prediction {i+1}/{len(start_indices)} (start index: {start_idx})")
                progress_window.update()
                # Run single comparison for this index
                if self.run_single_comparison(start_idx):
                    successful_comparisons += 1
            progress_window.destroy()
            if successful_comparisons == 0:
                messagebox.showerror("Error", "No successful comparisons computed.")
                return
            # Enable comparison checkbox
            self.enable_comparison.set(True)
            # Update slider range properly using the existing method that respects data constraints
            self.update_comparison_slider_range()
            # If we have computed data, set the slider to a reasonable starting position within current range
            if self.comparison_data_dict:
                computed_indices = list(self.comparison_data_dict.keys())
                current_min = self.comparison_slider.cget('from')
                current_max = self.comparison_slider.cget('to')
                # Find the first computed index that's within the current slider range
                valid_indices = [idx for idx in computed_indices if current_min <= idx <= current_max]
                if valid_indices:
                    self.comparison_start_var.set(min(valid_indices))
                    self.comparison_start_index = min(valid_indices)
                else:
                    # If no computed indices are in range, set to the start of the range
                    self.comparison_start_var.set(int(current_min))
                    self.comparison_start_index = int(current_min)
            # Refresh plot
            self.plot_state()
            print(f"Full comparison completed. {successful_comparisons}/{len(start_indices)} successful comparisons.")
            print(f"Available indices: {list(self.comparison_data_dict.keys())}")
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Error", f"Model comparison failed: {str(e)}")
            print(f"Detailed error: {e}")
            import traceback
            traceback.print_exc()
            
    def extract_initial_state(self):
        """Extract initial state from the data."""
        return self.extract_initial_state_at_index(0)
        
    def extract_initial_state_at_index(self, index):
        """Extract initial state from the data at a specific index."""
        # Sync data to common before extracting
        self.common.data = self.data
        return self.common.extract_initial_state_at_index(index)
        
    def extract_control_sequence(self, horizon):
        """Extract control sequence from the data."""
        return self.extract_control_sequence_at_index(0, horizon)
        
    def extract_control_sequence_at_index(self, start_index, horizon):
        """Extract control sequence from the data starting at a specific index with separate control delays."""
        # Get delays from UI
        try:
            steering_delay = int(self.steering_delay_var.get()) if hasattr(self, 'steering_delay_var') and self.steering_delay_var.get().isdigit() else 0
        except:
            steering_delay = 0
        
        try:
            acceleration_delay = int(self.acceleration_delay_var.get()) if hasattr(self, 'acceleration_delay_var') and self.acceleration_delay_var.get().isdigit() else 0
        except:
            acceleration_delay = 0
        
        # Sync data to common and use its method
        self.common.data = self.data
        return self.common.extract_control_sequence_at_index(start_index, horizon, steering_delay, acceleration_delay)
        
    def get_timestep(self):
        """Get timestep from data or use default."""
        self.common.data = self.data
        self.common.time_column = self.time_column
        return self.common.get_timestep()
        
    def convert_predictions_to_dict(self, predicted_states):
        """Convert JAX predictions to dictionary format matching CSV columns."""
        return self.common.convert_predictions_to_dict(predicted_states)
    
    def calculate_error_metrics(self, ground_truth, prediction, state_name):
        """Calculate error metrics between ground truth and prediction for a given state."""
        if len(ground_truth) == 0 or len(prediction) == 0:
            return None
        
        # Ensure both arrays have the same length for comparison
        min_length = min(len(ground_truth), len(prediction))
        gt_data = np.array(ground_truth[:min_length])
        pred_data = np.array(prediction[:min_length])
        
        # Calculate error (deviation from ground truth to prediction)
        error = pred_data - gt_data
        
        # Calculate metrics
        mean_error = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        error_std = np.std(error)
        
        return {
            'mean_error': mean_error,
            'max_error': max_error,
            'error_std': error_std,
            'rmse': np.sqrt(np.mean(error**2))
        }
    
    def get_ground_truth_for_comparison(self, start_idx, horizon, state_name):
        """Get ground truth data for comparison with prediction."""
        if self.data is None or state_name not in self.data.columns:
            return None
        
        end_idx = min(start_idx + horizon, len(self.data))
        return self.data[state_name].iloc[start_idx:end_idx].values
    
    def calculate_metrics_for_displayed_predictions(self, state_name):
        """Calculate error metrics for the predictions that are actually displayed."""
        if self.data is None or state_name not in self.data.columns:
            return None
        
        if not hasattr(self, 'comparison_data_dict') or not self.comparison_data_dict:
            return None
        
        # Get the current visible range
        start_idx = self.start_index
        end_idx = self.end_index if self.end_index is not None else len(self.data)
        
        if start_idx >= end_idx:
            return None
        
        # Collect all predictions and corresponding ground truth that are displayed
        all_predictions = []
        all_ground_truth = []
        
        try:
            # Check if showing all comparisons or single comparison
            show_all = hasattr(self, 'show_all_comparisons') and self.show_all_comparisons.get()
            
            if show_all:
                # Calculate errors for all displayed predictions
                for comp_start_idx, comparison_data in self.comparison_data_dict.items():
                    # Only include predictions that start within the current visible range
                    if comp_start_idx < start_idx or comp_start_idx >= end_idx:
                        continue
                    
                    if state_name in comparison_data:
                        prediction = np.array(comparison_data[state_name])
                        horizon = len(prediction)
                        
                        # Get ground truth for this prediction's time range
                        pred_end_idx = min(comp_start_idx + horizon, len(self.data))
                        gt_slice = self.data[state_name].iloc[comp_start_idx:pred_end_idx].values
                        
                        # Ensure same length
                        min_len = min(len(prediction), len(gt_slice))
                        all_predictions.extend(prediction[:min_len])
                        all_ground_truth.extend(gt_slice[:min_len])
            else:
                # Calculate errors for single displayed prediction
                comp_start_idx = self.comparison_start_var.get() if hasattr(self, 'comparison_start_var') else 0
                
                if comp_start_idx in self.comparison_data_dict:
                    comparison_data = self.comparison_data_dict[comp_start_idx]
                    
                    if state_name in comparison_data:
                        prediction = np.array(comparison_data[state_name])
                        horizon = len(prediction)
                        
                        # Get ground truth for this prediction's time range
                        pred_end_idx = min(comp_start_idx + horizon, len(self.data))
                        gt_slice = self.data[state_name].iloc[comp_start_idx:pred_end_idx].values
                        
                        # Ensure same length
                        min_len = min(len(prediction), len(gt_slice))
                        all_predictions.extend(prediction[:min_len])
                        all_ground_truth.extend(gt_slice[:min_len])
            
            if len(all_predictions) == 0:
                return None
            
            # Convert to numpy arrays
            pred_data = np.array(all_predictions)
            gt_data = np.array(all_ground_truth)
            
            # Calculate error (deviation from ground truth to prediction)
            error = pred_data - gt_data
            
            # Calculate metrics
            mean_error = np.mean(np.abs(error))
            max_error = np.max(np.abs(error))
            error_std = np.std(error)
            rmse = np.sqrt(np.mean(error**2))
            
            return {
                'mean_error': mean_error,
                'max_error': max_error,
                'error_std': error_std,
                'rmse': rmse
            }
            
        except Exception as e:
            print(f"Error calculating metrics for displayed predictions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_metrics_for_entire_range(self, state_name):
        """Calculate error metrics for the entire data range using model predictions.
        NOTE: This function is kept for backward compatibility but is no longer used.
        Use calculate_metrics_for_displayed_predictions instead."""
        if self.data is None or state_name not in self.data.columns:
            return None
        
        # Get the current data range
        start_idx = self.start_index
        end_idx = self.end_index if self.end_index is not None else len(self.data)
        
        if start_idx >= end_idx:
            return None
        
        # Get ground truth data for the entire range
        ground_truth = self.data[state_name].iloc[start_idx:end_idx].values
        
        # Generate model predictions for the entire range
        # We'll use a sliding window approach to get predictions for each timestep
        predictions = []
        
        try:
            # Get model parameters
            model_name = self.get_model_key(self.model_var.get())
            param_file = self.get_car_parameters_filename(self.params_var.get())
            car_params = self.load_car_parameters(param_file)
            
            if car_params is None:
                return None
            
            # Use a reasonable horizon for individual predictions (e.g., 10 steps)
            prediction_horizon = min(10, end_idx - start_idx)
            
            # Generate predictions for each timestep in the range
            for i in range(start_idx, end_idx - prediction_horizon + 1):
                # Extract initial state and control sequence
                initial_state = self.extract_initial_state_at_index(i)
                control_sequence = self.extract_control_sequence_at_index(i, prediction_horizon)
                dt = self.get_timestep()
                
                # Run model prediction
                predicted_states = self._run_model_prediction(model_name, initial_state, control_sequence, car_params, dt, prediction_horizon)
                
                if predicted_states is not None:
                    # Get the state index for the selected state
                    state_idx = self.state_indices.get(state_name, 0)
                    # Take only the first prediction step (next timestep)
                    predictions.append(predicted_states[0, state_idx])
                else:
                    # If prediction fails, use the current ground truth value
                    predictions.append(ground_truth[i - start_idx])
            
            # Pad predictions to match ground truth length if needed
            while len(predictions) < len(ground_truth):
                predictions.append(ground_truth[len(predictions)])
            
            # Ensure both arrays have the same length
            min_length = min(len(ground_truth), len(predictions))
            gt_data = np.array(ground_truth[:min_length])
            pred_data = np.array(predictions[:min_length])
            
            # Calculate error (deviation from ground truth to prediction)
            error = pred_data - gt_data
            
            # Calculate metrics
            mean_error = np.mean(np.abs(error))
            max_error = np.max(np.abs(error))
            error_std = np.std(error)
            rmse = np.sqrt(np.mean(error**2))
            
            return {
                'mean_error': mean_error,
                'max_error': max_error,
                'error_std': error_std,
                'rmse': rmse
            }
            
        except Exception as e:
            print(f"Error calculating metrics for entire range: {e}")
            return None
        
    def calculate_error_metrics(self, ground_truth, prediction, state_name):
        """Calculate error metrics between ground truth and prediction for a given state."""
        self.common.data = self.data
        return self.common.calculate_error_metrics(ground_truth, prediction, state_name)
    
    def get_ground_truth_for_comparison(self, start_idx, horizon, state_name):
        """Get ground truth data for comparison with prediction."""
        if self.data is None or state_name not in self.data.columns:
            return None
        
        end_idx = min(start_idx + horizon, len(self.data))
        return self.data[state_name].iloc[start_idx:end_idx].values
    
    # Removed plot_offline_mode and related offline methods - they are now in visualization_offline.py
    def _removed_offline_methods(self):
        """Generate and save all plots for all states in offline mode (no UI)."""
        print("Starting offline plotting mode...")
        
        # Ensure data is loaded
        if self.data is None:
            print("Error: No CSV file loaded. Please set 'csv_file_path' in config.")
            return
        
        # Get model and parameters from config or use defaults
        model_name_key = None
        param_file = None
        
        # Try to get from config
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r') as f:
                    config = json.load(f)
                    default_model = config.get('default_car_model', '')
                    default_params = config.get('default_car_parameters', '')
                    
                    # Find model key
                    for key, value in self.available_models.items():
                        if value == default_model:
                            model_name_key = key
                            break
                    
                    # Find parameter file
                    if default_params in self.available_car_params.values():
                        param_file = default_params
            except Exception as e:
                print(f"Warning: Could not read config: {e}")
        
        # Use defaults if not set
        if model_name_key is None:
            model_name_key = list(self.available_models.keys())[0]
        if param_file is None:
            param_file = list(self.available_car_params.values())[0]
        
        print(f"Using model: {self.available_models[model_name_key]}, parameters: {param_file}")
        
        # Get settings from config or use defaults
        horizon = self.horizon_steps if hasattr(self, 'horizon_steps') else 50
        steering_delay = self.steering_delay_steps if hasattr(self, 'steering_delay_steps') else 2
        acceleration_delay = self.acceleration_delay_steps if hasattr(self, 'acceleration_delay_steps') else 2
        
        # Load car parameters
        car_params = self.load_car_parameters(param_file)
        if car_params is None:
            print(f"Error: Failed to load car parameters from {param_file}")
            return
        
        # Determine range for comparisons
        effective_start = self.start_index if self.start_index is not None else 0
        effective_end = self.end_index if self.end_index is not None else len(self.data)
        max_start_idx = effective_end - horizon
        
        if max_start_idx <= effective_start:
            print(f"Error: Horizon ({horizon}) is larger than available data range.")
            return
        
        # Compute predictions for all start indices (with reasonable step size for performance)
        range_size = max_start_idx - effective_start
        step_size = max(1, range_size // 100)  # Compute ~100 predictions max
        start_indices = list(range(effective_start, max_start_idx, step_size))
        
        print(f"Computing {len(start_indices)} predictions...")
        self.comparison_data_dict = {}
        
        # Set model and params for offline mode (use attributes directly, no UI vars)
        self._offline_model_key = model_name_key
        self._offline_param_file = param_file
        self._offline_steering_delay = steering_delay
        self._offline_acceleration_delay = acceleration_delay
        
        # Run comparisons
        successful = 0
        for i, start_idx in enumerate(start_indices):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(start_indices)}")
            
            if self._run_single_comparison_offline(start_idx, horizon):
                successful += 1
        
        print(f"Completed {successful}/{len(start_indices)} successful predictions.")
        
        # Get available states
        available_states = [col for col in self.state_columns if col in self.data.columns]
        
        if not available_states:
            print("Error: No valid state columns found in CSV.")
            return
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(self.csv_file_path) if self.csv_file_path else '.', 'plots_offline')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating plots for {len(available_states)} states...")
        
        # Generate and save plots for each state
        for state_idx, state_name in enumerate(available_states):
            print(f"  Plotting {state_name} ({state_idx+1}/{len(available_states)})...")
            
            # Create figure with multiple subplots (main, delta, controls)
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 1, hspace=0.3)
            
            # Main plot
            ax_main = fig.add_subplot(gs[0, 0])
            
            # Plot ground truth
            start_idx = self.start_index
            end_idx = self.end_index if self.end_index is not None else len(self.data)
            
            if self.time_column in self.data.columns:
                time_data = self.data[self.time_column].iloc[start_idx:end_idx]
            else:
                time_data = np.arange(start_idx, end_idx)
            
            state_data = self.data[state_name].iloc[start_idx:end_idx]
            ax_main.plot(time_data, state_data, 'k-', label='Ground Truth', linewidth=2)
            
            # Plot all predictions
            for comp_start_idx, comparison_data in self.comparison_data_dict.items():
                if comp_start_idx < start_idx or comp_start_idx >= end_idx:
                    continue
                
                if state_name in comparison_data:
                    full_horizon = len(comparison_data[state_name])
                    
                    # Generate time data
                    if self.time_column in self.data.columns:
                        if comp_start_idx + full_horizon <= len(self.data):
                            comp_time = self.data[self.time_column].iloc[comp_start_idx:comp_start_idx + full_horizon]
                        else:
                            available_time = self.data[self.time_column].iloc[comp_start_idx:]
                            dt = self.get_timestep()
                            missing_steps = full_horizon - len(available_time)
                            if len(available_time) > 0:
                                last_time = available_time.iloc[-1]
                                extra_time = np.arange(1, missing_steps + 1) * dt + last_time
                                comp_time = np.concatenate([available_time.to_numpy(), extra_time])
                            else:
                                comp_time = np.arange(comp_start_idx, comp_start_idx + full_horizon) * dt
                    else:
                        dt = self.get_timestep()
                        comp_time = np.arange(comp_start_idx, comp_start_idx + full_horizon) * dt
                    
                    full_prediction = np.array(comparison_data[state_name])
                    self._plot_prediction_with_gradient_offline(ax_main, comp_time, full_prediction, 
                                                                full_horizon, 
                                                                label='Model Predictions' if comp_start_idx == list(self.comparison_data_dict.keys())[0] else None)
            
            ax_main.set_xlabel('Time (s)' if self.time_column in self.data.columns else 'Step')
            ax_main.set_ylabel(state_name.replace('_', ' ').title())
            ax_main.set_title(f'State Comparison: {state_name}')
            ax_main.legend()
            ax_main.grid(True, alpha=0.3)
            if self.time_column in self.data.columns:
                x_min = self.data[self.time_column].iloc[start_idx]
                x_max = self.data[self.time_column].iloc[end_idx - 1] if end_idx <= len(self.data) else self.data[self.time_column].iloc[-1]
                ax_main.set_xlim(x_min, x_max)
            else:
                ax_main.set_xlim(start_idx, end_idx - 1)
            
            # Delta plot
            ax_delta = fig.add_subplot(gs[1, 0], sharex=ax_main)
            state_data_values = self.data[state_name].iloc[start_idx:end_idx].values
            if len(state_data_values) > 1:
                if state_name == 'pose_theta':
                    delta_raw = np.diff(state_data_values)
                    delta_gt = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
                else:
                    delta_gt = np.diff(state_data_values)
                delta_time = time_data.iloc[:-1].values if hasattr(time_data, 'iloc') else time_data[:-1]
                ax_delta.plot(delta_time, delta_gt, 'k-', label='Ground Truth Delta', linewidth=2)
                
                # Plot delta for predictions
                for comp_count, (comp_start_idx, comparison_data) in enumerate(self.comparison_data_dict.items()):
                    if comp_start_idx < start_idx or comp_start_idx >= end_idx or state_name not in comparison_data:
                        continue
                    prediction = np.array(comparison_data[state_name])
                    if len(prediction) > 1:
                        if state_name == 'pose_theta':
                            delta_raw = np.diff(prediction)
                            delta_pred = np.arctan2(np.sin(delta_raw), np.cos(delta_raw))
                        else:
                            delta_pred = np.diff(prediction)
                        # Time data for delta
                        if self.time_column in self.data.columns:
                            if comp_start_idx + len(prediction) <= len(self.data):
                                comp_time = self.data[self.time_column].iloc[comp_start_idx:comp_start_idx + len(prediction)]
                            else:
                                available_time = self.data[self.time_column].iloc[comp_start_idx:]
                                dt = self.get_timestep()
                                missing_steps = len(prediction) - len(available_time)
                                if len(available_time) > 0:
                                    last_time = available_time.iloc[-1]
                                    extra_time = np.arange(1, missing_steps + 1) * dt + last_time
                                    comp_time = np.concatenate([available_time.to_numpy(), extra_time])
                                else:
                                    comp_time = np.arange(comp_start_idx, comp_start_idx + len(prediction)) * dt
                        else:
                            dt = self.get_timestep()
                            comp_time = np.arange(comp_start_idx, comp_start_idx + len(prediction)) * dt
                        delta_pred_time = comp_time.iloc[:-1].values if hasattr(comp_time, 'iloc') else comp_time[:-1]
                        self._plot_delta_prediction_with_gradient_offline(ax_delta, delta_pred_time, delta_pred, 
                                                                          len(delta_pred), 
                                                                          label='Model Prediction Delta' if comp_count == 0 else None)
            
            ax_delta.set_xlabel('Time (s)' if self.time_column in self.data.columns else 'Step')
            ax_delta.set_ylabel(f'Δ {state_name.replace("_", " ").title()}')
            ax_delta.set_title(f'Delta State: {state_name}')
            ax_delta.legend()
            ax_delta.grid(True, alpha=0.3)
            
            # Controls plot
            ax_controls = fig.add_subplot(gs[2, 0], sharex=ax_main)
            if STEERING_CONTROL_COLUMN in self.data.columns:
                steering_data = self.data[STEERING_CONTROL_COLUMN].iloc[start_idx:end_idx]
                ax_controls.plot(time_data, steering_data, 'r-', label='Steering Angle', linewidth=2)
                ax_controls.set_ylabel('Steering Angle (rad)', color='r')
                ax_controls.tick_params(axis='y', labelcolor='r')
                ax_controls.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1)
            
            if ACCELERATION_CONTROL_COLUMN in self.data.columns:
                ax_controls2 = ax_controls.twinx()
                accel_data = self.data[ACCELERATION_CONTROL_COLUMN].iloc[start_idx:end_idx]
                ax_controls2.plot(time_data, accel_data, 'b-', label='Acceleration', linewidth=2)
                ax_controls2.set_ylabel('Acceleration (m/s²)', color='b')
                ax_controls2.tick_params(axis='y', labelcolor='b')
                ax_controls2.axhline(y=0, color='b', linestyle='--', alpha=0.7, linewidth=1)
                
                # Combined legend
                lines1, labels1 = ax_controls.get_legend_handles_labels()
                lines2, labels2 = ax_controls2.get_legend_handles_labels()
                ax_controls.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            else:
                ax_controls.legend()
            
            ax_controls.set_xlabel('Time (s)' if self.time_column in self.data.columns else 'Step')
            ax_controls.set_title('Control Inputs')
            ax_controls.grid(True, alpha=0.3)
            
            # Save figure
            safe_state_name = state_name.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f'{safe_state_name}_comparison.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
        
        print(f"All plots saved to: {output_dir}")
    
    def _run_single_comparison_offline(self, start_index, horizon):
        """Run single comparison in offline mode (without UI variables)."""
        try:
            # Get model and parameters
            model_name = self._offline_model_key
            param_file = self._offline_param_file
            
            # Load car parameters
            car_params = self.load_car_parameters(param_file)
            if car_params is None:
                return False
            
            # Prepare initial state and controls with delay
            initial_state = self.extract_initial_state_at_index(start_index)
            
            # Use offline delay values
            original_steering_delay = getattr(self, '_offline_steering_delay', 2)
            original_accel_delay = getattr(self, '_offline_acceleration_delay', 2)
            
            # Temporarily set delays (we'll extract control sequence manually)
            control_sequence = self._extract_control_sequence_offline(start_index, horizon, 
                                                                      original_steering_delay, original_accel_delay)
            
            # Store start_index for residual model
            self._current_start_index = start_index
            
            # Get timestep
            dt = self.get_timestep()
            
            # Run model prediction
            predicted_states = self._run_model_prediction(model_name, initial_state, control_sequence, car_params, dt, horizon)
            if predicted_states is None:
                return False
            
            # Store comparison data
            self.comparison_data_dict[start_index] = self.convert_predictions_to_dict(predicted_states)
            return True
            
        except Exception as e:
            print(f"  Single comparison failed for index {start_index}: {e}")
            return False
    
    def _extract_control_sequence_offline(self, start_index, horizon, steering_delay, acceleration_delay):
        """Extract control sequence in offline mode with specified delays."""
        if self.data is None:
            return jnp.zeros((horizon, 2), dtype=jnp.float32)
        
        control_sequence = np.zeros((horizon, 2), dtype=np.float32)
        
        # Steering control
        steering_start_index = start_index - steering_delay
        steering_end_index = min(steering_start_index + horizon, len(self.data))
        
        if steering_start_index >= 0 and steering_start_index < len(self.data):
            actual_horizon = max(0, steering_end_index - steering_start_index)
            if actual_horizon > 0 and STEERING_CONTROL_COLUMN in self.data.columns:
                control_data = self.data[STEERING_CONTROL_COLUMN].iloc[steering_start_index:steering_end_index].values
                control_sequence[:actual_horizon, 0] = control_data
        else:
            available_start = max(0, steering_start_index)
            available_end = min(available_start + horizon, len(self.data))
            if available_end > available_start and STEERING_CONTROL_COLUMN in self.data.columns:
                first_control = self.data[STEERING_CONTROL_COLUMN].iloc[0]
                offset = max(0, -steering_start_index)
                actual_length = min(horizon - offset, available_end - available_start)
                if actual_length > 0:
                    control_sequence[offset:offset + actual_length, 0] = self.data[STEERING_CONTROL_COLUMN].iloc[available_start:available_start + actual_length].values
                if offset > 0:
                    control_sequence[:offset, 0] = first_control
        
        # Acceleration control
        acceleration_start_index = start_index - acceleration_delay
        acceleration_end_index = min(acceleration_start_index + horizon, len(self.data))
        
        if acceleration_start_index >= 0 and acceleration_start_index < len(self.data):
            actual_horizon = max(0, acceleration_end_index - acceleration_start_index)
            if actual_horizon > 0 and ACCELERATION_CONTROL_COLUMN in self.data.columns:
                control_data = self.data[ACCELERATION_CONTROL_COLUMN].iloc[acceleration_start_index:acceleration_end_index].values
                control_sequence[:actual_horizon, 1] = control_data
        else:
            available_start = max(0, acceleration_start_index)
            available_end = min(available_start + horizon, len(self.data))
            if available_end > available_start and ACCELERATION_CONTROL_COLUMN in self.data.columns:
                first_control = self.data[ACCELERATION_CONTROL_COLUMN].iloc[0]
                offset = max(0, -acceleration_start_index)
                actual_length = min(horizon - offset, available_end - available_start)
                if actual_length > 0:
                    control_sequence[offset:offset + actual_length, 1] = self.data[ACCELERATION_CONTROL_COLUMN].iloc[available_start:available_start + actual_length].values
                if offset > 0:
                    control_sequence[:offset, 1] = first_control
        
        return jnp.array(control_sequence)
    
    def _plot_prediction_with_gradient_offline(self, ax, time_data, prediction_data, horizon, label=None):
        """Plot prediction with color gradient in offline mode."""
        try:
            from matplotlib.colors import LinearSegmentedColormap
            colors = ["#5200F5", "#FF00BF", "#FF0000"]
            alpha_base = 0.4
            cmap = LinearSegmentedColormap.from_list("gradient", colors, N=100)
        except ImportError:
            start_color = np.array([1.0, 0.4, 0.4])
            end_color = np.array([0.5, 0.0, 0.0])
            alpha_base = 0.8
            cmap = None
        
        if hasattr(time_data, 'iloc'):
            time_data = time_data.values
        elif hasattr(time_data, 'to_numpy'):
            time_data = time_data.to_numpy()
        time_data = np.asarray(time_data)
        
        if len(prediction_data) == 1:
            if cmap is not None:
                color = cmap(0.0)
            else:
                color = start_color
            alpha = alpha_base
            ax.plot(time_data[0], prediction_data[0], 'o', color=color, markersize=3, alpha=alpha, label=label)
        else:
            for i in range(len(prediction_data) - 1):
                color_position = i / max(1, len(prediction_data) - 1)
                if cmap is not None:
                    color = cmap(color_position)
                else:
                    color = start_color * (1 - color_position) + end_color * color_position
                alpha = alpha_base * (1.0 - 0.3 * color_position)
                ax.plot(time_data[i:i+2], prediction_data[i:i+2], 'o', color=color, markersize=3, 
                       alpha=alpha, label=label if i == 0 else None)
    
    def _plot_delta_prediction_with_gradient_offline(self, ax, time_data, delta_data, horizon, label=None):
        """Plot delta prediction with gradient in offline mode."""
        try:
            from matplotlib.colors import LinearSegmentedColormap
            colors = ["#5200F5", "#FF00BF", "#FF0000"]
            alpha_base = 0.4
            cmap = LinearSegmentedColormap.from_list("gradient", colors, N=100)
        except ImportError:
            start_color = np.array([1.0, 0.4, 0.4])
            end_color = np.array([0.5, 0.0, 0.0])
            alpha_base = 0.8
            cmap = None
        
        for i in range(len(delta_data) - 1):
            color_position = i / max(1, len(delta_data) - 1)
            if cmap is not None:
                color = cmap(color_position)
            else:
                color = start_color * (1 - color_position) + end_color * color_position
            alpha = alpha_base * (1.0 - 0.3 * color_position)
            ax.plot(time_data[i:i+2], delta_data[i:i+2], 'o', color=color, markersize=3, 
                   alpha=alpha, label=label if i == 0 else None)
    
    def run(self):
        """Start the application."""
        # Only auto-load if no CSV was loaded from config
        if self.data is None:
            # Set specific recording file to auto-load
            default_csv = os.path.join(os.path.dirname(__file__), '..', 'ExperimentRecordings', 
                                      '2025-09-15_04-49-49_Recording1_0_IPZ7_rpgd-lite-jax_25Hz_vel_1.0_noise_c[0.0, 0.0]_mu_None_mu_c_None_.csv')
            if os.path.exists(default_csv):
                # Auto-load specific recording file
                try:
                    self.data = pd.read_csv(default_csv, comment='#')
                    self.csv_file_path = default_csv  # Store the file path
                    # Sync to common object
                    self.common.data = self.data
                    self.common.csv_file_path = default_csv
                    filename = os.path.basename(default_csv)
                    self.file_label.config(text=f"Loaded: {filename} (auto)")
                    
                    # Set up initial state
                    available_states = [col for col in self.state_columns if col in self.data.columns]
                    self.state_combo['values'] = available_states
                    if available_states:
                        self.state_var.set(available_states[1])  # Start with linear_vel_x
                    
                    # Update other data combo box with all CSV columns
                    all_columns = list(self.data.columns)
                    if hasattr(self, 'other_data_combo'):
                        self.other_data_combo['values'] = all_columns
                        self.other_data_var.set("")  # Reset to empty (default)
                    # Clear selected additional data list when auto-loading CSV
                    self.selected_other_data = []
                    if hasattr(self, 'other_data_listbox'):
                        self.update_other_data_listbox()
                        
                    # Set default model options
                    self.model_var.set(list(self.available_models.values())[0])
                    self.params_var.set(list(self.available_car_params.values())[0])
                    
                    # Initialize data range with default values
                    self.start_index = 0
                    self.end_index = min(500, len(self.data))  # Default range 0-500
                    self.start_index_var.set("0")
                    self.end_index_var.set(str(self.end_index))
                    
                    # Update comparison slider range
                    if hasattr(self, 'comparison_slider'):
                        self.update_comparison_slider_range()
                    
                    # Enable reload button
                    if hasattr(self, 'reload_button'):
                        self.reload_button.config(state='normal')
                    
                    # Save config after auto-loading
                    self.save_config()
                    
                    self.plot_state()
                except Exception as e:
                    print(f"Could not auto-load specific recording CSV: {e}")
                
        self.root.mainloop()


if __name__ == "__main__":
    app = StateComparisonVisualizer()
    app.run()