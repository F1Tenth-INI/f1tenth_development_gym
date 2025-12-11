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
matplotlib.rcParams['font.size'] = 24  # Base plot font size
matplotlib.rcParams['axes.titlesize'] = 28  # Plot title size
matplotlib.rcParams['axes.labelsize'] = 24  # Plot label size
matplotlib.rcParams['xtick.labelsize'] = 20  # Tick label size
matplotlib.rcParams['ytick.labelsize'] = 20  # Tick label size
matplotlib.rcParams['legend.fontsize'] = 22  # Legend size
matplotlib.rcParams['figure.titlesize'] = 30  # Figure title size

# Add the parent directories to path to import simulation modules
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'sim/f110_sim/envs'))
sys.path.append(os.path.join(parent_dir, 'utilities'))


from sim.f110_sim.envs.car_model_jax import car_steps_sequential_jax
# Import residual dynamics dynamically when needed
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import STATE_VARIABLES, STATE_INDICES

# Control column names - change these if CSV column names change
STEERING_CONTROL_COLUMN = 'angular_control_executed'
ACCELERATION_CONTROL_COLUMN = 'translational_control_executed'


class StateComparisonVisualizer:
    """Main class for the state comparison visualization tool."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("State Comparison Visualizer")
        
        # Detect and handle high DPI displays
        self.setup_dpi_scaling()
        
        # Setup larger UI fonts for control panel
        self.setup_ui_fonts()
        
        self.root.geometry("1800x1100")  # Extra large window for much larger fonts
        
        # Data storage
        self.data = None
        self.csv_file_path = None  # Store the path to the loaded CSV file
        self.time_column = 'time'
        self.state_columns = list(STATE_VARIABLES)
        self.control_columns = [STEERING_CONTROL_COLUMN, ACCELERATION_CONTROL_COLUMN]
        
        # Config file path
        self.config_file_path = os.path.join(os.path.dirname(__file__), 'visualization_config.json')
        
        # Visualization parameters
        self.start_index = 0
        self.end_index = None
        self.comparison_start_index = 0
        self.comparison_data_dict = {}  # Store predictions for all start indices
        
        # Slider optimization parameters
        self.slider_update_timer = None
        self.slider_update_delay = 300  # milliseconds to wait after slider stops moving
        
        # Residual dynamics (loaded dynamically)
        self.residual_dynamics = None
        self.residual_functions_loaded = False
        
        # Available car models
        self.available_models = {
            'pacejka': 'Pure Pacejka Model',
            'pacejka_custom': 'Pacejka Model with Customization',
            'direct': 'Direct Dynamics Neural Network',
            'residual': 'Residual Dynamics Model',
        }
        
        # Residual dynamics model (lazy loaded)
        self.residual_model = None
        self._residual_model_loaded = False

        # Wider selectors so long option text remains visible in the dropdown
        self.selector_width_chars = 32
        
        # Available car parameter files - automatically detect all YAML files
        self.available_car_params = {}
        car_files_dir = os.path.join(parent_dir, 'utilities', 'car_files')
        if os.path.exists(car_files_dir):
            for filename in os.listdir(car_files_dir):
                if filename.endswith('.yml') or filename.endswith('.yaml'):
                    self.available_car_params[filename] = filename
        
        # Use centralized state mapping from utilities
        self.state_indices = STATE_INDICES
        
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
        
        self.setup_ui()
        
        # Load config file if it exists
        self.load_config()
        
        # Bind keyboard shortcuts
        self.root.bind('<Command-s>', lambda event: self.save_plot())
        self.root.bind('<Control-s>', lambda event: self.save_plot())  # For non-Mac systems
        self.root.focus_set()  # Make sure the window can receive key events
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def setup_dpi_scaling(self):
        """Setup DPI scaling for high resolution displays."""
        try:
            # Get screen DPI
            dpi = self.root.winfo_fpixels('1i')
            
            # Calculate scaling factor (assuming 96 DPI as baseline)
            scale_factor = max(1.0, dpi / 96.0)
            
            # Apply scaling for plots on high-DPI displays
            if scale_factor > 1.2:  # Only scale for high DPI displays
                # Moderate scaling for plot readability
                plot_scale = max(1.3, scale_factor * 0.85)  # Moderate scaling for plots
                
                matplotlib.rcParams['figure.dpi'] = int(100 * plot_scale)
                matplotlib.rcParams['font.size'] = int(24 * plot_scale)
                matplotlib.rcParams['axes.titlesize'] = int(28 * plot_scale)
                matplotlib.rcParams['axes.labelsize'] = int(24 * plot_scale)
                matplotlib.rcParams['xtick.labelsize'] = int(20 * plot_scale)
                matplotlib.rcParams['ytick.labelsize'] = int(20 * plot_scale)
                matplotlib.rcParams['legend.fontsize'] = int(22 * plot_scale)
                matplotlib.rcParams['figure.titlesize'] = int(30 * plot_scale)
                
                print(f"High-DPI plot scaling applied (DPI: {dpi:.1f}). Plot scale: {plot_scale:.1f}x")
            
            # Set tkinter scaling for better text rendering
            try:
                self.root.tk.call('tk', 'scaling', max(1.2, scale_factor))
            except:
                pass  # Some systems don't support tk scaling
                
        except Exception as e:
            print(f"Could not detect DPI scaling: {e}")
            # Use larger default fonts for plots
            matplotlib.rcParams['font.size'] = 24
            matplotlib.rcParams['axes.titlesize'] = 28
            matplotlib.rcParams['axes.labelsize'] = 24
            matplotlib.rcParams['xtick.labelsize'] = 20
            matplotlib.rcParams['ytick.labelsize'] = 20
            matplotlib.rcParams['legend.fontsize'] = 22
            matplotlib.rcParams['figure.titlesize'] = 30
    
    def setup_ui_fonts(self):
        """Setup much larger fonts for the UI control panel."""
        try:
            import tkinter.font as tkFont
            
            # Create large 24pt fonts for all UI elements
            self.large_font = tkFont.Font(family="Arial", size=24, weight="normal")
            self.large_bold_font = tkFont.Font(family="Arial", size=24, weight="bold")
            self.button_font = tkFont.Font(family="Arial", size=24, weight="normal")
            self.metrics_font = tkFont.Font(family="Arial", size=24, weight="bold")
            
            # Configure ttk styles with much larger fonts
            style = ttk.Style()
            
            # Configure all styles with larger fonts and padding
            style.configure('Large.TLabel', font=self.large_font)
            style.configure('Bold.TLabel', font=self.large_bold_font)
            style.configure('Large.TButton', font=self.button_font, padding=(10, 10))
            style.configure('Large.TEntry', font=self.large_font, padding=10)
            style.configure('Large.TCombobox', font=self.large_font, padding=10)
            style.configure('Large.TCheckbutton', font=self.large_font)
            
            # Set default fonts and padding for all ttk widgets
            style.configure('TLabel', font=self.large_font)
            style.configure('TButton', font=self.button_font, padding=(10, 10))
            style.configure('TEntry', font=self.large_font, padding=10)
            style.configure('TCombobox', font=self.large_font, padding=10)
            style.configure('TCheckbutton', font=self.large_font)
            style.configure('TLabelFrame', font=self.large_bold_font)
            
            # Configure combobox dropdown list to have larger font
            self.root.option_add('*TCombobox*Listbox.font', self.large_font)
            # Also configure the dropdown height
            style.map('TCombobox', fieldbackground=[('readonly', 'white')])
            style.map('Large.TCombobox', fieldbackground=[('readonly', 'white')])
            
            print("Large UI fonts configured (24pt for all elements)")
            
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
            if csv_path and os.path.exists(csv_path):
                self.data = pd.read_csv(csv_path, comment='#')
                self.csv_file_path = csv_path
                self.file_label.config(text=f"Loaded: {os.path.basename(csv_path)}")
                available_states = [col for col in self.state_columns if col in self.data.columns]
                self.state_combo['values'] = available_states
                if available_states:
                    self.state_var.set(available_states[0])
                
                # Update other data combo box with all CSV columns
                all_columns = list(self.data.columns)
                if hasattr(self, 'other_data_combo'):
                    self.other_data_combo['values'] = all_columns
                    self.other_data_var.set("")  # Reset to empty (default)
                
                # Enable reload button
                if hasattr(self, 'reload_button'):
                    self.reload_button.config(state='normal')
            
            # Load settings
            self.start_index = config.get('start_index', 0)
            self.end_index = config.get('end_index', None)
            self.start_index_var.set(str(self.start_index))
            self.end_index_var.set(str(self.end_index) if self.end_index is not None else "")
            self.horizon_var.set(str(config.get('horizon_steps', 50)))
            self.steering_delay_var.set(str(config.get('steering_delay_steps', 2)))
            self.acceleration_delay_var.set(str(config.get('acceleration_delay_steps', 2)))
            
            if self.data is not None and hasattr(self, 'comparison_slider'):
                self.update_comparison_slider_range()
                self.plot_state()
        except Exception as e:
            print(f"Could not load config: {e}")
    
    def save_config(self):
        """Save current configuration to JSON file."""
        try:
            def safe_int(var, default):
                val = var.get().strip()
                return int(val) if val and val.isdigit() else default
            
            with open(self.config_file_path, 'w') as f:
                json.dump({
                    'csv_file_path': self.csv_file_path or '',
                    'start_index': self.start_index,
                    'end_index': self.end_index,
                    'horizon_steps': safe_int(self.horizon_var, 50),
                    'steering_delay_steps': safe_int(self.steering_delay_var, 2),
                    'acceleration_delay_steps': safe_int(self.acceleration_delay_var, 2),
                }, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")
        
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
        
        # Font size control (for plot fonts)
        ttk.Label(metrics_frame, text="Plot Font Size:", style='Bold.TLabel').pack(anchor='w')
        self.font_size_var = tk.StringVar(value="12")  # Default to normal size
        font_size_combo = ttk.Combobox(metrics_frame, textvariable=self.font_size_var,
                                      values=["8", "10", "12", "14", "16", "18", "20"], 
                                      state='readonly', width=self.selector_width_chars,
                                      style='Large.TCombobox')
        font_size_combo.pack(anchor='w', pady=2)
        font_size_combo.bind('<<ComboboxSelected>>', self.on_font_size_changed)
    
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
            self.comparison_slider = tk.Scale(slider_frame, from_=0, to=0, 
                                            variable=self.comparison_start_var,
                                            orient=tk.HORIZONTAL, length=400,
                                            font=self.large_font if hasattr(self, 'large_font') and self.large_font else ('Arial', 24),
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
                # Read CSV, skipping comment lines
                self.data = pd.read_csv(file_path, comment='#')
                self.csv_file_path = file_path  # Store the file path
                
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
                
                # Save config after loading CSV
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
    
    def on_sync_scales_toggled(self):
        """Handle sync scales checkbox toggle."""
        self.plot_state()
        self.plot_state()
        
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
    
    def on_show_delta_state_toggled(self):
        """Handle show delta state checkbox toggle."""
        self._update_plot_layout()
        self.plot_state()
        
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
        
    def on_show_metrics_toggled(self):
        """Handle show metrics checkbox toggle."""
        self.update_metrics_display()
        
    def on_font_size_changed(self, event=None):
        """Handle font size change."""
        try:
            font_size = int(self.font_size_var.get())
            
            # Update matplotlib font sizes with normal proportions
            matplotlib.rcParams['font.size'] = font_size
            matplotlib.rcParams['axes.titlesize'] = font_size + 2  # Normal title
            matplotlib.rcParams['axes.labelsize'] = font_size  # Same as base
            matplotlib.rcParams['xtick.labelsize'] = font_size - 2  # Smaller ticks
            matplotlib.rcParams['ytick.labelsize'] = font_size - 2  # Smaller ticks
            matplotlib.rcParams['legend.fontsize'] = font_size - 1  # Slightly smaller legend
            matplotlib.rcParams['figure.titlesize'] = font_size + 4  # Normal figure title
            
            # Update existing plot
            if self.fig is not None:
                # Redraw with new font sizes
                self.canvas.draw()
                
        except ValueError:
            # Invalid font size, revert to default
            self.font_size_var.set("12")
        
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
        
        # Calculate metrics for the entire data range (from start_index to end_index)
        metrics = self.calculate_metrics_for_entire_range(selected_state)
        
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
        selected_model_key = self.get_model_key(self.model_var.get())
        
    
        if self.enable_comparison.get():
            self.plot_state()
            
    def on_params_changed(self, event=None):
        """Handle car parameters selection change."""
        if self.enable_comparison.get():
            return
            # self.plot_state()
            
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
            
        try:
            from matplotlib.colors import LinearSegmentedColormap
            
            # Define color maps for different trajectories
            colors = ["#5200F5", "#FF00BF", "#FF0000"]  # Orange -> Dark orange -> Brown
            alpha_base = 0.4
            
            # Create custom colormap
            cmap = LinearSegmentedColormap.from_list("gradient", colors, N=100)
            
        except ImportError:
            # Fallback to simple gradient if matplotlib doesn't support advanced colormaps
            if trajectory_index == 0:
                start_color = np.array([1.0, 0.4, 0.4])  # Light red
                end_color = np.array([0.5, 0.0, 0.0])    # Dark red
                alpha_base = 0.8
            else:
                start_color = np.array([1.0, 0.6, 0.0])  # Orange
                end_color = np.array([0.5, 0.3, 0.1])    # Brown
                alpha_base = 0.4
            cmap = None
            
        # Convert time_data to numpy array if it's a pandas Series
        if hasattr(time_data, 'iloc'):
            time_data = time_data.values
        elif hasattr(time_data, 'to_numpy'):
            time_data = time_data.to_numpy()
        time_data = np.asarray(time_data)
        
        # Plot prediction as segments with gradient
        if len(prediction_data) == 1:
            # Special case: horizon=1, plot single point
            if cmap is not None:
                color = cmap(0.0)
            else:
                color = start_color
            alpha = alpha_base
            self.ax.plot(time_data[0], prediction_data[0], 
                       'o', color=color, markersize=3, alpha=alpha,
                       label=label)
        else:
            # Normal case: plot segments with gradient
            for i in range(len(prediction_data) - 1):
                # Calculate color based on position in horizon (0 to 1)
                color_position = i / max(1, len(prediction_data) - 1)
                
                if cmap is not None:
                    color = cmap(color_position)
                else:
                    # Simple interpolation fallback
                    color = start_color * (1 - color_position) + end_color * color_position
                    
                # Calculate alpha that decreases over horizon
                alpha = alpha_base * (1.0 - 0.3 * color_position)  # Fade from full alpha to 70% alpha
                
                # Plot segment as dots
                self.ax.plot(time_data[i:i+2], prediction_data[i:i+2], 
                           'o', color=color, markersize=3, alpha=alpha,
                           label=label if i == 0 else None)
    
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
        elif self.enable_comparison.get():
            # We have the data, just replot
            self.plot_state()
            
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
        for filename, name in self.available_car_params.items():
            if name == display_name:
                return filename
        return list(self.available_car_params.keys())[0]  # Default
        
    def get_model_key(self, display_name):
        """Get the model key from display name."""
        for key, name in self.available_models.items():
            if name == display_name:
                return key
        return list(self.available_models.keys())[0]  # Default
        
    def load_car_parameters(self, param_file):
        """Load car parameters from YAML file."""
        try:
            # Force fresh load from file
            vehicle_params = VehicleParameters(param_file)
            params_array = vehicle_params.to_np_array()
            print(f"Reloaded car parameters from {param_file} - array length: {len(params_array)}")
            return params_array
        except Exception as e:
            messagebox.showerror("Error", f"Error loading car parameters: {e}")
            return None
            
    def _validate_comparison_requirements(self):
        """Validate that comparison can be run."""
        if self.data is None:
            messagebox.showerror("Error", "No data loaded. Please load a CSV file first.")
            return False
            
        try:
            horizon = int(self.horizon_var.get())
            if horizon <= 0:
                raise ValueError("Horizon must be positive")
        except ValueError:
            messagebox.showerror("Error", "Invalid horizon value. Please enter a positive integer.")
            return False
            
        try:
            control_delay = int(self.control_delay_var.get()) if hasattr(self, 'control_delay_var') and self.control_delay_var.get() else 0
            if control_delay < 0:
                messagebox.showerror("Error", "Control delay must be non-negative.")
                return False
        except ValueError:
            messagebox.showerror("Error", "Invalid control delay value. Please enter a non-negative integer.")
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
            
            # Force reload car parameters fresh from file (no caching)
            car_params = self.load_car_parameters(param_file)
            if car_params is None:
                return False
            
            # Prepare initial state and controls with delay
            # Note: Control delay is handled in extract_control_sequence_at_index
            # If delay=4, then at timestep t, we use control from timestep t-4
            initial_state = self.extract_initial_state_at_index(start_index)
            control_sequence = self.extract_control_sequence_at_index(start_index, horizon)
            
            # Store start_index for residual model to extract history
            self._current_start_index = start_index
            
            # Get timestep from data
            dt = self.get_timestep()
            
            # Run model prediction based on selected model
            predicted_states = self._run_model_prediction(model_name, initial_state, control_sequence, car_params, dt, horizon)
            if predicted_states is None:
                return False
            
            # Store comparison data for this start index
            self.comparison_data_dict[start_index] = self.convert_predictions_to_dict(predicted_states)
            return True
            
        except Exception as e:
            print(f"Single comparison failed for index {start_index}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _run_model_prediction(self, model_name, initial_state, control_sequence, car_params, dt, horizon):
        """Run model prediction based on model name."""
        try:
            if model_name == 'pacejka':
                return car_steps_sequential_jax(
                    initial_state, control_sequence, car_params, dt, horizon, 
                    model_type='pacejka',
                    intermediate_steps=4
                )
            elif model_name == 'pacejka_custom':
                return car_steps_sequential_jax(
                    initial_state, control_sequence, car_params, dt, horizon,
                    model_type='pacejka_custom',
                    intermediate_steps=4
                )
            elif model_name == 'residual':
                # Lazy load residual model
                if not self._residual_model_loaded:
                    try:
                        sys.path.append(os.path.join(parent_dir, 'TrainingLite', 'dynamic_residual_jax'))
                        from dynamics_model_residual import DynamicsModelResidual
                        self.residual_model = DynamicsModelResidual()
                        self._residual_model_loaded = True
                    except Exception as e:
                        print(f"Failed to load residual model: {e}")
                        return None
                
                # Set history if available
                if hasattr(self, '_current_start_index') and self.data is not None and self._current_start_index >= 10:
                    start_idx = self._current_start_index
                    state_history = np.array([self.extract_initial_state_at_index(start_idx - 10 + i) for i in range(10)])
                    control_history = np.array([self.extract_control_sequence_at_index(start_idx - 10 + i, 1)[0] for i in range(10)])
                    self.residual_model.set_history(state_history, control_history)
                
                # Autoregressive prediction
                predicted_states = []
                current_state = np.array(initial_state)
                for i in range(horizon):
                    control = control_sequence[i] if control_sequence.ndim == 2 else control_sequence
                    current_state = self.residual_model.predict(current_state, control)
                    predicted_states.append(np.array(current_state))
                
                return np.array(predicted_states)
            else:
                print(f"Unknown model: {model_name}")
                return None
        except NameError as e:
            # Handle case where dynamics functions aren't imported
            messagebox.showerror("Error", f"Dynamics models not available: {e}")
            return None
        except Exception as e:
            print(f"Model prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def run_full_comparison(self):
        """Run model comparison for multiple start indices to enable sliding comparison."""
        import importlib
        import sys
        # Force reload car_model_jax and dynamic_model_pacejka_jax using importlib.reload
        if 'sim.f110_sim.envs.dynamic_model_pacejka_jax' in sys.modules:
            importlib.reload(sys.modules['sim.f110_sim.envs.dynamic_model_pacejka_jax'])
        else:
            import sim.f110_sim.envs.dynamic_model_pacejka_jax
        if 'sim.f110_sim.envs.car_model_jax' in sys.modules:
            importlib.reload(sys.modules['sim.f110_sim.envs.car_model_jax'])
        else:
            import sim.f110_sim.envs.car_model_jax
        global car_steps_sequential_jax
        car_steps_sequential_jax = sys.modules['sim.f110_sim.envs.car_model_jax'].car_steps_sequential_jax

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
        if self.data is None or index >= len(self.data):
            return jnp.zeros(10, dtype=jnp.float32)
            
        # Use centralized state mapping from utilities
        initial_state = np.zeros(10, dtype=np.float32)
        for col_name, idx in self.state_indices.items():
            if col_name in self.data.columns:
                initial_state[idx] = self.data[col_name].iloc[index]
                
        return jnp.array(initial_state)
        
    def extract_control_sequence(self, horizon):
        """Extract control sequence from the data."""
        return self.extract_control_sequence_at_index(0, horizon)
        
    def extract_control_sequence_at_index(self, start_index, horizon):
        """Extract control sequence from the data starting at a specific index with separate control delays."""
        if self.data is None:
            return jnp.zeros((horizon, 2), dtype=jnp.float32)
        
        # Get separate control delays from UI
        try:
            steering_delay = int(self.steering_delay_var.get()) if hasattr(self, 'steering_delay_var') and self.steering_delay_var.get().isdigit() else 0
        except:
            steering_delay = 0
        
        try:
            acceleration_delay = int(self.acceleration_delay_var.get()) if hasattr(self, 'acceleration_delay_var') and self.acceleration_delay_var.get().isdigit() else 0
        except:
            acceleration_delay = 0
        
        # Control order: [desired_steering_angle, acceleration]
        control_sequence = np.zeros((horizon, 2), dtype=np.float32)
        
        # Handle steering control with its own delay
        steering_start_index = start_index - steering_delay
        steering_end_index = min(steering_start_index + horizon, len(self.data))
        
        if steering_start_index >= 0 and steering_start_index < len(self.data):
            # Normal case: steering control data is available
            actual_horizon = max(0, steering_end_index - steering_start_index)
            if actual_horizon > 0:
                # Use the defined steering control column
                if STEERING_CONTROL_COLUMN in self.data.columns:
                    control_data = self.data[STEERING_CONTROL_COLUMN].iloc[steering_start_index:steering_end_index].values
                    control_sequence[:actual_horizon, 0] = control_data
        else:
            # Edge case: steering delay pushes us before the start of data
            available_start = max(0, steering_start_index)
            available_end = min(available_start + horizon, len(self.data))
            
            if available_end > available_start:
                # Use the defined steering control column
                if STEERING_CONTROL_COLUMN in self.data.columns:
                    first_control = self.data[STEERING_CONTROL_COLUMN].iloc[0]
                    offset = max(0, -steering_start_index)
                    actual_length = min(horizon - offset, available_end - available_start)
                    if actual_length > 0:
                        control_sequence[offset:offset + actual_length, 0] = self.data[STEERING_CONTROL_COLUMN].iloc[available_start:available_start + actual_length].values
                    # Fill the initial delayed timesteps with the first control value
                    if offset > 0:
                        control_sequence[:offset, 0] = first_control
        
        # Handle acceleration control with its own delay
        acceleration_start_index = start_index - acceleration_delay
        acceleration_end_index = min(acceleration_start_index + horizon, len(self.data))
        
        if acceleration_start_index >= 0 and acceleration_start_index < len(self.data):
            # Normal case: acceleration control data is available
            actual_horizon = max(0, acceleration_end_index - acceleration_start_index)
            if actual_horizon > 0:
                # Use the defined acceleration control column
                if ACCELERATION_CONTROL_COLUMN in self.data.columns:
                    control_data = self.data[ACCELERATION_CONTROL_COLUMN].iloc[acceleration_start_index:acceleration_end_index].values
                    control_sequence[:actual_horizon, 1] = control_data
        else:
            # Edge case: acceleration delay pushes us before the start of data
            available_start = max(0, acceleration_start_index)
            available_end = min(available_start + horizon, len(self.data))
            
            if available_end > available_start:
                # Use the defined acceleration control column
                if ACCELERATION_CONTROL_COLUMN in self.data.columns:
                    first_control = self.data[ACCELERATION_CONTROL_COLUMN].iloc[0]
                    offset = max(0, -acceleration_start_index)
                    actual_length = min(horizon - offset, available_end - available_start)
                    if actual_length > 0:
                        control_sequence[offset:offset + actual_length, 1] = self.data[ACCELERATION_CONTROL_COLUMN].iloc[available_start:available_start + actual_length].values
                    # Fill the initial delayed timesteps with the first control value
                    if offset > 0:
                        control_sequence[:offset, 1] = first_control
            
        return jnp.array(control_sequence)
        
    def get_timestep(self):
        """Get timestep from data or use default."""
        if self.data is not None and self.time_column in self.data.columns and len(self.data) > 1:
            return float(self.data[self.time_column].iloc[1] - self.data[self.time_column].iloc[0])
        return 0.04  # Default 50Hz
        
    def convert_predictions_to_dict(self, predicted_states):
        """Convert JAX predictions to dictionary format matching CSV columns."""
        comparison_data = {}
        # Use centralized state mapping from utilities
        for col_name, idx in self.state_indices.items():
            if col_name in self.state_columns:
                comparison_data[col_name] = predicted_states[:, idx]
                
        return comparison_data
    
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
        mean_error = np.mean(error)
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
    
    def calculate_metrics_for_entire_range(self, state_name):
        """Calculate error metrics for the entire data range using model predictions."""
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
            mean_error = np.mean(error)
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
