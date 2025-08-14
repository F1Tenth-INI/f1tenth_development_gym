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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple

# Add the parent directories to path to import simulation modules
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'sim/f110_sim/envs'))
sys.path.append(os.path.join(parent_dir, 'utilities'))


from sim.f110_sim.envs.dynamic_model_pacejka_jax import car_steps_sequential_jax
# Import residual dynamics dynamically when needed
from utilities.car_files.vehicle_parameters import VehicleParameters
from utilities.state_utilities import STATE_VARIABLES, STATE_INDICES



class StateComparisonVisualizer:
    """Main class for the state comparison visualization tool."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("State Comparison Visualizer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data = None
        self.time_column = 'time'
        self.state_columns = list(STATE_VARIABLES)
        self.control_columns = ['angular_control', 'translational_control']
        
        # Visualization parameters
        self.start_index = 0
        self.end_index = None
        self.comparison_start_index = 0
        self.comparison_data_dict = {}  # Store predictions for all start indices
        
        # Residual dynamics (loaded dynamically)
        self.residual_dynamics = None
        self.residual_functions_loaded = False
        
        # Available car models
        self.available_models = {
            'pacejka': 'Pure Pacejka Model',
            # 'pacejka_residual': 'Pacejka Model with Residuals',
        }
        
        # Available car parameter files
        self.available_car_params = {
            'mpc_car_parameters.yml': 'MPC Car Parameters',
            'gym_car_parameters.yml': 'Gym Car Parameters',
            'ini_car_parameters.yml': 'INI Car Parameters',
            'custom_car_parameters.yml': 'Custom Car Parameters'
        }
        
        # Use centralized state mapping from utilities
        self.state_indices = STATE_INDICES
        
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Command-s>', lambda event: self.save_plot())
        self.root.bind('<Control-s>', lambda event: self.save_plot())  # For non-Mac systems
        self.root.focus_set()  # Make sure the window can receive key events
        
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
        # File loading section
        file_frame = ttk.LabelFrame(parent, text="Data Loading")
        file_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Button(file_frame, text="Load CSV File", 
                  command=self.load_csv_file).pack(pady=2)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded", 
                                   wraplength=200)
        self.file_label.pack(pady=2)
        
        # Data range controls
        ttk.Label(file_frame, text="Start Index:").pack()
        self.start_index_var = tk.StringVar(value="0")
        ttk.Entry(file_frame, textvariable=self.start_index_var, width=10).pack()
        
        ttk.Label(file_frame, text="End Index:").pack()
        self.end_index_var = tk.StringVar(value="")
        ttk.Entry(file_frame, textvariable=self.end_index_var, width=10).pack()
        
        ttk.Button(file_frame, text="Update Range", 
                  command=self.update_data_range).pack(pady=2)
        
        # State selection
        state_frame = ttk.LabelFrame(parent, text="State Selection")
        state_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        ttk.Label(state_frame, text="Select State:").pack()
        self.state_var = tk.StringVar()
        self.state_combo = ttk.Combobox(state_frame, textvariable=self.state_var,
                                       values=self.state_columns, state='readonly')
        self.state_combo.pack(pady=2)
        self.state_combo.bind('<<ComboboxSelected>>', self.on_state_changed)
        
        self.show_controls = tk.BooleanVar(value=False)
        ttk.Checkbutton(state_frame, text="Show Control Plots",
                       variable=self.show_controls,
                       command=self.on_show_controls_toggled).pack(pady=2)
        
        # Comparison options
        comparison_frame = ttk.LabelFrame(parent, text="Model Comparison")
        comparison_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.enable_comparison = tk.BooleanVar(value=True)  # Default enabled
        ttk.Checkbutton(comparison_frame, text="Enable Comparison",
                       variable=self.enable_comparison,
                       command=self.on_comparison_toggled).pack()
        
        ttk.Label(comparison_frame, text="Car Model:").pack()
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(comparison_frame, textvariable=self.model_var,
                                       values=list(self.available_models.values()),
                                       state='readonly')
        self.model_combo.pack(pady=2)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_changed)
        
        ttk.Label(comparison_frame, text="Car Parameters:").pack()
        self.params_var = tk.StringVar()
        self.params_combo = ttk.Combobox(comparison_frame, textvariable=self.params_var,
                                        values=list(self.available_car_params.values()),
                                        state='readonly')
        self.params_combo.pack(pady=2)
        self.params_combo.bind('<<ComboboxSelected>>', self.on_params_changed)
        
        
        
        ttk.Label(comparison_frame, text="Horizon Steps:").pack()
        self.horizon_var = tk.StringVar(value="50")
        horizon_entry = ttk.Entry(comparison_frame, textvariable=self.horizon_var, width=10)
        horizon_entry.pack()
        horizon_entry.bind('<KeyRelease>', self.on_horizon_changed)
        
        ttk.Label(comparison_frame, text="Control Delay Steps:").pack()
        self.control_delay_var = tk.StringVar(value="0")
        control_delay_entry = ttk.Entry(comparison_frame, textvariable=self.control_delay_var, width=10)
        control_delay_entry.pack()
        control_delay_entry.bind('<KeyRelease>', self.on_control_delay_changed)
        
        # Add explanatory label for control delay
        delay_help_label = ttk.Label(comparison_frame, 
                                   text="(0 = no delay, 4 = use control\nfrom 4 timesteps ago)",
                                   font=('TkDefaultFont', 8), 
                                   foreground='gray')
        delay_help_label.pack(pady=(0, 5))
        
        ttk.Button(comparison_frame, text="Run Full Comparison",
                  command=self.run_full_comparison).pack(pady=5)
        
        ttk.Button(comparison_frame, text="Save Plot",
                  command=self.save_plot).pack(pady=2)
        
        self.show_all_comparisons = tk.BooleanVar()
        ttk.Checkbutton(comparison_frame, text="Show All Comparisons",
                       variable=self.show_all_comparisons,
                       command=self.on_show_all_comparisons_toggled).pack(pady=2)
        
    def setup_plot_area(self, parent):
        """Setup the plotting area."""
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.backends._backend_tk import NavigationToolbar2Tk
            from matplotlib.figure import Figure
            
            # Create slider frame
            slider_frame = ttk.Frame(parent)
            slider_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
            
            ttk.Label(slider_frame, text="Comparison Start Index:").pack(side=tk.LEFT)
            self.comparison_start_var = tk.IntVar(value=0)
            self.comparison_slider = tk.Scale(slider_frame, from_=0, to=0, 
                                            variable=self.comparison_start_var,
                                            orient=tk.HORIZONTAL, length=400,
                                            command=self.on_comparison_slider_changed)
            self.comparison_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            self.comparison_index_label = ttk.Label(slider_frame, text="Index: 0")
            self.comparison_index_label.pack(side=tk.RIGHT, padx=5)
            
            # Create figure with zoom/pan capabilities
            self.fig = Figure(figsize=(12, 10), dpi=100)
            
            # Initially create just the main plot (single subplot)
            self.ax = self.fig.add_subplot(111)  # Full-size single plot
            self.ax_controls = None  # Will be created when needed
            
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
                
                # Update file label
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"Loaded: {filename}")
                
                # Update state combo box with available columns
                available_states = [col for col in self.state_columns if col in self.data.columns]
                self.state_combo['values'] = available_states
                
                if available_states:
                    self.state_var.set(available_states[0])
                
                # Reset data range with default values
                self.start_index = 0
                self.end_index = min(500, len(self.data))  # Default range 0-500
                self.start_index_var.set("0")
                self.end_index_var.set(str(self.end_index))
                
                # Update comparison slider range
                if hasattr(self, 'comparison_slider'):
                    self.update_comparison_slider_range()
                    
                # Plot initial state
                self.plot_state()
                
                messagebox.showinfo("Success", 
                                  f"Loaded {len(self.data)} data points from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {str(e)}")
                
    def load_residual_dynamics(self):
        """Load residual dynamics module and create ResidualDynamics instance."""
        if self.residual_functions_loaded:
            return True
            
        try:
            # Import residual dynamics functions
            from sim.f110_sim.envs.dynamic_model_pacejka_jax_residual_simple import (
                ResidualDynamics,
                car_steps_sequential_jax_residual,
            )
            
            # Store the functions as instance attributes so we can use them later
            self.car_steps_sequential_jax_residual = car_steps_sequential_jax_residual
            
            # Create ResidualDynamics instance (this will load the model)
            print("Loading residual dynamics model...")
            self.residual_dynamics = ResidualDynamics()
            
            self.residual_functions_loaded = True
            print("✅ Residual dynamics loaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load residual dynamics: {e}")
            messagebox.showwarning("Warning", 
                                 f"Failed to load residual dynamics: {e}\n"
                                 "The 'Pacejka Model with Residuals' option will not work correctly.")
            return False
    
    def on_state_changed(self, event=None):
        """Handle state selection change."""
        self.plot_state()
        
    def on_show_controls_toggled(self):
        """Handle show controls checkbox toggle."""
        if self.fig is not None and self.canvas is not None:
            # Store current axis limits to preserve zoom/pan state
            xlim = self.ax.get_xlim() if self.ax is not None else None
            ylim = self.ax.get_ylim() if self.ax is not None else None
            
            # Clear the entire figure
            self.fig.clear()
            
            if self.show_controls.get():
                # Create subplot layout (main + control)
                self.ax = self.fig.add_subplot(211)  # Main plot (top)
                self.ax_controls = self.fig.add_subplot(212, sharex=self.ax)  # Control plot (bottom)
            else:
                # Create single plot layout (main only)
                self.ax = self.fig.add_subplot(111)  # Full-size single plot
                self.ax_controls = None
            
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
            
        # Replot the data with the new layout
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
        
        # If residual model is selected, load residual dynamics
        if selected_model_key == 'pacejka_residual' and not self.residual_functions_loaded:
            if not self.load_residual_dynamics():
                # If loading failed, switch back to regular model
                self.model_var.set(self.available_models['pacejka'])
                return
        
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
    
    def on_control_delay_changed(self, event=None):
        """Handle control delay value change."""
        try:
            delay_str = self.control_delay_var.get()
            if delay_str.strip():  # Only validate if not empty
                delay_val = int(delay_str)
                if delay_val < 0:
                    messagebox.showwarning("Warning", "Control delay should be non-negative (>= 0)")
                    self.control_delay_var.set("0")
                    return
                elif delay_val > 100:  # Reasonable upper limit
                    messagebox.showwarning("Warning", "Control delay seems very large. Are you sure?")
        except ValueError:
            if self.control_delay_var.get().strip():  # Only show error for non-empty invalid input
                messagebox.showerror("Error", "Control delay must be a non-negative integer")
                self.control_delay_var.set("0")
                return
        
        # Clear existing comparison data when control delay changes
        if hasattr(self, 'comparison_data_dict'):
            self.comparison_data_dict = {}
        # Update comparison slider range when control delay changes
        self.update_comparison_slider_range()
    
    
    def plot_state(self):
        """Plot the selected state variable."""
        if self.fig is None or self.ax is None or self.canvas is None or self.data is None:
            return
            
        selected_state = self.state_var.get()
        if not selected_state or selected_state not in self.data.columns:
            return
            
        self.ax.clear()
        
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
        
        self.ax.set_xlabel('Time (s)' if self.time_column in self.data.columns else 'Step')
        self.ax.set_ylabel(selected_state.replace('_', ' ').title())
        self.ax.set_title(f'State Comparison: {selected_state}')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Set axis limits to respect the current data range
        if self.time_column in self.data.columns:
            x_min = self.data[self.time_column].iloc[start_idx]
            x_max = self.data[self.time_column].iloc[end_idx - 1] if end_idx <= len(self.data) else self.data[self.time_column].iloc[-1]
            self.ax.set_xlim(x_min, x_max)
        else:
            self.ax.set_xlim(start_idx, end_idx - 1)
        
        # Plot controls if enabled
        if (hasattr(self, 'ax_controls') and self.ax_controls is not None and 
            self.show_controls.get()):
            self.plot_controls(start_idx, end_idx, time_data)
        
        self.canvas.draw()
        
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
            
        # Plot prediction as segments with gradient
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
            
            # Plot segment
            self.ax.plot(time_data[i:i+2], prediction_data[i:i+2], 
                       color=color, linewidth=2, alpha=alpha,
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
        
        if 'angular_control' in control_data:
            line1 = self.ax_controls.plot(time_data, control_data['angular_control'], 
                                        'r-', label='Steering Angle', linewidth=2)
            self.ax_controls.set_ylabel('Steering Angle (rad)', color='r')
            self.ax_controls.tick_params(axis='y', labelcolor='r')
            lines_legend.extend(line1)
            labels_legend.append('Steering Angle')
        
        # Create second y-axis for acceleration if available
        if 'translational_control' in control_data:
            ax_controls2 = self.ax_controls.twinx()
            line2 = ax_controls2.plot(time_data, control_data['translational_control'], 
                                    'b-', label='Acceleration', linewidth=2)
            ax_controls2.set_ylabel('Acceleration (m/s²)', color='b')
            ax_controls2.tick_params(axis='y', labelcolor='b')
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
        """Handle comparison slider change."""
        self.comparison_start_index = int(float(value))
        if hasattr(self, 'comparison_index_label'):
            self.comparison_index_label.config(text=f"Index: {self.comparison_start_index}")
        
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
                # Save the figure with high DPI for better quality
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight', 
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
            vehicle_params = VehicleParameters(param_file)
            return vehicle_params.to_np_array()
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
            
            # Load car parameters
            car_params = self.load_car_parameters(param_file)
            if car_params is None:
                return False
            
            # Prepare initial state and controls with delay
            # Note: Control delay is handled in extract_control_sequence_at_index
            # If delay=4, then at timestep t, we use control from timestep t-4
            initial_state = self.extract_initial_state_at_index(start_index)
            control_sequence = self.extract_control_sequence_at_index(start_index, horizon)
            
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
            return False
            
    def _run_model_prediction(self, model_name, initial_state, control_sequence, car_params, dt, horizon):
        """Run model prediction based on model name."""
        try:
            if model_name == 'pacejka':
                return car_steps_sequential_jax(
                    initial_state, control_sequence, car_params, dt, horizon, 
                    model_type='pacejka',
                    intermediate_steps=1
                )
            else:
                print(f"Unknown model: {model_name}")
                return None
        except NameError as e:
            # Handle case where dynamics functions aren't imported
            messagebox.showerror("Error", f"Dynamics models not available: {e}")
            return None
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return None
            
    def run_full_comparison(self):
        """Run model comparison for multiple start indices to enable sliding comparison."""
        horizon = self._validate_comparison_requirements()
        if not horizon:
            return
        
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
        """Extract control sequence from the data starting at a specific index with control delay."""
        if self.data is None:
            return jnp.zeros((horizon, 2), dtype=jnp.float32)
        
        # Get control delay from UI
        try:
            control_delay = int(self.control_delay_var.get()) if hasattr(self, 'control_delay_var') and self.control_delay_var.get().isdigit() else 0
        except:
            control_delay = 0
        
        # Apply control delay: control at timestep t is actually the control from timestep (t - control_delay)
        # For positive delay, we need to shift the control sequence backward in time
        control_start_index = start_index - control_delay
        
        # Ensure we don't go beyond data bounds
        control_end_index = min(control_start_index + horizon, len(self.data))
        
        # Control order: [desired_steering_angle, acceleration]
        control_sequence = np.zeros((horizon, 2), dtype=np.float32)
        
        # Extract control data with delay consideration
        if control_start_index >= 0 and control_start_index < len(self.data):
            # Normal case: control data is available
            actual_horizon = max(0, control_end_index - control_start_index)
            
            if actual_horizon > 0:
                # Use available control columns
                if 'angular_control' in self.data.columns:
                    control_data = self.data['angular_control'].iloc[control_start_index:control_end_index].values
                    control_sequence[:actual_horizon, 0] = control_data
                if 'translational_control' in self.data.columns:
                    control_data = self.data['translational_control'].iloc[control_start_index:control_end_index].values
                    control_sequence[:actual_horizon, 1] = control_data
        else:
            # Edge case: control delay pushes us before the start of data
            # For the initial timesteps where delayed control isn't available, use the first available control
            available_start = max(0, control_start_index)
            available_end = min(available_start + horizon, len(self.data))
            
            if available_end > available_start:
                # Fill with the first available control values
                if 'angular_control' in self.data.columns:
                    first_control = self.data['angular_control'].iloc[0]
                    offset = max(0, -control_start_index)
                    actual_length = min(horizon - offset, available_end - available_start)
                    control_sequence[offset:offset + actual_length, 0] = self.data['angular_control'].iloc[available_start:available_end].values
                    # Fill the initial delayed timesteps with the first control value
                    if offset > 0:
                        control_sequence[:offset, 0] = first_control
                        
                if 'translational_control' in self.data.columns:
                    first_control = self.data['translational_control'].iloc[0]
                    offset = max(0, -control_start_index)
                    actual_length = min(horizon - offset, available_end - available_start)
                    control_sequence[offset:offset + actual_length, 1] = self.data['translational_control'].iloc[available_start:available_end].values
                    # Fill the initial delayed timesteps with the first control value
                    if offset > 0:
                        control_sequence[:offset, 1] = first_control
            
        return jnp.array(control_sequence)
        
    def get_timestep(self):
        """Get timestep from data or use default."""
        if self.data is not None and self.time_column in self.data.columns and len(self.data) > 1:
            return float(self.data[self.time_column].iloc[1] - self.data[self.time_column].iloc[0])
        return 0.02  # Default 50Hz
        
    def convert_predictions_to_dict(self, predicted_states):
        """Convert JAX predictions to dictionary format matching CSV columns."""
        comparison_data = {}
        # Use centralized state mapping from utilities
        for col_name, idx in self.state_indices.items():
            if col_name in self.state_columns:
                comparison_data[col_name] = predicted_states[:, idx]
                
        return comparison_data
        
    def run(self):
        """Start the application."""
        # Set default CSV file if it exists
        default_csv = os.path.join(os.path.dirname(__file__), '..', 'ExperimentRecordings', 'Test.csv')
        if os.path.exists(default_csv):
            # Auto-load default file
            try:
                self.data = pd.read_csv(default_csv, comment='#')
                self.file_label.config(text="Loaded: Test.csv (default)")
                
                # Set up initial state
                available_states = [col for col in self.state_columns if col in self.data.columns]
                self.state_combo['values'] = available_states
                if available_states:
                    self.state_var.set(available_states[1])  # Start with linear_vel_x
                    
                # Set default model options
                self.model_var.set(list(self.available_models.values())[0])
                self.params_var.set(list(self.available_car_params.values())[0])
                
                # Initialize data range with default values
                self.start_index = 0
                self.end_index = min(500, len(self.data))  # Default range 0-500
                
                # Update comparison slider range
                if hasattr(self, 'comparison_slider'):
                    self.update_comparison_slider_range()
                
                self.plot_state()
            except Exception as e:
                print(f"Could not auto-load default CSV: {e}")
                
        self.root.mainloop()


if __name__ == "__main__":
    app = StateComparisonVisualizer()
    app.run()
