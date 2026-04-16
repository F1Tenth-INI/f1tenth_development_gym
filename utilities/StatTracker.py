import numpy as np
import csv
import os
import time
import matplotlib.pyplot as plt
import json
from datetime import datetime
from utilities.state_utilities import STATE_INDICES


class StatTracker:
    """
    Tracks training statistics and observations during SAC training.
    Uses a dictionary for efficient O(1) lookups and updates of transitions.
    
    Extracts raw state values from normalized observations:
    - obs[0]: linear_vel_x (scaled by 0.1)
    - obs[1]: linear_vel_y (scaled by 1.0)
    - obs[2]: angular_vel_z (scaled by 0.5)
    - obs[3]: steering_angle (scaled by 1/0.4)
    """
    
    def __init__(
        self,
        save_dir: str = "stat_logs",
        save_name: str = "stats_log.csv",
        max_buffer_size: int = 100000,
        extended_obs_action_save: bool = False,
        csv_float_decimals: int = 4,
    ):
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, save_name)
        self.transition_dict = {}  # {ID: transition_dict}
        self.ID_counter = 0

        self.buffer_position_to_id = {} #

        self.max_buffer_size = max_buffer_size
 
        os.makedirs(save_dir, exist_ok=True)
        
        # State normalization -> might be a more elegant way to save unnormalize
        self.state_norm_factors = np.array([0.1, 1.0, 0.5, 1/0.4], dtype=np.float32)

        self.d_e_norm_factors = np.array([0.5, 0.5], dtype=np.float32)  # reward, done

        self.training_length_list = []

        self.buffer_size_list = []

        self.total_sample_calls = 0
        
        # Track post-training operation timings
        self.csv_logging_time_list = []
        self.serialization_time_list = []
        self.broadcast_time_list = []

        self.save_full_obs_action_enabled = extended_obs_action_save
        self.csv_float_decimals = max(0, int(csv_float_decimals))

    def _format_float_for_csv(self, value) -> str:
        return f"{float(value):.{self.csv_float_decimals}f}"

    def _format_sequence_for_csv(self, sequence) -> str:
        return "[" + ",".join(self._format_sequence_item_for_csv(v) for v in sequence) + "]"

    def _format_sequence_item_for_csv(self, value) -> str:
        if value is None:
            return "None"
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        if isinstance(value, (float, np.floating)):
            return self._format_float_for_csv(value)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, np.ndarray):
            return self._format_sequence_for_csv(value.tolist())
        if isinstance(value, (list, tuple)):
            return self._format_sequence_for_csv(value)
        return str(value)

    def _serialize_value_for_csv(self, value):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (float, np.floating)):
            return self._format_float_for_csv(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, np.ndarray):
            return self._format_sequence_for_csv(value.tolist())
        if isinstance(value, (list, tuple)):
            return self._format_sequence_for_csv(value)
        return value
    
    # def unnormalize_obs(self, obs: np.ndarray) -> dict:    # -> i would maybe also like x and y pos to be saved
    #     if len(obs) < 4:
    #         return {}
        
    #     raw_state = obs[:4] / self.state_norm_factors

    #     raw_d_e = obs[160:162] / self.d_e_norm_factors  # reward, done
        
    #     return {
    #         'linear_vel_x': float(raw_state[0]),
    #         'linear_vel_y': float(raw_state[1]),
    #         'angular_vel_z': float(raw_state[2]),
    #         'steering_angle': float(raw_state[3]),
    #         'd_raw': float(raw_d_e[0]),
    #         'e_raw': float(raw_d_e[1]),
    #     }


    def register_transition(self, obs, action, buffer_position, reward, done, TD_error=None, info=None):
        """
        Register a transition for logging. Returns ID for later updates.
        
        Args:
            obs: Observation array (normalized)
            buffer_position: Position in replay buffer
            reward: Reward value (required)
            done: Whether episode is done (required)
            TD_error: Temporal difference error (optional)
            
        Returns:
            Transition ID for later lookup/updateF
        """
         
        transition_entry = {
            'id': self.ID_counter,
            'timestamp': datetime.now().time().isoformat(),
            'buffer_position': buffer_position,
            'sample_count': 0,  # Track how many times this transition was sampled
            'sample_calls_at_birth': self.total_sample_calls, 
            'sample_calls_at_death': None, 
            'reward': float(reward),
            'done': bool(done),
        }
        
        # Extract raw state values
        # raw_state = self.unnormalize_obs(obs)
        obs_dict = {
            'linear_vel_x': float(obs[0]),
            'linear_vel_y': float(obs[1]),
            'angular_vel_z': float(obs[2]),
            'steering_angle': float(obs[3]),
            'd_raw': float(obs[80]),
            'e_raw': float(obs[81]),
        }

        # Add raw state values
        transition_entry.update(obs_dict)

        weight_dict = {
            'state_weight': None,
            'reward_weight': None,
            'combined_weight': None,
        }

        transition_entry.update(weight_dict)

        pose_dict = {
            'pose_x': info.get('pose_x') if info else None,
            'pose_y': info.get('pose_y') if info else None,
        }

        transition_entry.update(pose_dict)

        TD_dict = {
            'TD_error_list': [],
            'TD_error_mean': None,
            'TD_error_max': None,
            'TD_error_min': None,
            'TD_error_latest': None,
            'TD_error_first': None,
        }

        transition_entry.update(TD_dict)
        
        # Add optional values
        if TD_error is not None:
            transition_entry['TD_error_list'].append(float(TD_error))

        if self.save_full_obs_action_enabled:
            # Store as arrays in memory (serialize only when saving CSV)
            transition_entry['obs'] = obs.copy() if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)
            transition_entry['action'] = action.copy() if isinstance(action, np.ndarray) else np.array(action, dtype=np.float32)
        
        self.transition_dict[self.ID_counter] = transition_entry

        old_id = self.buffer_position_to_id.get(buffer_position)
        if old_id is not None:
            self.transition_dict[old_id]['sample_calls_at_death'] = self.total_sample_calls

        self.buffer_position_to_id[buffer_position] = self.ID_counter

        self.ID_counter += 1
        
        return
    

    
    def update_transition(self, transition_id, TD_error=None, reward=None):
        if transition_id not in self.transition_dict:
            print(f"[StatTracker] Warning: Transition ID {transition_id} not found")
            return False
        
        t = self.transition_dict[transition_id]
        
        if TD_error is not None:
            t['TD_error_list'].append(float(TD_error))
        if reward is not None:
            t['reward'] = float(reward)

        return True

    def batch_update_TD_errors(self, buffer_indices: np.ndarray, TD_errors: np.ndarray):
        for i, buffer_idx in enumerate(buffer_indices):
            transition_id = self.buffer_position_to_id.get(buffer_idx)
            if transition_id is not None and transition_id in self.transition_dict:
                self.transition_dict[transition_id]['TD_error_list'].append(float(TD_errors[i]))

    def add_transition_static_weights(self, buffer_idx, state_weight=None, reward_weight=None):
        transition_id = self.buffer_position_to_id.get(buffer_idx)
        if transition_id is not None and transition_id in self.transition_dict:
                self.transition_dict[transition_id]['state_weight'] = float(state_weight)
                self.transition_dict[transition_id]['reward_weight'] = float(reward_weight)

    def update_training_length_list(self, length):
        self.training_length_list.append(float(length))
        self.buffer_size_list.append(min(self.max_buffer_size, self.ID_counter))
        return
    
    def update_csv_logging_time(self, duration):
        """Track time spent logging to CSV"""
        self.csv_logging_time_list.append(float(duration))
    
    def update_serialization_time(self, duration):
        """Track time spent serializing state dict"""
        self.serialization_time_list.append(float(duration))
    
    def update_broadcast_time(self, duration):
        """Track time spent broadcasting weights"""
        self.broadcast_time_list.append(float(duration))
    
    def batch_update_sample_count(self, buffer_indices):
        """
        Update sample count for sampled transitions.
        Can be disabled for performance during training via Settings.SAC_STAT_TRACKER_SAMPLE_COUNT_ENABLED.
        """
        self.total_sample_calls += 1
        
        # Optimized: Direct iteration without list comprehension
        for idx in buffer_indices:
            transition_id = self.buffer_position_to_id.get(idx)
            if transition_id is not None and transition_id in self.transition_dict:
                self.transition_dict[transition_id]['sample_count'] += 1
        return
    
    def get_transition(self, transition_id):
        return self.transition_dict.get(transition_id, None)
    
    def save_csv(self, append=True):
        if not self.transition_dict:
            return
        
        file_exists = os.path.isfile(self.file_path) and append
        
        first = next(iter(self.transition_dict.values()))
        ordered_keys = ['id'] + [k for k in first.keys() if k != 'id']
        
        try:
            mode = 'a' if file_exists else 'w'
            with open(self.file_path, mode=mode, newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=ordered_keys)
                
                if not file_exists:
                    writer.writeheader()
                
                for t in self.transition_dict.values():
                    # Serialize list/array fields and enforce float precision in CSV output.
                    row = {}
                    for k, v in t.items():
                        row[k] = self._serialize_value_for_csv(v)
                    writer.writerow(row)
            
            print(f"[StatTracker] Saved {len(self.transition_dict)} entries to {self.file_path}")
        except Exception as e:
            print(f"[StatTracker] Error saving to CSV: {e}")
    
    def clear(self):
        """Clear all logged transitions from memory."""
        self.transition_dict.clear()
    
    def get_logs(self) -> dict:
        """Get all logged transitions as a dictionary."""
        return self.transition_dict.copy()
    
    def get_logs_as_list(self) -> list:
        """Get all logged transitions as a list of dicts (for iteration)."""
        return list(self.transition_dict.values())
    
    def size(self) -> int:
        """Get the number of tracked transitions."""
        return len(self.transition_dict)
    
    def get_stats(self) -> dict:
        """
        Get summary statistics from logged transitions.
        
        Returns:
            Dictionary with mean/std/min/max for each logged field
        """
        if not self.transition_dict:
            return {}
        
        stats = {}
        
        # Get all numeric keys
        numeric_keys = ['linear_vel_x', 'linear_vel_y', 'angular_vel_z', 
                       'steering_angle', 'reward', 'sample_count']
        
        for key in numeric_keys:
            values = [t[key] for t in self.transition_dict.values() if key in t and t[key] is not None]
            if values:
                values = np.array(values)
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return stats
    
    def print_stats(self):
        """Print summary statistics to console."""
        stats = self.get_stats()
        if not stats:
            print("[StatTracker] No data to print")
            return
        
        print("\n[StatTracker] Summary Statistics:")
        print("=" * 80)
        for key, values in stats.items():
            print(f"{key:20s}: mean={values['mean']:10.4f}, std={values['std']:10.4f}, "
                  f"min={values['min']:10.4f}, max={values['max']:10.4f} (n={values['count']})")
        print("=" * 80 + "\n")

        if len(self.training_length_list) > 0:
            # Create 2x2 subplot layout
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Training Length + Buffer Size
            ax1 = axs[0, 0]
            ax1.set_xlabel('Training iteration')
            ax1.set_ylabel('Training Length (s)', color='tab:blue')
            ax1.plot(self.training_length_list, color='tab:blue', label='Training Length')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_title('Core Training Time')
            
            ax1_twin = ax1.twinx()
            ax1_twin.set_ylabel('Buffer Size', color='tab:orange')
            ax1_twin.plot(self.buffer_size_list, color='tab:orange', label='Buffer Size')
            ax1_twin.tick_params(axis='y', labelcolor='tab:orange')
            
            # Plot 2: CSV Logging Time
            if len(self.csv_logging_time_list) > 0:
                ax2 = axs[0, 1]
                ax2.plot(self.csv_logging_time_list, color='tab:green', label='CSV Logging Time')
                ax2.set_xlabel('Training iteration')
                ax2.set_ylabel('CSV Logging Time (s)', color='tab:green')
                ax2.set_title('CSV Logging Overhead')
                ax2.tick_params(axis='y', labelcolor='tab:green')
            
            # Plot 3: Serialization Time
            if len(self.serialization_time_list) > 0:
                ax3 = axs[1, 0]
                ax3.plot(self.serialization_time_list, color='tab:purple', label='Serialization Time')
                ax3.set_xlabel('Training iteration')
                ax3.set_ylabel('Serialization Time (s)', color='tab:purple')
                ax3.set_title('State Dict Serialization Time')
                ax3.tick_params(axis='y', labelcolor='tab:purple')
            
            # Plot 4: Broadcast Time
            if len(self.broadcast_time_list) > 0:
                ax4 = axs[1, 1]
                ax4.plot(self.broadcast_time_list, color='tab:red', label='Broadcast Time')
                ax4.set_xlabel('Training iteration')
                ax4.set_ylabel('Broadcast Time (s)', color='tab:red')
                ax4.set_title('Weight Broadcast Time')
                ax4.tick_params(axis='y', labelcolor='tab:red')
            
            plt.suptitle('Training Performance Metrics Over Time', fontsize=14, y=1.00)
            fig.tight_layout()
            
            plot_path = os.path.join(self.save_dir, 'training_performance_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[StatTracker] Saved training performance plot to {plot_path}")