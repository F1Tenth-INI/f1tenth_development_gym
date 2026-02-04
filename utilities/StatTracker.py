import numpy as np
import csv
import os
import time
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
    
    def __init__(self, save_dir: str = "stat_logs", save_name: str = "stats_log.csv"):
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, save_name)
        self.transition_dict = {}  # {ID: transition_dict}
        self.ID_counter = 0

        self.buffer_position_to_id = {} #
 
        os.makedirs(save_dir, exist_ok=True)
        
        # State normalization -> might be a more elegant way to save unnormalize
        self.state_norm_factors = np.array([0.1, 1.0, 0.5, 1/0.4], dtype=np.float32)

        self.d_e_norm_factors = np.array([0.5, 0.5], dtype=np.float32)  # reward, done
    
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


    def register_transition(self, obs, buffer_position, reward, done, TD_error=None):
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
            'd_raw': float(obs[160]),
            'e_raw': float(obs[161]),
        }

        # Add raw state values
        transition_entry.update(obs_dict)

        transition_entry.update({'TD_error_list': []})
        
        # Add optional values
        if TD_error is not None:
            transition_entry['TD_error_list'].append(float(TD_error))
        
        self.transition_dict[self.ID_counter] = transition_entry
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
    
    # def batch_update_transition(self, transition_id, TD_error=None, reward=None):
    #     if transition_id not in self.transition_dict:
    #         print(f"[StatTracker] Warning: Transition ID {transition_id} not found")
    #         return False
        
    #     t = self.transition_dict[transition_id]
        
    #     if TD_error is not None:
    #         t['TD_error_list'].append(float(TD_error))
    #     if reward is not None:
    #         t['reward'] = float(reward)

    #     return True
    
    def batch_update_TD_errors(self, buffer_indices: np.ndarray, TD_errors: np.ndarray):
        for i, buffer_idx in enumerate(buffer_indices):
            transition_id = self.buffer_position_to_id.get(buffer_idx)
            if transition_id is not None and transition_id in self.transition_dict:
                self.transition_dict[transition_id]['TD_error_list'].append(float(TD_errors[i]))
    
    
    def batch_update_sample_count(self, buffer_indices):
        # print('im updating')
        updated_ids = [self.buffer_position_to_id[idx] for idx in buffer_indices]

        for transition_id in updated_ids:
            t = self.transition_dict[transition_id]
            t['sample_count'] = t.get('sample_count', 0) + 1
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
                    # Convert numpy arrays to strings for CSV
                    row = {}
                    for k, v in t.items():
                        if isinstance(v, np.ndarray):
                            row[k] = np.array2string(v, separator=',', max_line_width=10000)
                        else:
                            row[k] = v
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