"""
Sample trajectories based on error scores (DAGGER Step 2).

This script samples trajectories from back-to-front data, prioritizing
trajectories with higher error scores (softmax sampling).

Input: CSV files with 'trajectory_error_score' column (from Step 1)
Output: Sampled CSV files with re-indexed trajectories

Usage:
    python SampleTrajectories_ByErrorScore.py
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input folder (output from PreprocessData_BackToFront_Trajectories.py)
INPUT_FOLDER = './SI_Toolkit_ASF/Experiments/Experiments_05_12_2025_BackToFront/Recordings/Train/'

# Output folder
OUTPUT_FOLDER = './SI_Toolkit_ASF/Experiments/Experiments_05_12_2025_DAGGER/Recordings/Train/'

# Sampling parameters
SAMPLING_TEMPERATURE = 1.0  # Lower = more focus on high-error trajectories (0.5 = aggressive, 2.0 = mild)
SAMPLE_FRACTION = 1.0       # 1.0 = same number of trajectories as original
MIN_SAMPLES_PER_FILE = 100  # Minimum number of trajectories to sample per file
SAMPLE_WITH_REPLACEMENT = True  # Allow sampling same trajectory multiple times
RANDOM_SEED = 42            # For reproducibility

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def sample_trajectories_by_error_score(
    input_folder,
    output_folder,
    temperature=1.0,
    sample_fraction=1.0,
    min_samples=100,
    with_replacement=True,
    random_seed=42
):
    """
    Sample trajectories based on error scores using softmax weighting.
    
    Trajectories with higher error scores are sampled more frequently,
    helping the network learn from its mistakes (DAGGER-style).
    
    Args:
        input_folder: Folder with back-to-front CSVs containing 'trajectory_error_score'
        output_folder: Where to save sampled data
        temperature: Softmax temperature (lower = more focus on high-error)
        sample_fraction: Fraction of original trajectories to sample
        min_samples: Minimum trajectories per file
        with_replacement: Allow sampling same trajectory multiple times
        random_seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("SAMPLE TRAJECTORIES BY ERROR SCORE")
    print("=" * 80)
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Temperature: {temperature}")
    print(f"Sample fraction: {sample_fraction}")
    print(f"Min samples per file: {min_samples}")
    print(f"With replacement: {with_replacement}")
    print("=" * 80)
    
    rng = np.random.RandomState(random_seed)
    
    # Find all CSV files
    input_files = glob.glob(os.path.join(input_folder, '**/*.csv'), recursive=True)
    
    if not input_files:
        print(f"\nERROR: No CSV files found in {input_folder}")
        print("Make sure to run PreprocessData_BackToFront_Trajectories.py first!")
        return False
    
    print(f"\nProcessing {len(input_files)} files...")
    
    all_stats = []
    
    for input_path in tqdm(input_files, desc="Sampling trajectories"):
        # Determine output path
        rel_path = os.path.relpath(input_path, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        
        # Load data
        df = pd.read_csv(input_path, comment='#')
        
        # Check for error score column
        if 'trajectory_error_score' not in df.columns:
            print(f"\n  Warning: No 'trajectory_error_score' in {rel_path}")
            print(f"           Copying file as-is (uniform sampling)")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            continue
        
        # Get unique trajectory scores (one score per trajectory)
        traj_scores = df.groupby('experiment_index')['trajectory_error_score'].first()
        n_trajectories = len(traj_scores)
        
        # Determine number of samples
        n_samples = max(int(n_trajectories * sample_fraction), min_samples)
        
        # Compute softmax probabilities
        scores = traj_scores.values.copy()
        
        # Handle NaN scores (replace with minimum score)
        nan_mask = np.isnan(scores)
        if nan_mask.any():
            scores[nan_mask] = np.nanmin(scores)
        
        # Subtract max for numerical stability
        scores = scores - np.max(scores)
        
        # Softmax
        probs = np.exp(scores / temperature)
        probs = probs / probs.sum()
        
        # Sample trajectory indices
        sampled_indices = rng.choice(
            traj_scores.index.values,
            size=n_samples,
            replace=with_replacement,
            p=probs
        )
        
        # Build sampled dataframe with re-indexed trajectories
        sampled_dfs = []
        for new_idx, orig_idx in enumerate(sampled_indices):
            traj_df = df[df['experiment_index'] == orig_idx].copy()
            traj_df['experiment_index'] = new_idx  # Re-index
            sampled_dfs.append(traj_df)
        
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sampled_df.to_csv(output_path, index=False)
        
        # Collect stats
        sampled_scores = sampled_df.groupby('experiment_index')['trajectory_error_score'].first()
        all_stats.append({
            'file': rel_path,
            'original_count': n_trajectories,
            'sampled_count': n_samples,
            'original_mean_error': traj_scores.mean(),
            'sampled_mean_error': sampled_scores.mean(),
        })
    
    # Print summary
    if all_stats:
        print("\n" + "=" * 80)
        print("SAMPLING SUMMARY")
        print("=" * 80)
        
        total_orig = sum(s['original_count'] for s in all_stats)
        total_sampled = sum(s['sampled_count'] for s in all_stats)
        avg_orig_error = np.mean([s['original_mean_error'] for s in all_stats])
        avg_sampled_error = np.mean([s['sampled_mean_error'] for s in all_stats])
        
        print(f"Total trajectories: {total_orig} → {total_sampled}")
        print(f"Mean error score:   {avg_orig_error:.6f} → {avg_sampled_error:.6f}")
        if avg_orig_error > 0:
            print(f"Error ratio:        {avg_sampled_error / avg_orig_error:.2f}x (higher = more high-error samples)")
        
        print(f"\nOutput saved to: {output_folder}")
    
    print("\n" + "=" * 80)
    print("SAMPLING COMPLETE!")
    print("=" * 80)
    return True


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    sample_trajectories_by_error_score(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        temperature=SAMPLING_TEMPERATURE,
        sample_fraction=SAMPLE_FRACTION,
        min_samples=MIN_SAMPLES_PER_FILE,
        with_replacement=SAMPLE_WITH_REPLACEMENT,
        random_seed=RANDOM_SEED
    )
