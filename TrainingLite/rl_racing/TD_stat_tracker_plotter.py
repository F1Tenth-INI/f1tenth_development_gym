import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

# Load CSV - only load columns we need to save memory
print("Loading CSV...")
df = pd.read_csv(
    'TrainingLite\\rl_racing\\models\\STAT_TRACK_50k_buffer_train_every_10_03state\\stat_logs\\stats_log.csv',
    usecols=['id', 'TD_error_list', 'sample_count']  # Only load what we need
)
print(f"Loaded {len(df)} rows")

# Find which rows to parse (middle 20)
total_len = len(df)
start_idx = int(total_len/2)
end_idx = start_idx + 20

# Only parse the TD_error_list for rows we're actually plotting
print(f"Parsing rows {start_idx} to {end_idx}...")
df_subset = df.iloc[start_idx:end_idx].copy()
df_subset['TD_error_list'] = df_subset['TD_error_list'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Plot TD error trajectory (x = #samples, y = TD error, line = transition ID)
print("Plotting...")
fig, ax = plt.subplots(figsize=(12, 6))
for idx, row in df_subset.iterrows():
    errors = row['TD_error_list']
    if errors:
        ax.plot(errors, label=f"ID {idx}", alpha=0.6)

ax.set_xlabel('Sample number (times transition was sampled)')
ax.set_ylabel('TD Error')
ax.set_title('TD Error Trajectory Per Transition')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()


# ============ ADDITIONAL PLOTS (OPTIMIZED FOR LARGE FILES) ============

# For each sample iteration, compute mean TD error across all transitions
print("Computing mean TD errors over time...")
all_td_errors = []
# Parse and collect TD errors in batches to avoid memory issues
batch_size = 10000
for batch_start in range(0, total_len, batch_size):
    batch_end = min(batch_start + batch_size, total_len)
    print(f"Processing batch {batch_start} to {batch_end}...")
    batch = df.iloc[batch_start:batch_end].copy()
    batch['TD_error_list'] = batch['TD_error_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    for errors in batch['TD_error_list']:
        if errors and isinstance(errors, list):
            all_td_errors.extend(errors)

# Group by "epoch" (every N samples)
print("Computing epoch means...")
epoch_means = []
window = 100
for i in range(0, len(all_td_errors), window):
    epoch_means.append(np.mean(all_td_errors[i:i+window]))

plt.figure(figsize=(10, 5))
plt.plot(epoch_means, label='Mean TD Error', linewidth=2)
plt.fill_between(range(len(epoch_means)), 0, epoch_means, alpha=0.3)
plt.xlabel('Training Epoch (window size=100)')
plt.ylabel('Mean TD Error')
plt.title('Learning Progress: Mean TD Error Over Time')
plt.grid()
plt.show()


# Convergence: (first - last) / first
print("Computing TD error reductions...")
reductions = []
# Process in batches again
for batch_start in range(0, total_len, batch_size):
    batch_end = min(batch_start + batch_size, total_len)
    print(f"Processing batch {batch_start} to {batch_end}...")
    batch = df.iloc[batch_start:batch_end].copy()
    batch['TD_error_list'] = batch['TD_error_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    for errors in batch['TD_error_list']:
        if isinstance(errors, list) and len(errors) > 1:
            reduction = (errors[0] - errors[-1]) / (errors[0] + 1e-6)
            reductions.append(reduction)

plt.figure(figsize=(10, 5))
plt.scatter(range(len(reductions)), reductions, alpha=0.5)
plt.axhline(y=np.mean(reductions), color='r', linestyle='--', label=f'Mean Reduction: {np.mean(reductions):.2%}')
plt.xlabel('Transition ID')
plt.ylabel('TD Error Reduction %')
plt.title('How Much TD Errors Improved Per Transition')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(reductions, bins=100, range=(-100, 100), alpha=0.7, edgecolor='black')
plt.axvline(x=np.median(reductions), color='r', linestyle='--', linewidth=2, label=f'Median: {np.median(reductions):.1f}%')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.xlabel('TD Error Reduction %')
plt.ylabel('Number of Transitions')
plt.title('Distribution of TD Error Improvements')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


print("Computing sample count vs reduction correlation...")
sample_counts = []
reductions_filtered = []

# Process in batches for the last plot
for batch_start in range(0, total_len, batch_size):
    batch_end = min(batch_start + batch_size, total_len)
    print(f"Processing batch {batch_start} to {batch_end}...")
    batch = df.iloc[batch_start:batch_end].copy()
    batch['TD_error_list'] = batch['TD_error_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    for _, row in batch.iterrows():
        errors = row['TD_error_list']
        count = row['sample_count']
        if isinstance(errors, list) and len(errors) > 1 and count > 0:
            reduction = (errors[0] - errors[-1]) / (errors[0] + 1e-6) * 100
            sample_counts.append(count)
            reductions_filtered.append(reduction)

plt.figure(figsize=(10, 6))
plt.scatter(sample_counts, reductions_filtered, alpha=0.4, s=10)
plt.xlabel('Times Sampled (prioritization indicator)')
plt.ylabel('TD Error Reduction %')
plt.title('Did High-Priority Transitions Improve More?')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.show()

print(f"Transitions with improvement (>0%): {sum(r > 0 for r in reductions) / len(reductions):.1%}")
print(f"25th percentile: {np.percentile(reductions, 25):.1f}%")
print(f"Median: {np.percentile(reductions, 50):.1f}%")
print(f"75th percentile: {np.percentile(reductions, 75):.1f}%")
print(f"Mean (of positives only): {np.mean([r for r in reductions if r > 0]):.1f}%")