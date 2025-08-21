import pandas as pd
import os
import matplotlib.pyplot as plt

model_name = "SAC_RCA1_wpts_lidar_7"
training_index = 19


def plot_training_csv(model_name, training_index):

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(root_dir, "TrainingLite","rl_racing","models", model_name)
    csv_path=os.path.join(model_dir, f'training_log_{training_index}.csv')
    # Load the CSV
    df = pd.read_csv(csv_path)

    X_AXIS = "global_step"  # or "episode" or "step"
    SAVE_PATH = "training_summary.png"
    # Drop rows with missing reward

    # Load CSV
    df = pd.read_csv(csv_path)

    # Drop any rows without global step
    df = df.dropna(subset=[X_AXIS])

    # Automatically find numeric columns to plot
    plot_columns = ["episode", "reward", "length", "lap_time", "crashed", "laptime_min"]

    # Set up subplots
    n = len(plot_columns)
    fig, axs = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axs = [axs]  # make iterable if single plot

    for i, col in enumerate(plot_columns):
        ax = axs[i]
        ax.scatter(df[X_AXIS], df[col], s=10)  # Use scatter for dots, s=10 sets the dot size
        ax.set_ylabel(col.replace("_", " ").title())
        ax.grid(True)

    axs[-1].set_xlabel(X_AXIS.replace("_", " ").title())
    fig.suptitle("Training Progress", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(SAVE_PATH)
    print(f"Plot saved to: {SAVE_PATH}")



if __name__ == "__main__":
    plot_training_csv(model_name, training_index)
