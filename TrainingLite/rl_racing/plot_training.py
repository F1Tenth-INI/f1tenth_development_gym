import pandas as pd
import os
import matplotlib.pyplot as plt
import ast

model_name = "SAC_RCA1_wpts_lidar_44"
training_index = 4


def plot_training_csv(model_name, training_index):

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(root_dir, "TrainingLite","rl_racing","models", model_name)
    csv_path=os.path.join(model_dir, f'training_log_{training_index}.csv')

    if not os.path.exists(csv_path):
        csv_path = os.path.join(model_dir, 'training_log.csv')

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Load the CSV
    df = pd.read_csv(csv_path)

    X_AXIS = "global_step"  # or "episode" or "step"
    SAVE_PATH = os.path.join(model_dir, f'training_plot.png')
    # Drop rows with missing reward

    # Load CSV
    df = pd.read_csv(csv_path)

    # Drop any rows without global step
    df = df.dropna(subset=[X_AXIS])

    # Extract min laptime from laptimes column
    if "laptimes" in df.columns:
        min_laptimes = []
        for laptimes_str in df["laptimes"]:
            try:
                laptimes = ast.literal_eval(laptimes_str) if laptimes_str and laptimes_str != '[]' else []
            except Exception:
                laptimes = []
            min_laptimes.append(min(laptimes) if laptimes else float('nan'))
        df["laptimes_min"] = min_laptimes

    # Automatically find numeric columns to plot
    plot_columns = ["episode", "reward", "length", "laptime_min"]
    if "laptimes_min" in df.columns:
        plot_columns.append("laptimes_min")

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
    fig.suptitle(f"Training Progress - {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(SAVE_PATH)
    print(f"Plot saved to: {SAVE_PATH}")



if __name__ == "__main__":
    plot_training_csv(model_name, training_index)
