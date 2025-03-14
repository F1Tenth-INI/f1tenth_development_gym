import os

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()

import pandas as pd

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results, window_func


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from TrainingLite.rl_racing.train_model import log_dir, model_name

print(log_dir)
timesteps = 1e8

fig, ax = plt.subplots(figsize=(10,4))

results_df = load_results(log_dir)
results_df = results_df[results_df.l.cumsum() <= timesteps]
x_all, y_all = ts2xy(results_df, results_plotter.X_TIMESTEPS)

x, y = x_all, y_all
# max_model_timestep = np.max(x)  

# plt.scatter(x=x, y=y, s=1)
# sns.scatterplot(x=x, y=y, ax=ax, size=2)#, alpha=0.5, linecolor="black", linewidths=0.1)
# x, y_mean = results_plotter.window_func(x, y, 100, np.mean)
# ax.plot(x, y_mean, color="orange")

window_smoothing = 200
x, y_mean = results_plotter.window_func(x_all, y_all, window_smoothing, np.mean)
x, y_std = window_func(x_all, y_all, window_smoothing, np.std)
ax.fill_between(x, y_mean + y_std, y_mean - y_std, color="C0", alpha=0.3)
ax.plot(x, y_mean, color="red", alpha=1.0)
# sns.lineplot(x=x_all, y=y_all)

# ax.set_title(f"Unconstrained")
ax.set_ylabel("Episode Rewards")
ax.set_xlabel("Timestep")

# ax.set_xlim(0, 5760000)#8640000) #, 22080000)#19200000) #  28320000)# 8640000)#20640000)#5200000)#7200000)
# ax.get_legend().remove()
plt.show()