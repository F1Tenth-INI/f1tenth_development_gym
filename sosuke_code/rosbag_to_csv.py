import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

b=bagreader("F1tenth_data/rosbags/manual_rosbag.bag")
csvfiles = []
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)

print(csvfiles[0])
data = pd.read_csv(csvfiles[0])
