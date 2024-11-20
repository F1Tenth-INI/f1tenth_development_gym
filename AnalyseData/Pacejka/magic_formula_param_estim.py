import rosbag
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

import sys
import os

sys.path.insert(0, os.getcwd())

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

bag = rosbag.Bag('/Users/Florian/Documents/INI/F1TENTH/f1tenth_development_gym/AnalyseData/Pacejka/data/pp_ETF1_2023-10-24-17-25-09.bag')

imu_data = []
car_state_data = []

for topic, msg, t in bag.read_messages(topics=['/imu/data', '/car_state/tum_state']):
    if topic == '/imu/data':
        imu_data.append({'time': t.to_sec(), 'f_x': msg.linear_acceleration.x, 'f_y': msg.linear_acceleration.y,'f_z': msg.linear_acceleration.z})
    elif topic == '/car_state/tum_state':
        # Assuming msg has attributes x, y
        car_state_data.append({'time': t.to_sec(), 'steering_angle': msg.steering_angle})

imu_df = pd.DataFrame(imu_data)

bag.close()



car_state_df = pd.DataFrame(car_state_data)

# imu_df = pd.DataFrame(imu_data).set_index('time')
# car_state_df = pd.DataFrame(car_state_data).set_index('time')

# Merge the two dataframes on the 'time' column
merged_df = pd.merge_asof(imu_df, car_state_df, on='time', direction='nearest')


# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 2.667  # desired cutoff frequency of the filter, Hz

# Apply the filter to all columns of imu_df
for column in ['f_x', 'f_y', 'f_z']:
    merged_df[column] = butter_lowpass_filter(merged_df[column], cutoff, fs, order)


merged_df['steering_angle'] = butter_lowpass_filter(merged_df['steering_angle'], cutoff, fs, order)


plt.figure(figsize=(10, 6))
plt.plot(merged_df['f_y'], label='f_y')
plt.plot(merged_df['f_z'], label='f_z')
plt.plot(10 * merged_df['steering_angle'], label='steering_angle')
plt.title('f_y over time')
plt.xlabel('Index')
plt.ylabel('f_y')
plt.legend()
plt.savefig("f_y_and_steering_angle_over_time.png")



plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['steering_angle'], merged_df['f_y'])
plt.title('steering_angle vs f_y ')
plt.xlabel('steering_angle')
plt.ylabel('f_y')
plt.savefig("steering_angle_vs_f_y.png")