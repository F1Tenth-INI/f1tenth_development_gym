import numpy as np
from utilities.Settings import Settings
import platform

# from scipy.ndimage import gaussian_filter

from matplotlib import pyplot as plt
import matplotlib


class LidarHelper:
    def __init__(self):

        # General information about lidar

        self.covered_angle_deg = Settings.LIDAR_COVERED_ANGLE_DEG
        self.covered_angle_rad = np.deg2rad(self.covered_angle_deg)
        self.num_scans_total = Settings.LIDAR_NUM_SCANS

        self.scan_angles_all_rad = np.linspace(
            -self.covered_angle_rad/2.0,
            self.covered_angle_rad/2.0,
            self.num_scans_total
        )

        # Data used for current experiment
        # Options:
        # provide lidar angle and decimation
        # provide a set of indices (To-be-done) allow to get more scans where higher precision is required,
        # provide lidar angle and number of scans (even numbers only)

        if Settings.LIDAR_PROCESSED_ANGLE_DEG == 'max':
            self.processed_angle_deg = self.covered_angle_deg
        else:
            self.processed_angle_deg = Settings.LIDAR_PROCESSED_ANGLE_DEG

        self.num_scan_indices_within_processed_angle = 2*np.ceil(0.5*self.num_scans_total*(self.processed_angle_deg/self.covered_angle_deg))

        half_unprocessed_indices = (self.num_scans_total-self.num_scan_indices_within_processed_angle)/2
        self.scan_indices_within_processed_angle = np.arange(self.num_scans_total)[int(np.ceil(half_unprocessed_indices)):self.num_scans_total-int(np.floor(half_unprocessed_indices))]

        self.processed_number_of_scans = None
        self.decimation = None

        if Settings.LIDAR_MODE == 'decimation':
            self.decimation = Settings.LIDAR_DECIMATION
            self.processed_scan_indices = self.scan_indices_within_processed_angle[::self.decimation]
        elif Settings.LIDAR_MODE == 'custom indices':
            self.decimation = Settings.LIDAR_DECIMATION
            self.custom_sigma = Settings.LIDAR_CUSTOM_STDEV
            self.processed_scan_indices = self.get_custom_processed_scan_indices()
            self.decimation = self.num_scans_total/len(self.processed_scan_indices)
        else:
            raise NotImplementedError


        self.processed_number_of_scans = len(self.processed_scan_indices)
        self.processed_angles_rad = self.scan_angles_all_rad[self.processed_scan_indices]
        self.all_lidar_scans = None
        self.processed_scans = None

        self.points_relative_to_car = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)
        self.points_map_coordinates = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)

        # CORRUPT LIDAR
        self.max_corrupted_ratio = Settings.LIDAR_MAX_CORRUPTED_RATIO
        self.max_number_of_corrupted_indices = int(np.floor(self.max_corrupted_ratio*self.processed_number_of_scans))
        self.corrupted_indices_of_processed_scans = None

        self.lidar_names = None

    def indices_from_pandas(self, data):
        lidar_col = [col for col in data if col.startswith('LIDAR')]
        processed_scan_indices = [int(lidar_col[i][len('LIDAR_'):]) for i in range(len(lidar_col))]
        return processed_scan_indices

    def get_lidar_scans_names(self):
        self.lidar_names = ['LIDAR_' + str(i).zfill(4) for i in self.processed_scan_indices]
        return self.lidar_names

    def corrupt_lidar_set_indices(self):
        number_of_corrupted_indices = np.random.randint(0, self.max_number_of_corrupted_indices+1)
        self.corrupted_indices_of_processed_scans = np.random.choice(np.arange(self.processed_number_of_scans), (number_of_corrupted_indices,), replace=False)

    def corrupt_scans(self):
        """ MPC expect that corrupted lidar scans have high value - not to create crash cost """
        self.processed_scans[self.processed_scans > 10.0] = 70.0
        self.processed_scans[self.corrupted_indices_of_processed_scans] = 70.0

    def corrupted_scans_high2zero(self):
        self.processed_scans[self.processed_scans > 15.0] = 0.0

    def corrupt_datafile(self, data):

        lidar_indices = self.indices_from_pandas(data)
        self.reinitialized_LIDAR_with_custom_processed_scan_indices(lidar_indices)
        lidar_col = self.get_lidar_scans_names()

        df_data_lidar = data.loc[:, lidar_col]
        df_data_lidar[df_data_lidar > 10.0] = 0.0
        for k in range(len(df_data_lidar)):
            self.corrupt_lidar_set_indices()
            df_data_lidar.iloc[k, self.corrupted_indices_of_processed_scans] = 0.0

        # count = (df_data_lidar == 0.0).sum() See how many scans are corrupted

        data.loc[:, lidar_col] = df_data_lidar
        return data


    def load_lidar_measurement(self, all_lidar_scans):
        self.all_lidar_scans = all_lidar_scans
        self.processed_scans = all_lidar_scans[self.processed_scan_indices]

    @staticmethod
    def get_lidar_points_in_map_coordinates_from_scans(
            lidar_scans,
            angles_rad,
            car_x, car_y, car_yaw):
        p1 = car_x + lidar_scans * np.cos(angles_rad + car_yaw)
        p2 = car_y + lidar_scans * np.sin(angles_rad + car_yaw)
        return np.stack((p1, p2), axis=1)

    def get_all_lidar_points_in_map_coordinates(self, car_x, car_y, car_yaw):
        return self.get_lidar_points_in_map_coordinates_from_scans(self.all_lidar_scans, self.scan_angles_all_rad, car_x, car_y, car_yaw)

    def get_processed_lidar_points_in_map_coordinates(self, car_x, car_y, car_yaw):
        return self.get_lidar_points_in_map_coordinates_from_scans(self.processed_scans, self.processed_angles_rad, car_x, car_y, car_yaw)

    def reinitialized_LIDAR_with_custom_processed_scan_indices(self, processed_scan_indices):
        self.processed_scan_indices = processed_scan_indices

        self.processed_number_of_scans = len(self.processed_scan_indices)
        self.processed_angles_rad = self.scan_angles_all_rad[self.processed_scan_indices]
        self.processed_scans = None

        self.points_relative_to_car = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)
        self.points_map_coordinates = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)


    def get_custom_processed_scan_indices(self):
        num_resample = len(self.scan_indices_within_processed_angle[::self.decimation])
        min_index, max_index = self.scan_indices_within_processed_angle[0], self.scan_indices_within_processed_angle[-1]
        sample_mean_index = int((max_index - min_index) / 2)
        sample_sigma = self.custom_sigma / self.covered_angle_deg * (max_index - min_index)
        samples = np.random.normal(sample_mean_index, sample_sigma, num_resample)
        samples = np.round(samples).astype(int)

        indices = np.clip(samples, min_index, max_index)
        return indices
        # raise NotImplementedError

    def plot_lidar_data(self):
        if platform.system() == 'Darwin':
            matplotlib.use('MacOSX')

        plt.ion()
        plt.clf()
        plt.plot(self.processed_scans)
        plt.show()
        plt.clf()

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('./ExperimentRecordings/F1TENTH_Blank-MPPI-1__2023-05-22_13-55-34.csv', comment='#')
    Lidar = LidarHelper()
    df_new = Lidar.corrupt_datafile(df)
    pass
