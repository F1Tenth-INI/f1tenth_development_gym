import yaml
import numpy as np
from utilities.Settings import Settings

class LIDAR:
    def __init__(self):

        # General information about lidar
        config = yaml.load(open("config.yml"),
                                         Loader=yaml.FullLoader)
        self.covered_angle_deg = config['LIDAR']['covered_angle_deg']
        self.covered_angle_rad = np.deg2rad(self.covered_angle_deg)
        self.num_scans_total = config['LIDAR']['num_scans']

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
            self.processed_scan_indices = self.get_custom_processed_scan_indices()
            self.decimation = self.num_scans_total/len(self.processed_scan_indices)
        else:
            raise NotImplementedError


        self.processed_number_of_scans = len(self.processed_scan_indices)
        self.processed_angles_rad = self.scan_angles_all_rad[self.processed_scan_indices]
        self.processed_scans = None

        self.points_relative_to_car = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)
        self.points_map_coordinates = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)

    def get_processed_lidar_distances(self, all_lidar_scans):
        self.processed_scans = all_lidar_scans[self.processed_scan_indices]

    def get_lidar_points_in_map_coordinates_from_processed_scans(self, processed_lidar_scans, car_x, car_y, car_yaw):
        p1 = car_x + processed_lidar_scans * np.cos(self.processed_angles_rad + car_yaw)
        p2 = car_y + processed_lidar_scans * np.sin(self.processed_angles_rad + car_yaw)
        self.points_map_coordinates[...] = np.stack((p1, p2), axis=1)
        return self.points_map_coordinates

    def get_lidar_points_in_map_coordinates_from_all_scans(self, all_lidar_scans, car_x, car_y, car_yaw):
        self.get_processed_lidar_distances(all_lidar_scans)
        self.get_lidar_points_in_map_coordinates_from_processed_scans(self.processed_scans, car_x, car_y, car_yaw)
        return self.points_map_coordinates

    def reinitialized_LIDAR_with_custom_processed_scan_indices(self, processed_scan_indices):
        self.processed_scan_indices = processed_scan_indices

        self.processed_number_of_scans = len(self.processed_scan_indices)
        self.processed_angles_rad = self.scan_angles_all_rad[self.processed_scan_indices]
        self.processed_scans = None

        self.points_relative_to_car = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)
        self.points_map_coordinates = np.zeros((self.processed_number_of_scans, 2), dtype=np.float32)


    def get_custom_processed_scan_indices(self):
        raise NotImplementedError

