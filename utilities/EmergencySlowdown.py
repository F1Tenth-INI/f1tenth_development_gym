import numpy as np


class EmergencySlowdown:
    """
    Emergency slowdown idea:
    We count the lidar measurements within a stripe in from of the car
    of width D starting at distance L_min and ending at L_max.
    We check if there is at least min_number_of_scans_for_obstacle_far to indicate obstacle.
    We calculate the distances from the car corresponding to the measurements.
    In take_minimal_possible_collision_distance if yes, we take the smallest measurement.
    Otherwise we group the measurements into possible objects of size up to max_distance_split_of_single_object.
    We check any of these possible objects has at least min_number_of_scans_for_obstacle measurements,
    adjusted for density of the scans, diminishing with distance.
    Take the smallest distance to the closest possible object satisfying the condition as L_impact.
    We calculate the speed reduction under assumption that it is 0 if L_impact<=L_stop and 1 if L_impact>=L_max and linearly interpolate between these values.
    """
    def __init__(
            self,
            D=0.45,  # Width of the observed stripe
            L_min=0.18,  # Minimum distance at which appearing points are checked
            L_stop=0.5,  # Full breaking distance (reduction_factor =  0.0)
            L_max=2.0,  # Full speed distance (reduction_factor = 1.0)
            min_num_of_scans_for_obstacle=20,  # How many scans need to be registered at a minimal distance to consider it as an obstacle
            min_number_of_scans_for_obstacle_far=2,  # How many scans need to be registered at a maximal distance to consider it as an obstacle
            max_distance_split_of_single_object=0.5,  # We group lidar scans which possibly mean collisions into objects depending on lidar measurement.
            take_minimal_possible_collision_distance=True
    ):
        self.D_half = D / 2.0

        self.L_min = L_min
        self.L_stop = L_stop
        self.L_max = L_max

        self.take_minimal_possible_collision_distance = take_minimal_possible_collision_distance

        self.min_num_of_scans_for_obstacle = min_num_of_scans_for_obstacle

        # The measaurements further away are weighted with scans_thresholds to account for diminishing density of scans with distance

        self.max_distance_split_of_single_object = max_distance_split_of_single_object

        self.lidar_angles = np.linspace(-2.35,2.35, 1080)
        self.lidar_angles_cos = np.cos(self.lidar_angles)
        self.scans_thresholds = self.D_half / np.sin(abs(self.lidar_angles))  # This is maximal distance which can be observed within the observed stripe with a given angle. Also all smaller distances for this angle will be within relevant stripe. Moreover as the density of scans diminishes with distance, this values are the weight used to assign importance to readings.
        self.scans_threshold_at_L_min = np.sqrt(self.D_half**2 + self.L_min**2)
        self.scans_threshold_at_L_min_inv = 1.0/self.scans_threshold_at_L_min

        alpha_max = np.arctan(self.D_half/L_min)
        self.scan_max = np.sqrt(self.D_half**2 + L_max**2)

        self.min_number_of_scans_for_obstacle_far = min_number_of_scans_for_obstacle_far

        self.scan_indices_within_alpha_max = np.where(abs(self.lidar_angles) <= alpha_max)

    def calculate_speed_reduction(self, lidar_scans):

        # initial filtering using precomputed indices to reduce computational load
        scans_within_alpha_max = lidar_scans[self.scan_indices_within_alpha_max]
        scans_thresholds_within_alpha_max = self.scans_thresholds[self.scan_indices_within_alpha_max]

        # Finding measurements within the stripe of interest
        scan_indices_with_measurement_within_stripe = np.where(
            (scans_within_alpha_max < scans_thresholds_within_alpha_max) & (scans_within_alpha_max > 0) & (scans_within_alpha_max < self.scan_max)
        )[0]
        if len(scan_indices_with_measurement_within_stripe) < self.min_number_of_scans_for_obstacle_far:
            return 1.0  # Not enough measurements indicating possible collision within the stripe
        scans_within_stripe = scans_within_alpha_max[scan_indices_with_measurement_within_stripe]
        angles_cos_within_stripe = self.lidar_angles_cos[self.scan_indices_within_alpha_max][scan_indices_with_measurement_within_stripe]
        scans_density_correction_within_stripe = scans_within_stripe * self.scans_threshold_at_L_min_inv
        collision_distances_within_stripe = angles_cos_within_stripe*scans_within_stripe

        if self.take_minimal_possible_collision_distance:
            L_impact = np.min(collision_distances_within_stripe)  # Not as precise as alternative below, but faster
            speed_reduction_factor = self.speed_reduction_factor_from_L_impact(L_impact)
            return speed_reduction_factor

        sorted_indices = np.argsort(collision_distances_within_stripe)
        sorted_collision_distances = collision_distances_within_stripe[sorted_indices]
        sorted_scans_density_correction = scans_density_correction_within_stripe[sorted_indices]

        event_collision_distances, event_weigths = self.grouping_into_objects(
            sorted_collision_distances,
            sorted_scans_density_correction,
            self.max_distance_split_of_single_object
        )
        indices_collision = np.where(event_weigths > self.scans_threshold_at_L_min)[0]
        if len(indices_collision) == self.min_number_of_scans_for_obstacle_far:
            return 1.0  # To few measurements to consider it as an obstacle

        first_index_collision = indices_collision[0]
        L_impact = event_collision_distances[first_index_collision]
        speed_reduction_factor = self.speed_reduction_factor_from_L_impact(L_impact)
        print(f"Speed reduction factor: {speed_reduction_factor}")
        return speed_reduction_factor

    def speed_reduction_factor_from_L_impact(self, L_impact):
        speed_reduction_factor = min(max(L_impact - self.L_stop, 0.0) / (self.L_max - self.L_stop), 1.0)
        return speed_reduction_factor

    @staticmethod
    def grouping_into_objects(sorted_collision_distances, sorted_scans_density_correction, max_distance_split_of_single_object=0.5):
        i_last = 0
        current_object_total_weight = 0.0
        object_weights = []
        object_collision_distances = [sorted_collision_distances[i_last]]
        for i in range(len(sorted_collision_distances)):
            current_object_total_weight += sorted_scans_density_correction[i]
            if sorted_collision_distances[i] - sorted_collision_distances[i_last] > max_distance_split_of_single_object:
                object_weights.append(current_object_total_weight)
                while sorted_collision_distances[i] - sorted_collision_distances[i_last] > max_distance_split_of_single_object:
                    i_last += 1
                    object_collision_distances.append(sorted_collision_distances[i_last])  # Add new object
                    current_object_total_weight = 0.0
        object_weights.append(current_object_total_weight)

        return object_collision_distances, object_weights
