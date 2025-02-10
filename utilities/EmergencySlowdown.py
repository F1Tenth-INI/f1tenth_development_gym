import numpy as np

SWITCH_LINE_AFTER_X_TIMESSTEPS_BRAKING = 400
KEEP_LINE_FOR_MIN_X_TIMESTEPS_FREERIDE = 20

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
            L_max=1.5,  # Full speed distance (reduction_factor = 1.0)
            min_num_of_scans_for_obstacle=20,  # How many scans need to be registered at a minimal distance to consider it as an obstacle
            min_number_of_scans_for_obstacle_far=2,  # How many scans need to be registered at a maximal distance to consider it as an obstacle
            max_distance_split_of_single_object=0.5,  # We group lidar scans which possibly mean collisions into objects depending on lidar measurement.
            take_minimal_possible_collision_distance=True
    ):
        self.D_half = D / 2.0

        self.L_min = L_min
        self.L_stop = L_stop
        self.L_max = L_max

        self.movement_direction = 0.0

        self.take_minimal_possible_collision_distance = take_minimal_possible_collision_distance

        self.min_num_of_scans_for_obstacle = min_num_of_scans_for_obstacle

        # The measaurements further away are weighted with scans_thresholds to account for diminishing density of scans with distance

        self.max_distance_split_of_single_object = max_distance_split_of_single_object

        self.lidar_angles = None
        self.lidar_angles_cos = None
        self.scans_thresholds = None
        # self.lidar_angles = np.linspace(-2.35,2.35, 1080)
        # self.lidar_angles_cos = np.cos(self.lidar_angles)
        # self.scans_thresholds = self.D_half / np.sin(abs(self.lidar_angles))  # This is maximal distance which can be observed within the observed stripe with a given angle. Also all smaller distances for this angle will be within relevant stripe. Moreover as the density of scans diminishes with distance, this values are the weight used to assign importance to readings.

        self.scans_threshold_at_L_min = np.sqrt(self.D_half**2 + self.L_min**2)
        self.scans_threshold_at_L_min_inv = 1.0/self.scans_threshold_at_L_min

        self.alpha_max = np.arctan(self.D_half/L_min)
        self.scan_max = np.sqrt(self.D_half**2 + L_max**2)

        self.min_number_of_scans_for_obstacle_far = min_number_of_scans_for_obstacle_far

        self.scan_indices_within_alpha_max = None
        # self.scan_indices_within_alpha_max = np.where(abs(self.lidar_angles) <= self.alpha_max)

        # Line switching code
        self.use_alternative_waypoints_for_control_flag = False
        self.counter_jam = 0
        self.counter_freeride = 0

        self.speed_reduction_factor = 1.0

        self.emergency_slowdown_sprites = None

    def calculate_speed_reduction(self, lidar_scans, lidar_angles, steering_angle):

        self.movement_direction = steering_angle  # Default to steering direction if speed is 0
        # # Calculate heading direction based on velocity vector
        # if vx == 0 and vy == 0:
        #     self.movement_direction = steering_angle  # Default to steering direction if speed is 0
        # else:
        #     self.movement_direction = np.arctan2(vy, vx)  # Velocity direction

        # Adjust lidar angles relative to the heading direction

        self.lidar_angles = lidar_angles
        self.lidar_angles_cos = np.cos(self.lidar_angles)
        self.scans_thresholds = self.D_half / np.sin(abs(self.lidar_angles))  # This is maximal distance which can be observed within the observed stripe with a given angle. Also all smaller distances for this angle will be within relevant stripe. Moreover as the density of scans diminishes with distance, this values are the weight used to assign importance to readings.
        self.scan_indices_within_alpha_max = np.where(abs(self.lidar_angles) <= self.alpha_max)

        adjusted_lidar_angles = self.lidar_angles - self.movement_direction
        adjusted_angles_cos = np.cos(adjusted_lidar_angles)

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
        adjusted_angles_cos_within_stripe = adjusted_angles_cos[self.scan_indices_within_alpha_max][scan_indices_with_measurement_within_stripe]
        scans_density_correction_within_stripe = scans_within_stripe * self.scans_threshold_at_L_min_inv
        collision_distances_within_stripe = adjusted_angles_cos_within_stripe * scans_within_stripe

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


    def update_emergency_slowdown_sprites(self, car_x, car_y, car_yaw):
        # Update emergency slowdown boundary lines based on current movement direction.
        # These lines define the boundaries of the detection stripe in the car's frame.
        # They are computed in the car's local coordinate system and then rotated to align with movement_direction.
        def transform_point(x, y, angle):
            # Transforms a local coordinate (x,y) into global coordinates based on heading angle.
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            return np.array((x * cos_a - y * sin_a, x * sin_a + y * cos_a))

        current_position = np.array((car_x, car_y))
        direction = car_yaw + self.movement_direction
        self.emergency_slowdown_sprites = {}
        left_start_global = transform_point(self.L_min, self.D_half, direction) + current_position
        left_end_global = transform_point(self.L_max, self.D_half, direction) + current_position
        self.emergency_slowdown_sprites["left_line"] = [left_start_global, left_end_global]

        right_start_global = transform_point(self.L_min, -self.D_half, direction) + current_position
        right_end_global = transform_point(self.L_max, -self.D_half, direction) + current_position
        self.emergency_slowdown_sprites["right_line"] = [right_start_global, right_end_global]

        # Stop line: a line perpendicular to movement direction at distance L_stop, spanning the stripe width.
        # Defined from (L_stop, -D_half) to (L_stop, +D_half) in the local frame.
        stop_left_global = transform_point(self.L_stop, -self.D_half, direction) + current_position
        stop_right_global = transform_point(self.L_stop, self.D_half, direction) + current_position
        self.emergency_slowdown_sprites["stop_line"] = [stop_left_global, stop_right_global]

        self.emergency_slowdown_sprites["speed_reduction_factor"] = self.speed_reduction_factor

        # Compute a display position for the speed reduction factor text.
        # Use the midpoint of the left line and add a small offset perpendicular to the line (in car's frame, upward is (0,1)).
        left_midpoint = (left_start_global + left_end_global) / 2
        offset_distance = 0.5  # Adjust this constant as needed for proper spacing.
        offset_vector = transform_point(0, offset_distance, direction)
        text_position = left_midpoint + offset_vector
        self.emergency_slowdown_sprites["display_position"] = text_position


    def stop_if_obstacle_in_front(self, lidar_scans, lidar_angles, next_waypoints_vx, steering_angle):
        self.speed_reduction_factor = self.calculate_speed_reduction(lidar_scans, lidar_angles, steering_angle)
        corrected_next_waypoints_vx = next_waypoints_vx * self.speed_reduction_factor

        if self.speed_reduction_factor < 0.5:
            self.counter_jam += 1
            self.counter_freeride = 0
        else:
            self.counter_freeride += 1

        if self.counter_jam >= SWITCH_LINE_AFTER_X_TIMESSTEPS_BRAKING:
            self.use_alternative_waypoints_for_control_flag = not self.use_alternative_waypoints_for_control_flag
            self.counter_jam = 0
            print("Switching to alternative raceline")

        if self.counter_freeride >= KEEP_LINE_FOR_MIN_X_TIMESTEPS_FREERIDE:
            self.counter_jam = 0

        # if speed_reduction_factor != 1.0:
        #     print(f"Braking with speed reduction {speed_reduction_factor}")


        return corrected_next_waypoints_vx, self.use_alternative_waypoints_for_control_flag
