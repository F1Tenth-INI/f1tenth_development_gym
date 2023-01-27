import sys
sys.path.insert(1, 'FollowtheGap')


import numpy as np
import math
import matplotlib.pyplot as plt

LOOK_FORWARD_ONLY =True

if LOOK_FORWARD_ONLY:
    lidar_range_min = 200
    lidar_range_max = 880
else:
    lidar_range_min = 0
    lidar_range_max = -1


class FollowTheGapPlanner:
    """
    Example Planner
    """
   
 

    def __init__(self, speed_fraction=1.0):
    
        print("Controller initialized")
    
        self.lidar_border_points = 1080 * [[0,0]]
        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35,2.35, 1080)
        self.simulation_index = 0
        self.speed_fraction = speed_fraction
        self.plot_lidar_data = False
        self.draw_lidar_data = False
        self.lidar_visualization_color = (0, 0, 0)
        self.lidar_live_gaps = []
        self.current_position = None
        self.translational_control = None
        self.angular_control = None
        
        self.draw_position_history = True



    def render(self, e):    
        return
        # Draw points only with render utilities
        # Pyglet doesnt work with ros






    def process_observation(self, ranges=None, ego_odom=None):
        """
        gives actuation given observation
        @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
        @ ego_odom: A dict with following indices:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
            'linear_vel_x': float,
            'linear_vel_y': float,
            'angular_vel_z': float,
        }
        """
        pose_x = ego_odom['pose_x']
        pose_y = ego_odom['pose_y']
        pose_theta = ego_odom['pose_theta']
        
        self.current_position = [pose_x, pose_y, ego_odom['linear_vel_x']]

        scans = np.array(ranges)
        # Take into account size of car
        scans -= 0.3

        distances = scans[lidar_range_min:lidar_range_max:10] # Only use every 10th lidar point
        angles = self.lidar_scan_angles[lidar_range_min:lidar_range_max:10]

        p1 = pose_x + distances * np.cos(angles + pose_theta)
        p2 = pose_y + distances * np.sin(angles + pose_theta)
        points = np.stack((p1, p2), axis=1)

        gaps, distances_filtered, gap_found = find_gaps(distances, angles)

        (largest_gap_angle, largest_gap_index, largest_gap_center, largest_gap_integral, largest_gap_width, largest_gap_max_distance) = find_largest_gap(gaps)

        largest_gap_middle_point = np.array(
            [[pose_x + largest_gap_max_distance * np.cos(largest_gap_center + pose_theta),
             pose_y + largest_gap_max_distance * np.sin(largest_gap_center + pose_theta)]]
        )

        # Speed Calculation
        speed = self.speed_fraction * largest_gap_max_distance
        if(speed < 0.1): speed = 0.1 #Dont stand still


        # print("largest_gap_integral", largest_gap_integral)
        # print("largest_gap_width", largest_gap_width)
        # print("max_distance", max_distance)


        speed = speed  #- 8 * abs(largest_gap_center)

        # Emergency Brake
        if(not gap_found):
            speed = 0.1
            print("Emergency Brake")

        for i in range(50, 58): # Only front cone, not all 180 degree (use 0 to 108 for all lidar points)
            index = 10*i # Only use every 10th lidar point
            if(scans[index] < 0.5):
                speed = 0.0
                print("Emergency Brake: Obstacle in front")

        steering_angle = largest_gap_center

        self.lidar_border_points = 50*largest_gap_middle_point
        # self.lidar_border_points = 50*points
        self.lidar_live_gaps = gaps
        # Plotting
        self.plot_lidar_data_f(angles,
                               distances, distances_filtered,
                               gaps, largest_gap_center)

        self.simulation_index += 1

        self.translational_control = speed
        self.angular_control = steering_angle

        return steering_angle, speed 

    def plot_lidar_data_f(self,
                          angles,
                          distances, distances_filtered,
                          gaps, largest_gap_center, ):

        if self.plot_lidar_data:
            if self.simulation_index % 10 == 0:
                ftg_plot_lidar_data(
                    angles,
                    distances, distances_filtered,
                    gaps, largest_gap_center,
                )


def find_gaps(distances, angles):
    """
    returns list with gaps
    """
    distances = distances.copy()

    # Filter distances

    closest_distance = 10000
    closest_distance_index = 0

    # ignore close points:
    for i in range(len(distances)):

        if (distances[i] < 3):
            distances[i] = 0

        if (distances[i] > 6):
            distances[i] = 6

        # Set points near closest distance to 0
        if (distances[i] < closest_distance):
            closest_distance = distances[i]
            closest_distance_index = i

    # IGNORE neighbors of closest point
    for i in range(closest_distance_index - 3, closest_distance_index + 3):
        if (i < len(distances)):
            distances[i] = 0

    gaps = []
    gap_open = False
    gap_opening_angle = 0
    gap_starting_index = 0
    gap_treshold = 1.499
    max_distance = 0
    gap_found = False
    gap_integral = 0

    for i in range(len(distances) - 1):
        # Rising
        if (not gap_open):
            max_distance = 0
            if (distances[i] < distances[i + 1] - gap_treshold):
                gap_opening_angle = angles[i + 1]  # + math.sin(0.05) * distances[i]
                gap_starting_index = i + 1
                gap_open = True
            if (distances[i + 1] > 6):
                gap_opening_angle = angles[i + 1]
                gap_starting_index = i + 1
                gap_open = True

        # Falling
        if (gap_open):
            gap_integral += distances[i]  # Integrating over gap
            if (max_distance < distances[i]):
                max_distance = distances[i]

            if (distances[i] > distances[i + 1] + gap_treshold):

                # Find out gap width:
                gap_width = i - gap_starting_index
                # print("gap_width",gap_width)
                # if(gap_width > 2):
                gap_closing_angle = angles[i]  # - math.sin(0.05) * distances[i]
                gap_closing_index = i

                # gap: [open angle, closing angle, starting index of distances, closing index of distances, gap integral, gap width]
                gap = [gap_opening_angle, gap_closing_angle, gap_starting_index, gap_closing_index, gap_integral,
                       gap_width, max_distance]

                # The gap has to have a certain area that we recognize it as gap (avoid traps)
                if (gap_integral > 30):
                    gaps.append(gap)

                gap_open = False
                gap_found = True

    distances_filtered = distances
    return gaps, distances_filtered, gap_found


def find_largest_gap(gaps):
    largest_gap_angle = 0
    largest_gap_index = 0
    largest_gap_center = 0
    largest_gap_integral = 0
    largest_gap_width = 0
    largest_gap_max_distance = 6  #FIXME!!! Just heuristic to make it work if no largest gap found
    for i in range(len(gaps)):
        gap = gaps[i]
        gap_angle = abs(gap[1] - gap[0])
        if (gap_angle) > largest_gap_angle:
            largest_gap_angle = gap_angle
            largest_gap_index = i
            largest_gap_center = (gap[0] + gap[1]) / 2
            largest_gap_integral = gap[4]
            largest_gap_width = gap[5]
            largest_gap_max_distance = gap[6]
    largest_gap = (largest_gap_angle, largest_gap_index, largest_gap_center, largest_gap_integral, largest_gap_width, largest_gap_max_distance)
    return largest_gap

def find_largest_gap_middle_point(pose_x, pose_y, pose_theta, distances, angles):

    gaps, distances_filtered, gap_found = find_gaps(distances, angles)

    (largest_gap_angle, largest_gap_index, largest_gap_center, largest_gap_integral, largest_gap_width,
     largest_gap_max_distance) = find_largest_gap(gaps)

    largest_gap_middle_point = np.array(
        [[pose_x + largest_gap_max_distance * np.cos(largest_gap_center + pose_theta),
          pose_y + largest_gap_max_distance * np.sin(largest_gap_center + pose_theta)]]
    )
    largest_gap_middle_point_distance = largest_gap_max_distance

    return largest_gap_middle_point, largest_gap_middle_point_distance, largest_gap_center

def ftg_plot_lidar_data(angles, distances, distances_filtered, gaps, largest_gap_center):
    plt.clf()
    plt.title("Lidar Data")
    plt.plot(angles, distances_filtered)
    plt.plot(angles, distances)

    for gap in gaps:
        plt.axvline(x=gap[0], color='k', linestyle='--')
        plt.axvline(x=gap[1], color='k', linestyle='--')

    plt.axvline(x=largest_gap_center, color='red', linestyle='--')

    plt.savefig("lidar.png")

        

