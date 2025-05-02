import numpy as np
import matplotlib.pyplot as plt

from utilities.waypoint_utils import *


class GenerativeMapCalculator:
    def __init__(self):
        self.waypoint_distance = 0.1  # Distance between waypoints [m]

    def create_waypoint_sequence(self, num_waypoints=1000):
        waypoints = []
        current_position = np.array([0.0, 0.0])
        current_angle = 0.0

        while len(waypoints) < num_waypoints:
            curve_length = np.random.uniform(0.5, 10.0)  # meters
            curvature_start = np.random.uniform(-0.5, 0.5)  # 1/m
            curvature_end = np.random.uniform(-0.5, 0.5)

            segment = self.create_clothoid_segment(
                current_position,
                current_angle,
                curvature_start,
                curvature_end,
                curve_length
            )

            if not segment:
                break  # prevent infinite loop on error

            segment_points, final_pos, final_angle = segment

            waypoints.extend(segment_points)
            current_position = final_pos
            current_angle = final_angle

        waypoints = waypoints[:num_waypoints]

        # Plot for visualization
        x, y = zip(*waypoints)
        plt.plot(x, y, marker='o', markersize=1, linewidth=1)
        plt.axis('equal')
        plt.title("Generated Clothoid-like Waypoint Sequence")
        plt.show()

        return waypoints

    def create_clothoid_segment(self, start_pos, start_angle, k0, k1, length):
        """
        Generate a clothoid-like segment where curvature changes linearly.
        :param start_pos: Starting point (x, y)
        :param start_angle: Starting heading angle in radians
        :param k0: Initial curvature (1/m)
        :param k1: Final curvature (1/m)
        :param length: Total arc length of the segment [m]
        :return: list of (x, y) waypoints, final position, final heading angle
        """
        ds = self.waypoint_distance
        n_points = int(length / ds)

        if n_points < 2:
            return None

        points = []
        x, y = start_pos
        theta = start_angle

        for i in range(n_points):
            s = i * ds
            t = s / length
            k = (1 - t) * k0 + t * k1  # Linear interpolation

            theta += k * ds
            x += ds * np.cos(theta)
            y += ds * np.sin(theta)
            points.append((x, y))

        final_pos = np.array([x, y])
        final_theta = theta

        return points, final_pos, final_theta


if __name__ == "__main__":
    
    if Settings.MAP_NAME != 'Blank':
        print("Error: This script is only for generating waypoints for the Blank map. Please change it in Setttings.py to Blank.")
        raise ValueError("This script is only for generating waypoints for the Blank map.")
    
    gen = GenerativeMapCalculator()
    waypoint_positions = gen.create_waypoint_sequence(num_waypoints=10000)
    
    
    waypoints = np.zeros((len(waypoint_positions), 7))
    waypoints[:, WP_S_IDX] = gen.waypoint_distance * np.arange(len(waypoint_positions))
    waypoints[:, WP_X_IDX] = np.array(waypoint_positions)[:, 0]
    waypoints[:, WP_Y_IDX] = np.array(waypoint_positions)[:, 1]
    waypoints[:, WP_PSI_IDX] = np.arctan2(np.diff(waypoints[:, WP_Y_IDX], prepend=0), np.diff(waypoints[:, WP_X_IDX], prepend=0)) + np.pi / 2
    waypoints[:, WP_KAPPA_IDX] = np.gradient(waypoints[:, WP_PSI_IDX], gen.waypoint_distance)
    waypoints[:, WP_VX_IDX] = 5.0  # Placeholder for velocity


    # Save waypoints to a csv file
    file_path = 'utilities/maps/Blank/Blank_wp.csv'
    np.savetxt(file_path, waypoints, delimiter=",", header="s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2", comments='')    
    
    
    print(f"Generated {len(waypoint_positions)} waypoints.")
