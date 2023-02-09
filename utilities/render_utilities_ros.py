import numpy as np
from utilities.state_utilities import *
import rospy
from visualization_msgs.msg import Marker, MarkerArray


class RenderUtils:
    def __init__(self):
        
        self.pub_rollout = rospy.Publisher('mppi/rollout', MarkerArray, queue_size=10)


        self.draw_lidar_data = True
        self.draw_position_history = True
        self.draw_waypoints = True
        self.draw_next_waypoints = True

        self.waypoint_visualization_color = (180, 180, 180)
        self.next_waypoint_visualization_color = (0, 127, 0)
        self.lidar_visualization_color = (255, 0, 255)
        self.gap_visualization_color = (0, 255, 0)
        self.mppi_visualization_color = (250, 25, 30)
        self.optimal_trajectory_visualization_color = (255, 165, 0)
        self.target_point_visualization_color = (255, 204, 0)
        self.position_history_color = (0, 204, 0)

        self.waypoint_vertices = None
        self.next_waypoint_vertices = None
        self.lidar_vertices = None
        self.gap_vertex = None
        self.mppi_rollouts_vertices = None
        self.optimal_trajectory_vertices = None
        self.target_vertex = None

        self.waypoints = None
        self.next_waypoints = None
        self.lidar_border_points = None
        self.rollout_trajectory, self.traj_cost = None, None
        self.optimal_trajectory = None
        self.largest_gap_middle_point = None
        self.target_point = None
        self.car_state = None
        

    # Pass all data that is updated during simulation
    def update(self, 
               lidar_points=None, 
               rollout_trajectory=None, 
               traj_cost=None, 
               optimal_trajectory=None,
               largest_gap_middle_point=None, 
               target_point=None, 
               next_waypoints=None,
               car_state = None,):
        
        
        print("Rollout trajectory", np.array(rollout_trajectory).shape)
        self.lidar_border_points = lidar_points
        self.rollout_trajectory, self.traj_cost = rollout_trajectory, traj_cost
        self.optimal_trajectory = optimal_trajectory
        self.largest_gap_middle_point = largest_gap_middle_point
        self.target_point = target_point
        self.next_waypoints = next_waypoints
        self.car_state = car_state
        
        if self.rollout_trajectory is not None:
            self.rollout_trajectory = self.rollout_trajectory[:5]

        
        if(self.next_waypoints == []):self.next_waypoints = None

    def render(self, e):
        # Todo: Publish data for RVIZ
    
        # Rollouts
        print("HRERE")
        
        rollout_markers = MarkerArray()
        p = 0
        t = 0
        if self.rollout_trajectory is not None:
            print("self.rollout_trajectory",np.array(self.rollout_trajectory).shape)
            for trajectory in self.rollout_trajectory:
                alpha = 0.5 ##1 #- costs[t]
                for point in trajectory:
                    # print("point shape", point.shape)
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.type = marker.SPHERE
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = alpha # global_wpnt.vx_mps / max_vx_mps
                    marker.color.r = 1.0

                    marker.id = p
                    marker.pose.position.x = point[5]
                    marker.pose.position.y = point[6]
                    marker.pose.orientation.w = 1
                    rollout_markers.markers.append(marker)
                    p += 1
                t += 1
            self.pub_rollout.publish(rollout_markers)


        return
       