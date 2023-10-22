from utilities.Settings import Settings
import numpy as np
from utilities.state_utilities import *

# Imports depending on ROS/Gym
if(Settings.ROS_BRIDGE):
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
else:
    if(Settings.RENDER_MODE is not None):
        from pyglet.gl import GL_POINTS
        import pyglet.gl as gl
        from pyglet import shapes



'''
HOW TO USE:

1. Use render utilities to render data from your planner class:

# Import 
from utilities.waypoint_utils import WaypointUtils

#Initialize
self.Render = RenderUtils()

# If there are waypoints to be rendered initially, pass them after initialization
self.Render.waypoints = self.waypoint_utils.waypoint_positions 

# Implement the render function inside your planner class (it will be called from the Gym at every step)
def render(self, e):
    self.Render.render(e)

# At every step update the data to be rendered
self.Render.update(
    lidar_points=self.lidar_points,
    next_waypoints= self.waypoint_utils.next_waypoint_positions,
    car_state = s
    [... more arguments to be implemented ...]
)

'''
class RenderUtils:
    def __init__(self):

        self.draw_lidar_data = True
        self.draw_position_history = False
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
        self.obstacle_visualization_color = (255, 0, 0)

        self.waypoint_vertices = None
        self.next_waypoint_vertices = None
        self.lidar_vertices = None
        self.gap_vertex = None
        self.mppi_rollouts_vertices = None
        self.optimal_trajectory_vertices = None
        self.target_vertex = None
        self.obstacle_vertices = None

        self.waypoints = None
        self.next_waypoints = None
        self.lidar_border_points = None
        self.rollout_trajectory = None
        self.traj_cost =None
        self.optimal_trajectory = None
        self.largest_gap_middle_point = None
        self.target_point = None
        self.car_state = None
        self.obstacles = None
        
        if(Settings.ROS_BRIDGE):
            print("initialising render utilities for ROS")

            rospy.init_node('gym_bridge_driver', anonymous=True)
            self.pub_rollout = rospy.Publisher('mppi/rollout', MarkerArray, queue_size=1)
            self.pub_target_point = rospy.Publisher('/pp/lookahead', Marker, queue_size=1)
        

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
        
        
        if(lidar_points is not None): self.lidar_border_points = lidar_points
        if(rollout_trajectory is not None): self.rollout_trajectory = rollout_trajectory,
        if(traj_cost is not None): self.traj_cost = traj_cost
        if(optimal_trajectory is not None): self.optimal_trajectory = optimal_trajectory
        if(largest_gap_middle_point is not None): self.largest_gap_middle_point = largest_gap_middle_point
        if(target_point is not None): self.target_point = target_point
        if(next_waypoints is not None): self.next_waypoints = next_waypoints
        if(car_state is not None): self.car_state = car_state
        
        if(self.next_waypoints == []):self.next_waypoints = None

    def update_mpc(self, rollout_trajectory, optimal_trajectory):
        self.rollout_trajectory = rollout_trajectory
        self.optimal_trajectory = optimal_trajectory

    def update_pp(self, target_point):
        self.target_point = target_point
        
    def update_obstacles(self, obstacles):
        self.obstacles = obstacles
        return
    
    def render(self, e = None):

        if(Settings.ROS_BRIDGE):
            self.render_ros()
        else:
            self.render_gym(e)
            

    def render_gym(self, e):
        
        # Waypoints
        gl.glPointSize(1)
        if self.draw_waypoints:        
            scaled_points = RenderUtils.get_scaled_points( self.waypoints )
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            if self.waypoint_vertices is None: # Render only once
                self.waypoint_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                               ('c3B', self.waypoint_visualization_color * howmany))
   
                
        if self.draw_next_waypoints and self.next_waypoints is not None:
            scaled_points = RenderUtils.get_scaled_points(self.next_waypoints)
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            if self.next_waypoint_vertices is None:
                self.next_waypoint_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                               ('c3B', self.next_waypoint_visualization_color * howmany))
            else:
                self.next_waypoint_vertices.vertices = scaled_points_flat
            
        gl.glPointSize(3)
        if self.draw_position_history and self.car_state is not None:
            points = np.array([self.car_state[POSE_X_IDX], self.car_state[POSE_Y_IDX]])
            speed = self.car_state[LINEAR_VEL_X_IDX]
            scaled_points = RenderUtils.get_scaled_points(points)
            # Color [r,g,b] values (int) between 0 and 255
            color = [
                max(min(int(10 * speed), 255), 0),
                min(max(int(255 - 10 * speed), 0), 255),
                0
            ]
            e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[0], scaled_points[1], 0.]),
                        ('c3B',color))


        if self.draw_lidar_data: 
            if self.lidar_border_points is not None:
                scaled_points = RenderUtils.get_scaled_points(self.lidar_border_points)
                howmany = scaled_points.shape[0]
                scaled_points_flat = scaled_points.flatten()
                if self.lidar_vertices is None:
                    self.lidar_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                                ('c3B', self.lidar_visualization_color * howmany))
                else:
                    self.lidar_vertices.vertices = scaled_points_flat

        if self.largest_gap_middle_point is not None:
            scaled_point_gap = RenderUtils.get_scaled_points(self.largest_gap_middle_point)
            scaled_points_gap_flat = scaled_point_gap.flatten()
            self.gap_vertex = shapes.Circle(scaled_points_gap_flat[0], scaled_points_gap_flat[1], 5, color=self.gap_visualization_color, batch=e.batch)


        if self.rollout_trajectory is not None:
            num_trajectories_to_plot = np.minimum(Settings.NUM_TRAJECTORIES_TO_PLOT, self.rollout_trajectory.shape[0])
            trajectory_points = self.rollout_trajectory[:num_trajectories_to_plot, :, POSE_X_IDX:POSE_Y_IDX+1]

            scaled_trajectory_points = RenderUtils.get_scaled_points(trajectory_points)

            howmany_mppi = scaled_trajectory_points.shape[0]*scaled_trajectory_points.shape[1]
            scaled_trajectory_points_flat = scaled_trajectory_points.flatten()

            if self.mppi_rollouts_vertices is None:
                self.mppi_rollouts_vertices = e.batch.add(howmany_mppi, GL_POINTS, None, ('v2f/stream', scaled_trajectory_points_flat),
                                               ('c3B', self.mppi_visualization_color * howmany_mppi))
            else:
                self.mppi_rollouts_vertices.vertices = scaled_trajectory_points_flat

        if self.optimal_trajectory is not None:
            optimal_trajectory_points = self.optimal_trajectory[:, :, POSE_X_IDX:POSE_Y_IDX+1]

            scaled_optimal_trajectory_points = RenderUtils.get_scaled_points(optimal_trajectory_points)

            howmany_mppi_optimal = scaled_optimal_trajectory_points.shape[0]*scaled_optimal_trajectory_points.shape[1]
            scaled_optimal_trajectory_points_flat = scaled_optimal_trajectory_points.flatten()

            if self.optimal_trajectory_vertices is None:
                self.optimal_trajectory_vertices = e.batch.add(howmany_mppi_optimal, GL_POINTS, None, ('v2f/stream', scaled_optimal_trajectory_points_flat),
                                               ('c3B', self.optimal_trajectory_visualization_color * howmany_mppi_optimal))
            else:
                self.optimal_trajectory_vertices.vertices = scaled_optimal_trajectory_points_flat

        if self.target_point is not None and (Settings.FOLLOW_RANDOM_TARGETS or Settings.CONTROLLER == 'pp' or Settings.CONTROLLER == 'stanley'):

            scaled_target_point = RenderUtils.get_scaled_points(self.target_point)
            scaled_target_point_flat = scaled_target_point.flatten()
            self.target_vertex = shapes.Circle(scaled_target_point_flat[0], scaled_target_point_flat[1], 10, color=self.target_point_visualization_color, batch=e.batch)
            
        
        if self.obstacles is not None:
            scaled_points = RenderUtils.get_scaled_points(self.obstacles)
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            
            # if self.obstacle_vertices is None:
            self.obstacle_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                            ('c3B', self.obstacle_visualization_color * howmany))
            # else:
            #     self.obstacle_vertices.vertices = scaled_points_flat

    
    def render_ros(self):
        #  Publish data for RVIZ
        
        # PP lookahead point
        if(self.target_point is not None):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = marker.SPHERE
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0 # global_wpnt.vx_mps / max_vx_mps
            marker.color.r = 1.0
            marker.color.g = 1.0

            marker.id = 1444444
            marker.pose.position.x = self.target_point[0]
            marker.pose.position.y = self.target_point[1]
            
            marker.pose.orientation.w = 1
            self.pub_target_point.publish(marker)
        
        # Rollouts
        if(self.rollout_trajectory is None): return
        rollout_markers = MarkerArray()        
        p = 0
        t = 0
        t = 0
        
        t = 0        
        
        rollout_trajectory = np.array(self.rollout_trajectory)
        rollout_points = rollout_trajectory[:,:,5:7]
        rollout_points = rollout_points[:10]
        
        for trajectory in rollout_points:
            alpha = 1 - 0.5
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
                marker.pose.position.x = point[0]
                marker.pose.position.y = point[1]
                marker.pose.orientation.w = 1
                rollout_markers.markers.append(marker)
                p += 1
            t += 1
            


        # optimal trajectory
        if(self.optimal_trajectory is not None):
            optimal_trajectory = self.optimal_trajectory[0]
            for point in optimal_trajectory:
            
                    marker = Marker()
                    marker.header.frame_id = 'map'
                    marker.type = marker.SPHERE
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = alpha # global_wpnt.vx_mps / max_vx_mps
                    marker.color.r = 1.0
                    marker.color.b = 1.0

                    marker.id = p
                    marker.pose.position.x = point[5]
                    marker.pose.position.y = point[6]
                    marker.pose.orientation.w = 1
                    rollout_markers.markers.append(marker)
                    p += 1
                
        self.pub_rollout.publish(rollout_markers)
        return
       
       
    @staticmethod
    def get_scaled_points(points):
        if(points == [] or points is None): return np.array([])
        return 50.*np.array(points)
