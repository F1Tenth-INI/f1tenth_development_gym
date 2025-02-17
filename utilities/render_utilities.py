from utilities.Settings import Settings
import numpy as np
from utilities.state_utilities import *

from typing import Dict, Any
import numbers

# Imports depending on ROS/Gym
if(Settings.ROS_BRIDGE):
    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
else:
    if(Settings.RENDER_MODE is not None):
        try:
            from pyglet.gl import GL_POINTS
            from pyglet.gl import GL_LINES
            import pyglet.gl as gl
            from pyglet import shapes
            import pyglet
            from pyglet.gl import glLineWidth
        except:
            Settings.RENDER_MODE = None




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
        self.draw_position_history = True
        self.draw_waypoints = True
        self.draw_next_waypoints = True

        self.waypoint_visualization_color = (180, 180, 180)
        self.next_waypoint_visualization_color = (0, 127, 0)
        self.next_waypoints_alternative_visualization_color = (127, 0, 127)
        self.lidar_visualization_color = (255, 0, 255)
        self.gap_visualization_color = (0, 255, 0)
        self.mppi_visualization_color = (250, 25, 30)
        self.optimal_trajectory_visualization_color = (255, 165, 0)
        self.target_point_visualization_color = (255, 204, 0)
        self.position_history_color = (0, 204, 0)
        self.obstacle_visualization_color = (255, 0, 0)
        
        self.label_dict = {}

        self.waypoint_vertices = None
        self.next_waypoint_vertices = None
        self.next_waypoints_alternative_vertices = None
        self.lidar_vertices = None
        self.gap_vertex = None
        self.mppi_rollouts_vertices = None
        self.optimal_trajectory_vertices = None
        self.target_vertex = None
        self.obstacle_vertices = None
        self.emergency_slowdown_lines = []

        self.waypoints = None
        self.waypoints_alternative = None
        self.next_waypoints = None
        self.next_waypoints_alternative = None
        self.lidar_border_points = None
        self.rollout_trajectory = None
        self.traj_cost =None
        self.optimal_trajectory = None
        self.largest_gap_middle_point = None
        self.target_point = None
        self.car_state = None
        self.obstacles = None

        self.past_car_states_alternative = None
        self.past_car_states_alternative_vertices = None
        
        self.steering_direction = None

        self.emergency_slowdown_sprites = None

        if(Settings.ROS_BRIDGE):
            print("initialising render utilities for ROS")

            # rospy.init_node('gym_bridge_driver', anonymous=True)
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
               next_waypoints_alternative=None,
               car_state = None,
               emergency_slowdown_sprites=None,
               past_car_states_alternative=None,
               ):
        if Settings.RENDER_MODE is None: return
        
        if(lidar_points is not None): self.lidar_border_points = lidar_points
        if(rollout_trajectory is not None): self.rollout_trajectory = rollout_trajectory,
        if(traj_cost is not None): self.traj_cost = traj_cost
        if(optimal_trajectory is not None): self.optimal_trajectory = optimal_trajectory
        if(largest_gap_middle_point is not None): self.largest_gap_middle_point = largest_gap_middle_point
        if(target_point is not None): self.target_point = target_point
        if(next_waypoints is not None): self.next_waypoints = next_waypoints
        if(next_waypoints_alternative is not None): self.next_waypoints_alternative = next_waypoints_alternative
        if(car_state is not None): self.car_state = car_state
        if emergency_slowdown_sprites is not None: self.emergency_slowdown_sprites = emergency_slowdown_sprites
        if past_car_states_alternative is not None: self.past_car_states_alternative = past_car_states_alternative

        

    def update_mpc(self, rollout_trajectory, optimal_trajectory):
        self.rollout_trajectory = rollout_trajectory
        self.optimal_trajectory = optimal_trajectory

    def update_pp(self, target_point):
        self.target_point = target_point
        
    def update_obstacles(self, obstacles):
        self.obstacles = obstacles
        return
    
    
    def set_label_dict(self, dict: Dict[str, Any]) -> None:
        """
        Update the info label on the rendering. Add some custom dict and it will be displayed on the rendering.

        Args:
            dict (dict): A dictionary containing key-value pairs to render information

        Returns:
            None
        """
        for key, value in dict.items():
            
            if isinstance(value, (numbers.Real, np.float32)):
                value = round(float(value), 4)
            self.label_dict[key] = value
            
            
    
    def render(self, e = None):

        if(Settings.ROS_BRIDGE):
            self.render_ros()
        else:
            self.render_gym(e)
            

    def render_gym(self, e):

        if Settings.RENDER_INFO:
            label_text = "\n".join([f"{key}: {value}" for key, value in sorted(self.label_dict.items())])
            e.info_label.text = label_text
        
        # Waypoints
        gl.glPointSize(1)
        if self.draw_waypoints:        

            waypoints = self.waypoints
            if self.waypoints_alternative is not None:
                waypoints = np.vstack((self.waypoints, self.waypoints_alternative))


            scaled_points = RenderUtils.get_scaled_points( waypoints )
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            if self.waypoint_vertices is None: # Render only once
                self.waypoint_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                               ('c3B', self.waypoint_visualization_color * howmany))

        if self.draw_next_waypoints and self.next_waypoints_alternative is not None:
            scaled_points = RenderUtils.get_scaled_points(self.next_waypoints_alternative)
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            if self.next_waypoints_alternative_vertices is None:
                self.next_waypoints_alternative_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                               ('c3B', self.next_waypoints_alternative_visualization_color * howmany))
            else:
                self.next_waypoints_alternative_vertices.vertices = scaled_points_flat
                
                
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
        
        
        # if self.draw_position_history and self.car_state is not None:
        if self.car_state is not None:
            
                
            # Draw an arrow starting at car_state[x,y] and pointing in the direction of car_state[steering_angle]
            steering_angle = self.car_state[STEERING_ANGLE_IDX]
            yaw = self.car_state[POSE_THETA_IDX]
            arrow_length = 1.0  # You can adjust this value

            # Get starting position from car_state
            start_point_x = self.car_state[POSE_X_IDX]
            start_point_y = self.car_state[POSE_Y_IDX]

            # Calculate end points
            end_point_x = start_point_x + arrow_length * np.cos(steering_angle + yaw)
            end_point_y = start_point_y + arrow_length * np.sin(steering_angle + yaw)

            [start_point_x, start_point_y, end_point_x, end_point_y] = RenderUtils.get_scaled_points([start_point_x, start_point_y, end_point_x, end_point_y])
            if hasattr(self, 'arrow'):
                self.arrow.delete()
            glLineWidth(2)  # Set the width to 5 pixels
            self.arrow = e.batch.add(2, pyglet.gl.GL_LINES, None, 
                        ('v2f/stream', [start_point_x, start_point_y, end_point_x, end_point_y]),
                        ('c3B/static', tuple((0, 204, 0)) * 2))  # color of the arrow
              
            # Position History             
            if self.draw_position_history:
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

        if self.past_car_states_alternative is not None:

            if self.past_car_states_alternative_vertices is not None:
                self.past_car_states_alternative_vertices.delete()
            points = self.past_car_states_alternative[:, POSE_X_IDX:POSE_Y_IDX+1]
            scaled_points = RenderUtils.get_scaled_points(points)
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            self.past_car_states_alternative_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                           ('c3B', [255, 255, 0] * howmany))
    
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

        if self.target_point is not None and (Settings.CONTROLLER == 'pp' or Settings.CONTROLLER == 'stanley'):

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

        # Render the emergency slowdown boundary lines if they are available.
        # Render the emergency slowdown boundary lines if they are available.
        if self.emergency_slowdown_sprites is not None:
            # Convert the line endpoints to scaled points for rendering.
            left_line_array = np.array(self.emergency_slowdown_sprites["left_line"])  # Shape (2,2)
            right_line_array = np.array(self.emergency_slowdown_sprites["right_line"])
            stop_line_array = np.array(self.emergency_slowdown_sprites["stop_line"])
            speed_reduction_factor = self.emergency_slowdown_sprites["speed_reduction_factor"]  # Between 0.0 and 1.0

            scaled_left_line = RenderUtils.get_scaled_points(left_line_array)
            scaled_right_line = RenderUtils.get_scaled_points(right_line_array)
            scaled_stop_line = RenderUtils.get_scaled_points(stop_line_array)

            # Delete existing lines before adding new ones
            if hasattr(self, 'emergency_slowdown_lines'):
                for line in self.emergency_slowdown_lines:
                    line.delete()

            # Interpolate color based on speed_reduction_factor (1.0 = green, 0.0 = red)
            red = int((1.0 - speed_reduction_factor) * 255)
            green = int(speed_reduction_factor * 255)
            line_color = (red, green, 0)  # RGB color interpolated based on factor

            # Draw left and right boundary lines with interpolated color
            gl.glLineWidth(2)
            self.emergency_slowdown_lines = [
                e.batch.add(2, GL_LINES, None, ('v2f/stream', scaled_left_line.flatten()),
                            ('c3B', line_color * 2)),
                e.batch.add(2, GL_LINES, None, ('v2f/stream', scaled_right_line.flatten()),
                            ('c3B', line_color * 2)),
                # Draw stop line in red
                e.batch.add(2, GL_LINES, None, ('v2f/stream', scaled_stop_line.flatten()),
                            ('c3B', (255, 0, 0) * 2))
            ]

            # Delete previous label if it exists to prevent duplicates
            if hasattr(self, 'emergency_slowdown_text'):
                self.emergency_slowdown_text.delete()

            # Create a new label with the same interpolated color
            if speed_reduction_factor < 1.0:

                # Draw the speed reduction factor text near the left line.
                # Calculate the text display position from the stored "display_position"
                display_position = self.emergency_slowdown_sprites["display_position"]
                scaled_display_position = RenderUtils.get_scaled_points(display_position)
                text = f"{speed_reduction_factor:.2f}"

                self.emergency_slowdown_text = pyglet.text.Label(
                    text,
                    font_name='Arial',
                    font_size=12,
                    x=scaled_display_position[0],
                    y=scaled_display_position[1],
                    color=(line_color[0], line_color[1], line_color[2], 255),  # RGBA tuple
                    batch=e.batch
                )

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
        # if(points == [] or points is None): return np.array([])
        # print(points)
        return 50.*np.array(points)
