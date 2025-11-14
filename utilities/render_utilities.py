from utilities.Settings import Settings
import numpy as np
from utilities.state_utilities import *

from typing import Dict, Any, Optional
import numbers

# Imports depending on ROS/Gym
if(Settings.ROS_BRIDGE):
    pass
else:
    if(Settings.RENDER_MODE is not None):
        try:
            from pyglet.gl import GL_POINTS, GL_LINES
            from pyglet.gl import glLineWidth, glPointSize
            import pyglet.gl as gl
            from pyglet import shapes
            import pyglet
        except ImportError as e:
            print("Pyglet is not installed. Please install it using 'pip install pyglet'.")
            print(f"ImportError: {e}")
   
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

if not Settings.ROS_BRIDGE and Settings.RENDER_MODE is not None:
    class PointSizeGroup(pyglet.graphics.Group):
        def __init__(self, point_size, parent=None):
            super().__init__(parent)
            self.point_size = point_size  # Desired point size for this group.

        def set_state(self):
            # Set the OpenGL point size to the custom value when drawing this group.
            glPointSize(self.point_size)

        def unset_state(self):
            # Revert the point size back to the default (or a previous value) after drawing.
            glPointSize(1)

else:
    class PointSizeGroup:
        def __init__(self, point_size):
            pass
        def set_state(self):
            pass
        def unset_state(self):
            pass

class RenderUtils:
    def __init__(self):

        self.draw_lidar_data = True
        self.draw_position_history = False
        self.draw_waypoints = False
        self.draw_next_waypoints = True

        self.draw_gt_history = True
        self.draw_prior_history = True
        self.draw_prior_full_history = True

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
        self.track_border_visualization_color = (255, 0, 0)

        self.gt_history_color = (0, 128, 255)      # blue-ish
        self.prior_history_color = (255, 255, 255)  # white
        self.prior_full_history_color = (0, 255, 255)  # cyan
        
        self.label_dict = {}
        self.waypoints: Optional[np.ndarray] = None
        self.waypoints_alternative: Optional[np.ndarray] = None
        self.waypoints_full: Optional[np.ndarray] = None
        self.next_waypoints: Optional[np.ndarray] = None
        self.next_waypoints_alternative: Optional[np.ndarray] = None
        self.lidar_border_points: Optional[np.ndarray] = None
        self.rollout_trajectory: Optional[np.ndarray] = None
        self.traj_cost =None
        self.optimal_trajectory = None
        self.largest_gap_middle_point = None
        self.target_point = None
        self.car_state = None
        self.obstacles = None
        self.track_border_points: Optional[np.ndarray] = None
        # cache PointSizeGroup instances to avoid creating new groups each frame
        self._point_size_groups = {}

        self.past_car_states_alternative = None
        self.past_car_states_alternative_vertices = None
        self.past_car_states_gt = None
        self.past_car_states_gt_vertices = None
        self.past_car_states_prior = None
        self.past_car_states_prior_vertices = None
        self.past_car_states_prior_full = None
        self.past_car_states_prior_full_vertices = None
        
        self.reset()

        if(Settings.ROS_BRIDGE):
           pass


    def reset(self):
        self.waypoint_vertices = None
        self.next_waypoint_vertices = None
        self.next_waypoints_alternative_vertices = None
        self.gap_vertex = None
        self.mppi_rollouts_vertices = None
        self.optimal_trajectory_vertices = None
        self.target_vertex = None
        self.obstacle_vertices = None
        self.emergency_slowdown_lines = []
        self.lidar_vertices = None
        self.track_border_vertices = None

        self.past_car_states_alternative = None
        self.past_car_states_alternative_vertices = None
        self.steering_direction = None
        self.emergency_slowdown_sprites = None

        # Delete position history vertices if they exist
        if hasattr(self, 'position_history_vertices') and self.position_history_vertices is not None:
            self.position_history_vertices.delete()
            self.position_history_vertices = None
        # Clear position history points
        self.position_history_points = []
        # Delete lidar vertices if they exist
        if hasattr(self, 'lidar_vertices') and self.lidar_vertices is not None:
            self.lidar_vertices.delete()
            self.lidar_vertices = None
        # Clear lidar points
        self.lidar_border_points = None
        # Clear track border points
        self.track_border_points = None


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
               car_state=None,
               emergency_slowdown_sprites=None,
               past_car_states_alternative=None,
               track_border_points=None,
                gt_past_car_states=None,
               prior_past_car_states=None,
               prior_full_past_car_states=None,
               ):
    
        if Settings.RENDER_MODE is None:
            return

        if lidar_points is not None:
            self.lidar_border_points = lidar_points
        if rollout_trajectory is not None:
            self.rollout_trajectory = rollout_trajectory
        if traj_cost is not None:
            self.traj_cost = traj_cost
        if optimal_trajectory is not None:
            self.optimal_trajectory = optimal_trajectory
        if largest_gap_middle_point is not None:
            self.largest_gap_middle_point = largest_gap_middle_point
        if target_point is not None:
            self.target_point = target_point
        if next_waypoints is not None:
            self.next_waypoints = next_waypoints
        if next_waypoints_alternative is not None:
            self.next_waypoints_alternative = next_waypoints_alternative
        if car_state is not None:
            self.car_state = car_state
        if emergency_slowdown_sprites is not None:
            self.emergency_slowdown_sprites = emergency_slowdown_sprites
        if past_car_states_alternative is not None:
            self.past_car_states_alternative = past_car_states_alternative
        if track_border_points is not None:
            self.track_border_points = track_border_points

   
        if Settings.RENDER_MODE is None: return
        

        if gt_past_car_states is not None: self.past_car_states_gt = gt_past_car_states
        if prior_past_car_states is not None: self.past_car_states_prior = prior_past_car_states
        if prior_full_past_car_states is not None: self.past_car_states_prior_full = prior_full_past_car_states

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

        # small normalizer to ensure point arrays are (N,2)
        def _normalize_points(points):
            if points is None:
                return None
            pts = np.array(points)
            if pts.size == 0:
                return None
            if pts.ndim == 1:
                if pts.size == 2:
                    return pts.reshape(1, 2)
                elif pts.size % 2 == 0:
                    return pts.reshape(-1, 2)
                else:
                    return None
            # pts.ndim >= 2
            if pts.shape[-1] == 2:
                return pts.reshape(-1, 2)
            total = pts.size
            if total % 2 == 0:
                return pts.reshape(-1, 2)
            return None

        # Note: reverted to simple per-set rendering blocks (no centralized helper)
        # This keeps behavior consistent with previous code and reduces unexpected
        # interactions from the helper.

        if self.draw_next_waypoints and self.next_waypoints_alternative is not None:
            pts = _normalize_points(self.next_waypoints_alternative)
            if pts is not None:
                scaled_points = RenderUtils.get_scaled_points(pts)
                howmany = scaled_points.shape[0]
                scaled_points_flat = scaled_points.flatten()
                if self.next_waypoints_alternative_vertices is None:
                    self.next_waypoints_alternative_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                                   ('c3B', self.next_waypoints_alternative_visualization_color * howmany))
                else:
                    self.next_waypoints_alternative_vertices.vertices = scaled_points_flat
                
                
        if self.draw_next_waypoints and self.next_waypoints is not None:
            pts = _normalize_points(self.next_waypoints)
            if pts is not None:
                scaled_points = RenderUtils.get_scaled_points(pts)
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
                # Initialize if not present
                if not hasattr(self, 'position_history_points') or self.position_history_points is None:
                    self.position_history_points = []
                # Add current position to history
                self.position_history_points.append([
                    self.car_state[POSE_X_IDX],
                    self.car_state[POSE_Y_IDX],
                    self.car_state[LINEAR_VEL_X_IDX]
                ])
                # Prepare arrays for rendering
                points_arr = np.array([[p[0], p[1]] for p in self.position_history_points])
                speeds_arr = np.array([p[2] for p in self.position_history_points])
                scaled_points = RenderUtils.get_scaled_points(points_arr)
                # Color for each point
                colors = [
                    [max(min(int(10 * s), 255), 0), min(max(int(255 - 10 * s), 0), 255), 0]
                    for s in speeds_arr
                ]
                colors_flat = [c for color in colors for c in color]
                # Delete previous position history vertices if they exist
                if hasattr(self, 'position_history_vertices') and self.position_history_vertices is not None:
                    self.position_history_vertices.delete()
                # Add all position history points as GL_POINTS
                howmany = scaled_points.shape[0]
                scaled_points_flat = [coord for pt in scaled_points for coord in (pt[0], pt[1], 0.)]
                self.position_history_vertices = e.batch.add(howmany, GL_POINTS, None, ('v3f/stream', scaled_points_flat),
                            ('c3B', colors_flat))

        if self.past_car_states_alternative is not None:

            if self.past_car_states_alternative_vertices is not None:
                self.past_car_states_alternative_vertices.delete()
            points = self.past_car_states_alternative[:, POSE_X_IDX:POSE_Y_IDX+1]
            scaled_points = RenderUtils.get_scaled_points(points)
            howmany = scaled_points.shape[0]
            scaled_points_flat = scaled_points.flatten()
            point_size_group = PointSizeGroup(4)
            self.past_car_states_alternative_vertices = e.batch.add(howmany, GL_POINTS, point_size_group, ('v2f/stream', scaled_points_flat),
                                           ('c3B', [255, 255, 0] * howmany))

        # NEW: Ground-truth past overlay (if enabled)
        if self.draw_gt_history and (self.past_car_states_gt is not None):
            if self.past_car_states_gt_vertices is not None:
                self.past_car_states_gt_vertices.delete()
            gt_pts = self.past_car_states_gt[:, POSE_X_IDX:POSE_Y_IDX+1]
            scaled_gt = RenderUtils.get_scaled_points(gt_pts)
            n_gt = scaled_gt.shape[0]
            self.past_car_states_gt_vertices = e.batch.add(
                n_gt, GL_POINTS, PointSizeGroup(4),
                ('v2f/stream', scaled_gt.flatten()),
                ('c3B', self.gt_history_color * n_gt)
            )

        # NEW: Kinematic prior past overlay (if enabled)
        if self.draw_prior_history and (self.past_car_states_prior is not None):
            if self.past_car_states_prior_vertices is not None:
                self.past_car_states_prior_vertices.delete()
            pr_pts = self.past_car_states_prior[:, POSE_X_IDX:POSE_Y_IDX+1]
            scaled_pr = RenderUtils.get_scaled_points(pr_pts)
            n_pr = scaled_pr.shape[0]
            self.past_car_states_prior_vertices = e.batch.add(
                n_pr, GL_POINTS, PointSizeGroup(4),
                ('v2f/stream', scaled_pr.flatten()),
                ('c3B', self.prior_history_color * n_pr)
            )

        # NEW: Whole-horizon anchored prior (cyan), controller cadence, oldest→newest
        if self.draw_prior_full_history and (self.past_car_states_prior_full is not None):
            if self.past_car_states_prior_full_vertices is not None:
                self.past_car_states_prior_full_vertices.delete()
            prf_pts = self.past_car_states_prior_full[:, POSE_X_IDX:POSE_Y_IDX+1]
            scaled_prf = RenderUtils.get_scaled_points(prf_pts)
            n_prf = scaled_prf.shape[0]
            self.past_car_states_prior_full_vertices = e.batch.add(
                n_prf, GL_POINTS, PointSizeGroup(4),
                ('v2f/stream', scaled_prf.flatten()),
                ('c3B', self.prior_full_history_color * n_prf)
            )

    
        if self.draw_lidar_data:
            pts = _normalize_points(self.lidar_border_points)
            if pts is not None:
                scaled_points = RenderUtils.get_scaled_points(pts)
                howmany = scaled_points.shape[0]
                scaled_points_flat = scaled_points.flatten()
                if self.lidar_vertices is None:
                    self.lidar_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                                ('c3B', self.lidar_visualization_color * howmany))
                else:
                    try:
                        self.lidar_vertices.vertices = scaled_points_flat
                    except Exception:
                        # recreate if update fails
                        try:
                            self.lidar_vertices.delete()
                        except Exception:
                            pass
                        self.lidar_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat), ('c3B', self.lidar_visualization_color * howmany))

        # Render global track border points (new feature)
        if self.track_border_points is not None:
            pts = _normalize_points(self.track_border_points)
            if pts is not None:
                scaled_points = RenderUtils.get_scaled_points(pts)
                howmany = scaled_points.shape[0]
                scaled_points_flat = scaled_points.flatten()
                if self.track_border_vertices is None:
                    self.track_border_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat),
                                                             ('c3B', self.track_border_visualization_color * howmany))
                else:
                    try:
                        self.track_border_vertices.vertices = scaled_points_flat
                    except Exception:
                        try:
                            self.track_border_vertices.delete()
                        except Exception:
                            pass
                        self.track_border_vertices = e.batch.add(howmany, GL_POINTS, None, ('v2f/stream', scaled_points_flat), ('c3B', self.track_border_visualization_color * howmany))

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

   
       
       
    @staticmethod
    def get_scaled_points(points):
        # if(points == [] or points is None): return np.array([])
        # print(points)
        return 50.*np.array(points)
