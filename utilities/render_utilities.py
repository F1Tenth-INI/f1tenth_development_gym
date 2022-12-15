from pyglet.gl import GL_POINTS
import pyglet.gl as gl
from pyglet import shapes
import numpy as np
from utilities.state_utilities import *


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
        
        
        self.lidar_border_points = lidar_points
        self.rollout_trajectory, self.traj_cost = rollout_trajectory, traj_cost
        self.optimal_trajectory = optimal_trajectory
        self.largest_gap_middle_point = largest_gap_middle_point
        self.target_point = target_point
        self.next_waypoints = next_waypoints
        self.car_state = car_state
        
        if(self.next_waypoints == []):self.next_waypoints = None

    def render(self, e):
        
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
            # color = [
            #     min(int(10 * speed),255) ,
            #     min(max(int(255- 10 * speed), 0), 255), 
            #     0
            # ]
            # e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[0], scaled_points[1], 0.]),
            #             ('c3B',color))


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

        if self.target_point is not None and Settings.FOLLOW_RANDOM_TARGETS:

            scaled_target_point = RenderUtils.get_scaled_points(self.target_point)
            scaled_target_point_flat = scaled_target_point.flatten()
            self.target_vertex = shapes.Circle(scaled_target_point_flat[0], scaled_target_point_flat[1], 10, color=self.target_point_visualization_color, batch=e.batch)

    @staticmethod
    def get_scaled_points(points):
        if(points == [] or points is None): return np.array([])
        return 50.*np.array(points)
