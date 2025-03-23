import pygame
import os
import numpy as np
import yaml
from PIL import Image
from utilities.Settings import Settings
from f110_sim.envs.collision_models import get_vertices

# Zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

# Vehicle shape constants
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

class PygameRenderer:
    """ A Pygame-based renderer for the F1TENTH simulator, properly handling map origin and scaling. """

    def __init__(self, width=1000, height=800, map_path = '', map_name = 'defaults'):
        pygame.init()
        self.width = width
        self.height = height
        
        self.map_path = map_path
        self.map_name = map_name
        
        self.yaml_path = os.path.join(Settings.MAP_PATH, Settings.MAP_NAME + ".yaml" )  
        self.png_path = os.path.join(Settings.MAP_PATH, Settings.MAP_NAME + ".png")  
        
        self.scale = 50  # Default scale (adjusted dynamically)
        self.running = True

        # Initialize Pygame window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("F1TENTH Simulator - Pygame Renderer")

        # Load map metadata
        self.map_img = None
        self.origin = [0, 0]
        self.resolution = 0.05  # Default resolution (meters per pixel)

        self.load_map_metadata()

        # Car settings
        self.car_size = (int(CAR_WIDTH * self.scale), int(CAR_LENGTH * self.scale))  # Scale car size to pixels
        self.car_color = [(255, 0, 0), (99, 52, 94)]  # Different colors for ego and other cars

        # State tracking
        self.poses = None
        self.num_agents = 0

        # Camera properties
        self.zoom_level = 1.2
        self.pan_offset = [0, 0]

        # Font for UI
        self.font = pygame.font.SysFont(None, 36)

    def load_map_metadata(self):
        """ Loads the map metadata including image file, origin, and resolution. """
        
        if self.yaml_path and os.path.exists(self.yaml_path):
            with open(self.yaml_path, "r") as file:
                map_data = yaml.safe_load(file)
            
            image_filename = map_data.get("image", "")
            map_dir = os.path.dirname(self.yaml_path)
            map_path = os.path.join(map_dir, image_filename)
            
            self.resolution = map_data.get("resolution", 0.05)
            self.origin = map_data.get("origin", [0, 0, 0])[:2]  # Ignore theta
        else:
            print(f"⚠️ Map metadata file not found: {self.yaml_path}")
            return

        # Load map image
        
        if os.path.exists(self.png_path):
            self.map_img = pygame.image.load(self.png_path)
            img_width, img_height = self.map_img.get_size()

            # Compute scale dynamically
            self.scale = self.width / img_width  

            self.map_img = pygame.transform.scale(
                self.map_img, (self.width, self.height)
            )
            print(f"Map Image Size: {self.map_img.get_size()} pixels")
            print(f"Loaded map from {map_path} with resolution {self.resolution} and origin {self.origin}")
        else:
            print(f"⚠️ Map file not found: {map_path}")

    def world_to_screen(self, x, y):
        """ Convert world coordinates (meters) to screen coordinates (pixels), matching Pyglet's transformation. """
        
        # Get map size in pixels
        img_width, img_height = self.map_img.get_size()
        
        # Convert world coordinates to pixel positions relative to the origin
        pixel_x = (x - self.origin[0]) / self.resolution
        pixel_y = (y - self.origin[1]) / self.resolution

        # Scale positions similar to Pyglet (multiply by 50)
        screen_x = int(pixel_x * self.scale)  
        screen_y = int(img_height - (pixel_y * self.scale))  # Flip y-axis

        return screen_x, screen_y



    def render(self, obs):
        """ Render the environment including cars, map, and text overlays. """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    self.zoom_level *= ZOOM_IN_FACTOR
                    self.scale *= ZOOM_IN_FACTOR
                elif event.button == 5:  # Scroll down
                    self.zoom_level *= ZOOM_OUT_FACTOR
                    self.scale *= ZOOM_OUT_FACTOR
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    Settings.CAMERA_AUTO_FOLLOW = not Settings.CAMERA_AUTO_FOLLOW
                elif event.key == pygame.K_LEFT:
                    self.pan_offset[0] -= 20
                elif event.key == pygame.K_RIGHT:
                    self.pan_offset[0] += 20
                elif event.key == pygame.K_UP:
                    self.pan_offset[1] -= 20
                elif event.key == pygame.K_DOWN:
                    self.pan_offset[1] += 20

        # Clear screen
        self.screen.fill((0, 0, 0))

        # Draw map
        if self.map_img:
            self.screen.blit(self.map_img, (0, 0))

        # Extract car states
        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        poses_theta = obs['poses_theta']
        self.num_agents = len(poses_x)

        # Draw cars (after the map)
        
        # DEBUG: Draw a simple red circle instead of the car
        for i in range(self.num_agents):
            car_x, car_y, car_theta = poses_x[i], poses_y[i], poses_theta[i]
            car_pos = self.world_to_screen(car_x, car_y)

            # Use get_vertices() to match Pyglet's corner calculations
            car_corners = get_vertices(np.array([car_x, car_y, car_theta]), CAR_LENGTH, CAR_WIDTH)
            screen_corners = [self.world_to_screen(v[0], v[1]) for v in car_corners]

            print(f"Pygame Renderer: Car {i} Center: {car_pos}, Corners: {screen_corners}")

            # Draw center dot
            pygame.draw.circle(self.screen, (255, 0, 0), car_pos, 5)

            # Draw corner dots
            for corner in screen_corners:
                pygame.draw.circle(self.screen, (0, 255, 0), corner, 3)  # Green dots for corners

        # Display state info
        state_text = f"State: x={poses_x[0]:.2f}, y={poses_y[0]:.2f}, psi={poses_theta[0]:.2f}, v_x={obs['linear_vels_x'][0]:.2f}"
        # lap_text = f"{Settings.STOP_TIMER_AFTER_N_LAPS}-Lap Time: {obs['lap_times'][0]:.2f}, Ego Lap Count: {obs['lap_counts'][obs['ego_idx']]:.0f}"
        lap_text = "test"
        self.draw_text(state_text, (20, 20))
        self.draw_text(lap_text, (20, 60))

        # Update display
        pygame.display.flip()

    def draw_text(self, text, position):
        """ Draws text on the screen at the given position. """
        text_surface = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(text_surface, position)

    def close(self):
        """ Clean up resources and close the renderer. """
        pygame.quit()
