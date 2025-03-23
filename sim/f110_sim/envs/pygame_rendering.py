import pygame
import numpy as np
import yaml
from PIL import Image
from f110_sim.envs.collision_models import get_vertices
from utilities.Settings import Settings

CAR_LENGTH = 0.58
CAR_WIDTH = 0.31

ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1 / ZOOM_IN_FACTOR

class EnvRenderer:
    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("F1TENTH Sim")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 30)

        self.width = width
        self.height = height
        self.zoom_level = 20.0
        self.camera_offset = np.array([0.0, 0.0], dtype=np.float32)

        self.map_points = []
        self.map_surface = None
        self.scaled_map_surface = None
        self.last_scale = None
        self.poses = None

        self.dragging = False
        self.last_mouse_pos = None

    def update_map(self, map_path, map_ext):
        with open(map_path + '.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
        res = metadata['resolution']
        origin_x, origin_y = metadata['origin'][:2]

        img = Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)
        img = np.array(img)
        map_height, map_width = img.shape[:2]
        obstacle_coords = np.argwhere(img == 0)

        self.map_points = []
        for y, x in obstacle_coords:
            wx = x * res + origin_x
            wy = y * res + origin_y
            self.map_points.append(np.array([wx, wy]))

        # Pre-render the map points to a high-resolution surface in world space
        self.map_surface = pygame.Surface((map_width, map_height), pygame.SRCALPHA)
        for p in self.map_points:
            scaled = (p - np.array([origin_x, origin_y])) / res  # pixel location in map image
            pygame.draw.circle(self.map_surface, (183, 193, 222), scaled.astype(int), 1)

        self.map_origin = np.array([origin_x, origin_y])
        self.map_resolution = res
        self.map_image_shape = (map_width, map_height)
        self.last_scale = None
        self.scaled_map_surface = None

    def update_obs(self, obs):
        self.ego_idx = obs['ego_idx']
        self.poses = np.stack((obs['poses_x'], obs['poses_y'], obs['poses_theta']), axis=-1)
        self.vels = obs['linear_vels_x']
        self.lap_count = obs['lap_counts'][self.ego_idx]
        self.sim_time = obs['simulation_time']

    def draw_car(self, pose, color):
        vertices = get_vertices(pose, CAR_LENGTH, CAR_WIDTH)
        points = [self.world_to_screen(v) for v in vertices]
        pygame.draw.polygon(self.screen, color, points)

    def world_to_screen(self, point):
        screen_center = np.array([self.width / 2, self.height / 2])
        transformed = (point - self.camera_offset) * self.zoom_level + screen_center
        return transformed.astype(int)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise Exception("Rendering window was closed.")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:
                    self.zoom(1)
                elif event.button == 5:
                    self.zoom(-1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                current_mouse_pos = pygame.mouse.get_pos()
                dx, dy = np.subtract(current_mouse_pos, self.last_mouse_pos)
                self.pan(dx, dy)
                self.last_mouse_pos = current_mouse_pos

    def zoom(self, direction):
        factor = ZOOM_IN_FACTOR if direction > 0 else ZOOM_OUT_FACTOR
        new_zoom = self.zoom_level * factor
        if 0.1 < new_zoom < 500:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            world_mouse_before = (np.array([mouse_x, mouse_y]) - np.array([self.width / 2, self.height / 2])) / self.zoom_level + self.camera_offset
            self.zoom_level = new_zoom
            world_mouse_after = (np.array([mouse_x, mouse_y]) - np.array([self.width / 2, self.height / 2])) / self.zoom_level + self.camera_offset
            self.camera_offset += world_mouse_before - world_mouse_after
            self.last_scale = None  # trigger re-scaling of the map

    def pan(self, dx, dy):
        self.camera_offset -= np.array([dx, dy]) / self.zoom_level

    def render(self):
        self.handle_events()
        self.screen.fill((9, 32, 87))

        # Efficient map rendering: scale once per zoom change and reuse
        if self.map_surface:
            scale = self.zoom_level * self.map_resolution
            if scale != self.last_scale:
                scaled_size = [int(s * scale) for s in self.map_image_shape]
                self.scaled_map_surface = pygame.transform.smoothscale(self.map_surface, scaled_size)
                self.last_scale = scale

            if self.scaled_map_surface:
                map_pos_in_world = self.map_origin
                map_pos_on_screen = self.world_to_screen(map_pos_in_world)
                self.screen.blit(self.scaled_map_surface, map_pos_on_screen)

        if self.poses is not None:
            for i, pose in enumerate(self.poses):
                color = (172, 97, 185) if i == self.ego_idx else (99, 52, 94)
                self.draw_car(pose, color)

            state_text = f"Sim Time: {self.sim_time:.2f} | Lap: {self.lap_count} | x: {self.poses[0][0]:.2f}, y: {self.poses[0][1]:.2f}, yaw: {self.poses[0][2]:.2f}, v: {self.vels[0]:.2f}"
            text_surface = self.font.render(state_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, self.height - 30))

        pygame.display.flip()
        # self.clock.tick(60)

    def close(self):
        pygame.quit()

    def flip(self):
        self.render()
