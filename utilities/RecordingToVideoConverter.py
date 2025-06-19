import pandas as pd
import cv2
import numpy as np
import math
import yaml
from tqdm import trange
from io import StringIO
import os
class RecordingToVideoConverter:
    """
    Converts a CSV recording of a car's trajectory into a video.
    The video shows the car's path on a map, with the car represented as a rectangle.
    """

    def __init__(self, csv_file_path, recording_name, map_name):
        self.csv_file = os.path.join(csv_file_path, recording_name + ".csv")
        self.map_name = map_name
                
        # === Config ===
        self.video_output_path = os.path.join(csv_file_path, recording_name+"_data")
        self.video_output_file = os.path.join(self.video_output_path,  "recording.mp4")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.video_output_path):
            os.makedirs(self.video_output_path)
        
        # WIDTH, HEIGHT = 800, 800
        self.SCALE = 20  # 1 sim unit = 20 pixels
        FPS = 100  # Based on timestep = 0.01s

        self.map_image = f"utilities/maps/{map_name}/{map_name}.png"  # Path to the map image
        self.map_yaml = f"utilities/maps/{map_name}/{map_name}.yaml"
        self.map_img = cv2.imread(self.map_image)
        self.HEIGHT, self.WIDTH = self.map_img.shape[:2]
        
                
        # Read map yaml
        with open(self.map_yaml, "r") as f:
            map_config = yaml.safe_load(f)

        self.MAP_ORIGIN_X, self.MAP_ORIGIN_Y, _ = map_config["origin"]
        self.MAP_RESOLUTION = map_config["resolution"]
        
        self.trail_img = np.zeros_like(self.map_img)  # Holds the trail permanently
        
        
        

        # === Load and clean CSV ===
        with open(self.csv_file, "r") as f:
            lines = f.readlines()

        # Find where the actual data starts (skip all `#` comments)
        for i, line in enumerate(lines):
            if not line.startswith("#"):
                header_line = lines[i]
                data_lines = lines[i+1:]
                break
            
        




        self.df = pd.read_csv(StringIO("".join([header_line] + data_lines)))

        # Cut datframe, only use first 1000 rows
        self.df = self.df.iloc[:10000]

        # === Initialize video writer ===
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'avc1' for H.264
        self.video_writer = cv2.VideoWriter(self.video_output_file, fourcc, FPS, (self.WIDTH, self.HEIGHT))

    # === World to screen transform ===
    def world_to_screen(self, x, y):
        screen_x = int(self.WIDTH / 2 + x * self.SCALE)
        screen_y = int(self.HEIGHT / 2 - y * self.SCALE)
        return screen_x, screen_y

    def world_to_image(self, x, y, flip_y=True):
        px = int((x - self.MAP_ORIGIN_X) / self.MAP_RESOLUTION)
        py = int((y - self.MAP_ORIGIN_Y) / self.MAP_RESOLUTION)
        if flip_y:
            py = self.map_img.shape[0] - py
        return px, py



    # === Draw car as rectangle ===
    def draw_car(self, img, x, y, theta, car_length=0.58, car_width=0.31):
        # Convert to image coordinates (no shift needed for center alignment)
        cx, cy = self.world_to_image(x, y)

        # Car size in pixels
        length_px = int(car_length / self.MAP_RESOLUTION)
        width_px = int(car_width / self.MAP_RESOLUTION)

        # Orientation
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Car rectangle corners (relative to center)
        half_l = length_px / 2
        half_w = width_px / 2
        corners = np.array([
            [ half_l,  half_w],
            [ half_l, -half_w],
            [-half_l, -half_w],
            [-half_l,  half_w]
        ])

        # Rotate and translate to image coordinates
        rotated = []
        for dx, dy in corners:
            px = int(cx + dx * cos_theta - dy * sin_theta)
            py = int(cy - dx * sin_theta - dy * cos_theta)
            rotated.append((px, py))

        # Convert to numpy array for OpenCV
        rotated = np.array(rotated, dtype=np.int32)

        # Draw the car as a filled polygon with anti-aliasing
        cv2.fillPoly(img, [rotated], (0, 0, 255))
            

    def velocity_to_color(self, v, v_min=-0.0, v_max=10.0):
        """ Map velocity to a BGR color from blue (slow) to red (fast). """
        v = max(v_min, min(v, v_max))  # clamp
        t = (v - v_min) / (v_max - v_min)
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 50
        return (b, g, r)


    def render_video(self):
        # === Render each frame ===
        for i in trange(len(self.df), desc="Rendering frames"):

            row = self.df.iloc[i]

            # frame = map_img.copy()
            frame = cv2.addWeighted(self.map_img, 1.0, self.trail_img, 1.0, 0)
            # frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
            
            # Only draw the new segment to self.trail_img
            if i > 0:
                prev_row = self.df.iloc[i - 1]
                curr_row = row

                x1, y1 = self.world_to_image(prev_row["pose_x"], prev_row["pose_y"])
                x2, y2 = self.world_to_image(curr_row["pose_x"], curr_row["pose_y"])

                v = curr_row["linear_vel_x"]
                color = self.velocity_to_color(v)
                cv2.line(self.trail_img, (x1, y1), (x2, y2), color, 1)


            # Car state
            x, y, theta = row["pose_x"], row["pose_y"], row["pose_theta"]
            self.draw_car(frame, x, y, theta)
            

            # Optional: draw waypoints
            for i in range(30):  # Assuming 30 waypoints in WYPT_X_*, WYPT_Y_*
                wx_col = f"WYPT_X_{i:02d}"
                wy_col = f"WYPT_Y_{i:02d}"
                if wx_col in row and wy_col in row:
                    wx, wy = row[wx_col], row[wy_col]
                    wxs, wys = self.world_to_image(wx, wy)
                    cv2.circle(frame, (wxs, wys), 2, (0, 255, 0), -1)

            # Write frame
            self.video_writer.write(frame)

        self.video_writer.release()
        print(f"Video saved to {self.video_output_file}")



if __name__ == "__main__":
    
    
    # Example usage

    converter = RecordingToVideoConverter("ExperimentRecordings","recording", "RCA1")
    converter.render_video()