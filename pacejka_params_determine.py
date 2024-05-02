import math
import csv
from PIL import Image, ImageDraw
import yaml

def generate_spiral_waypoints(num_waypoints, radius, spacing, filename, direction, starting_position=(0.0, 0.0)):
    waypoints = []
    angle = 0.0
    if direction != 1:
        starting_position = (starting_position[0] + radius, starting_position[1])

    for i in range(num_waypoints):
        s_m = i * spacing
        x_m = starting_position[0] + radius * angle * math.cos(angle)
        y_m = starting_position[1] + radius * angle * math.sin(angle)
        
        psi_rad = angle
        kappa_radpm = 0.02 * 60.0
        vx_mps = 10.0
        ax_mps2 = 0.0
        
        waypoints.append((s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps, ax_mps2))
        angle += 0.02
        

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['# s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'vx_mps', 'ax_mps2'])
            for waypoint in waypoints:
                writer.writerow(waypoint)


# create empty map with a square in the middle black and white
def create_map(resolution, size, path, map):
    image = Image.new("L", resolution, "black")

    # Calculate the coordinates for the square
    left = (resolution[0] - size) // 2
    top = (resolution[1] - size) // 2
    right = left + size
    bottom = top + size

    # Draw the square on the image
    draw = ImageDraw.Draw(image)
    
    # Fill out the outer rectangle
    draw.rectangle((left, top, right, bottom), fill="white")

    # Save the image as PNG
    image.save(path + map)

# Parameters
path = "utilities/maps/test/"
map = "TEST.png"
resolution = (500,500)
size = 350

direction = 1
num_waypoints = 1200
radius = 0.4
step = 0.099910
output_filename = path + "TEST_wp.csv"


# Generate waypoints
generate_spiral_waypoints(num_waypoints, radius, step, output_filename, direction)
create_map(resolution, size, path, map)