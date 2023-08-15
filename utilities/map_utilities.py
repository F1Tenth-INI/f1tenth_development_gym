import math
import matplotlib.pyplot as plt
import csv

from utilities.waypoint_utils import *
# from PIL import Image  
from PIL import Image, ImageDraw

class MapUtilities:
    
    
    @staticmethod  
    def squared_distance(p1, p2):
        squared_distance = abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2
        return squared_distance
    
    @staticmethod
    def get_closest_point(point, track_points):
        min_dist = 100000
        closest_index = 0
        
        for i, track_point in enumerate(track_points):
            sd = MapUtilities.squared_distance(point, track_point)
            if(sd < min_dist):
                min_dist = sd
                closest_index = i
        return closest_index, track_points[closest_index]
    
    @staticmethod
    def transform_track_data(track_points, scaling, offset_x, offset_y, width_scale_factor = 2.0):
        track_points = np.array(track_points)

        for track_point in track_points:
            # transform and scale
            track_point[0] = track_point[0] + offset_x
            track_point[1] = track_point[1] + offset_y
            
            # Adjust width
            track_point[2] = width_scale_factor * track_point[2] 
            track_point[3] = width_scale_factor * track_point[3] 

            track_point[0] = track_point[0] * scaling
            track_point[1] = track_point[1] * scaling
            track_point[2] = track_point[2] * scaling
            track_point[3] = track_point[3] * scaling

        print(track_points)
        return track_points
        
    @staticmethod
    def draw_map_and_create_yaml(map_name, path_to_map, pixel_per_m = 10, image_margin = 10, width_factor = 2., track_width_sumand = 0.4):
       
        def get_normal_vector_normed(v):
            v_n = [0,0]
            v_n[0] = -v[1]
            v_n[1] = v[0]
            
            norm = np.linalg.norm(v_n)
            v_n = v_n/norm
            return v_n
        
        track_points = pd.read_csv(path_to_map+map_name+".csv").to_numpy()
        first_track_point = track_points
        track_points = np.vstack([track_points, track_points[0, :]])

        # Shift trackpoints by map_margin (margin top/left)
        for track_point in track_points:
            track_point[0] += image_margin
            track_point[1] += image_margin

        track_vectors = []
        track_borders_left = []
        track_borders_right = []

        for i in range(len(track_points) - 1 ):
            track_point = track_points[i]
            track_point_position = track_points[i][:2]
            next_track_point_position = track_points[i+1][:2]
            track_vector = next_track_point_position - track_point_position
            track_vector_normal = get_normal_vector_normed(track_vector)
            track_width_right = track_point[2] 
            track_width_left = track_point[3] 
            track_point_left = track_point_position - (track_width_left + track_width_sumand) *width_factor * track_vector_normal
            
            track_borders_left.append(track_point_left)
            track_borders_right.append(track_point_position + (track_width_right + track_width_sumand) *width_factor* track_vector_normal)
            
            track_vectors.append(track_vector)

        track_points = track_points[:len(track_points)-1, :]

        track_borders_left.append(track_borders_left[0])
        track_borders_right.append(track_borders_right[0])
        track_borders_left = np.array(track_borders_left)
        track_borders_right = np.array(track_borders_right)



        offset_x = 0 # pixel_per_m * (abs(np.min(track_points[:, 0])) + image_margin)
        offset_y = 0 # pixel_per_m * (abs(np.min(track_points[:, 1])) + image_margin)
        offset = [offset_x, offset_y]

        width = int(pixel_per_m * (np.max(track_points[:, 0]) + image_margin)) 
        height = int(pixel_per_m * (np.max(track_points[:, 1] ) + image_margin))


        img  = Image.new( mode = "L", size = (width, height), color ="white" )

        draw = ImageDraw.Draw(img)
        # draw.line((0, 0) + img.size, fill= (255, 255, 255))
        # draw.line((0, img.size[1], img.size[0], 0), fill=128)


        draw.line((0, 1, img.size[0], 1), fill=0)
        draw.line((0,  img.size[1] - 1 , img.size[0],  img.size[1] -1 ), fill=0)
        draw.line((1, 0, 1, img.size[1] ), fill=0)
        draw.line(( img.size[0] - 1 , 0,  img.size[0] -1 , img.size[1] ), fill=0)

        starting_point = pixel_per_m *  track_points[0][:2]
        draw.point(xy=[starting_point[0], starting_point[1]], fill=(180) )

        origin_x = - starting_point[0] / pixel_per_m 
        origin_y = - (height - starting_point[1]) / pixel_per_m 

        # print("Starting point : ", starting_point )
        # print("Width/Height : ", width, height )
        # print("origin_x : ", origin_x )
        # print("origin_y : ", origin_y )

        # Calculate offset for waypoint due to image size and image margin
        wp_offset_x = origin_x + image_margin
        wp_offset_y = height/pixel_per_m + origin_y - image_margin
        print("wp_offset_x : ", wp_offset_x )
        print("wp_offset_y : ", wp_offset_y )

        for i in range(len(track_points)):
            track_point = pixel_per_m *  track_points[i][:2] + offset
            track_point_left = (pixel_per_m *  track_borders_left[i] + offset)
            track_point_left_next = ( pixel_per_m *  track_borders_left[i+1] + offset)
            track_point_right = ( pixel_per_m *  track_borders_right[i] + offset)
            track_point_right_next = (pixel_per_m *  track_borders_right[i+1] + offset)
            
            # print("Trackpoint", track_point)
            # draw.point(xy=[track_point_left[0], track_point_left[1]])
            # draw.point(xy=[track_point_right[0], track_point_right[1]])
            draw.line((track_point_left[0], track_point_left[1], track_point_left_next[0], track_point_left_next[1]),  fill=0, width=2)
            draw.line((track_point_right[0], track_point_right[1], track_point_right_next[0], track_point_right_next[1]),  fill=0, width=2)



        # img.show()
        img.save(path_to_map+map_name+".png")

        # Write YAML file for map
        data = {
            'image' : map_name + '.png',
            'resolution' : 1.0 / pixel_per_m,
            'origin' : [float(origin_x), float(origin_y), 0.00],
            'negate' : 0,
            'occupied_thresh' : 0.45,
            'free_thresh' : 0.196,
            }

        with open(path_to_map+map_name+".yaml", 'w') as file:
            documents = yaml.dump(data, file)
        return wp_offset_x, wp_offset_y
            
    @staticmethod
    def transform_waypoints(waypoints, scaling, offset_x, offset_y ):
        waypoints = np.array(waypoints)
        for waypoint in waypoints:
            
            waypoint[1] = waypoint[1] + offset_x
            waypoint[2] = waypoint[2] + offset_y
            waypoint = waypoint * scaling
            
        return waypoints     
        
    @staticmethod
    def save_waypoints(waypoints, map_name, path_to_waypoints="utilities/maps_files/waypoints/"):
        file_name = map_name+"_wp.csv"
        file = open(path_to_waypoints+file_name, 'w')
        writer = csv.writer(file)
        writer.writerows(waypoints)
        file.close()

    """@staticmethod
    def save_waypoints(waypoints, map_name, path_to_waypoints = "utilities/maps_files/waypoints/"):
        file_name = map_name+"_wp.csv"
        file = open(path_to_waypoints+file_name, 'w')
        writer = csv.writer(file)
        writer.writerows(waypoints)
        file.close()"""

            