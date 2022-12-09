import math
import matplotlib.pyplot as plt
import csv

from utilities.waypoint_utils import *
# from PIL import Image  
from PIL import Image, ImageDraw


def transform_track_data(map_name,
    map_scale_factor,
    width_scale_factor = 2.0,
    image_margin = 10 ):

    track_points = pd.read_csv("utilities/maps_files/tum_track_db/original/"+map_name+".csv").to_numpy()

    for track_point in track_points:
        track_point[0] =  map_scale_factor * track_point[0] 
        track_point[1] = map_scale_factor * track_point[1] 
        track_point[2] = map_scale_factor * width_scale_factor * track_point[2] 
        track_point[3] = map_scale_factor * width_scale_factor * track_point[3] 


    offset_x = (abs(np.min(track_points[:, 0])) + image_margin)
    offset_y = (abs(np.min(track_points[:, 1])) + image_margin)
    offset = [offset_x, offset_y]


    for track_point in track_points:
        track_point[0] =  track_point[0] + offset[0]
        track_point[1] = track_point[1] + offset[1]


    file = open("utilities/maps_files/tum_track_db/transformed/"+map_name+".csv", 'w')
    writer = csv.writer(file)
    writer.writerows(track_points)
    file.close()
    

def draw_map_and_create_yaml(map_name, scaling, pixel_per_m = 10, image_margin = 10, width_factor = 2., track_width_sumand = 0.4):
    def get_normal_vector_normed(v):
        v_n = [0,0]
        v_n[0] = -v[1]
        v_n[1] = v[0]
        
        norm = np.linalg.norm(v_n)
        v_n = v_n/norm
        return v_n


    track_points = pd.read_csv("utilities/maps_files/tum_track_db/transformed/"+map_name+".csv").to_numpy()
    first_track_point = track_points
    track_points = np.vstack([track_points, track_points[0, :]])


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

    print("Starting point : ", starting_point )
    print("Width/Height : ", width, height )
    print("origin_x : ", origin_x )
    print("origin_y : ", origin_y )

    wp_offset_x = origin_x
    wp_offset_y = height/pixel_per_m + origin_y
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
    img.save("utilities/maps_files/maps/"+map_name+".png")

    # Write YAML file 
    data = {
        'image' : map_name + '.png',
        'resolution' : 1.0 / pixel_per_m,
        'origin' : [float(origin_x), float(origin_y), 0.00],
        'negate' : 0,
        'occupied_thresh' : 0.45,
        'free_thresh' : 0.196,
        }

    with open(r"utilities/maps_files/maps/"+map_name+".yaml", 'w') as file:
        documents = yaml.dump(data, file)
        
        
        