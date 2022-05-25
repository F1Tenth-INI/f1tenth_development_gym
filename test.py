from re import S
import tensorflow as tf
import numpy as np

def get_minimmum_distances(distances):
    minima = tf.math.reduce_min(distances, axis=1)
    
    
    distance_threshold = tf.constant([0.1])

    indices_too_close = tf.math.less(minima, distance_threshold)
    crash_cost = tf.cast(indices_too_close, tf.float32) * 10000
    print(minima.numpy())
    print(crash_cost.numpy())

    

def distances_from_list_to_list_of_points(points1, points2):
    
    length1 = tf.shape(points1)[0]
    length2 = tf.shape(points2)[0]
    
    points1 = tf.tile([points1], [1, 1, length2])
    points1 = tf.reshape(points1, (length1 * length2, 2))
    
    points2 = tf.tile([points2], [length1, 1,1])
    points2 = tf.reshape(points2, (length1 * length2, 2))
    
    diff = points2 - points1
    squared_diff = tf.math.square(diff)
    squared_dist = tf.reduce_sum(squared_diff, axis=1)
    
    squared_dist = tf.reshape(squared_dist, [length1,length2])
    
    return squared_dist
    


def distances_to_list_of_points(point, points2):
    
    length = tf.shape(points2)[0]
    points1 = tf.tile([point], [length, 1])
    
    diff = points2 - points1
    squared_diff = tf.math.square(diff)
    squared_dist = tf.reduce_sum(squared_diff, axis=1)
    
    print(squared_dist.numpy())
    
    
point_list =     [[0,1],[1,0],[1,1],[1,1],[1,3],[2,1],]
border_points =  [[0,1],[1,1],[1,1],[1,3],]
point = [0,1]

# [[0. 0. 1. 1. 5. 4.]
#  [0. 0. 1. 1. 5. 4.]
#  [1. 1. 0. 0. 4. 1.]
#  [1. 1. 0. 0. 4. 1.]
#  [5. 5. 4. 4. 0. 5.]
#  [4. 4. 1. 1. 5. 0.]]
    
a = tf.constant(point, tf.float32)
b = tf.constant(point_list, tf.float32)
c = tf.constant(border_points, tf.float32)


def dist_to_array(point, points2):
    points1 = len(points2) * [point]

    points1 = np.array(points1)
    points2 = np.array(points2)

    diff_x = points1[:,0] - points2[:,0]
    diff_y = points1[:,1] - points2[:,1]
    squared_distances = np.square(diff_x) + np.square(diff_y)

    return squared_distances



squared_distances = distances_from_list_to_list_of_points(b,c)
# dist = distances_to_list_of_points(a, b)
# dist = dist_to_array(point, point_list)
print("Squared distances")
print(squared_distances.numpy())
# [[0. 1. 1. 5.]
#  [2. 1. 1. 9.]
#  [1. 0. 0. 4.]
#  [1. 0. 0. 4.]
#  [5. 4. 4. 0.]
#  [4. 1. 1. 5.]]


get_minimmum_distances(squared_distances)