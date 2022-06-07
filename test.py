from re import S
import tensorflow as tf
import numpy as np

import tensorflow_probability as tfp


from MPPI.mppi_planner import MppiPlanner

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid

from SI_Toolkit_ASF_global.predictors_customization_tf import next_state_predictor_ODE_tf






def interpolate_mean(a, number_of_steps):
    '''Interpolate control inputs
    @param a: list of control inputs (number_of_rollouts, horizon, number of control inputs)'''

    index_steps = number_of_steps - 1

    a = tf.repeat(a, number_of_steps, axis=1)

    interpolation = tf.zeros([tf.shape(a)[0], tf.shape(a)[1] - index_steps, tf.shape(a)[2]], dtype= tf.float32)

    i = tf.constant(0)
    while_condition = lambda i, interpolation, a: tf.less(i, number_of_steps)
    def body(i, interpolation, a):
        # do something here which you want to do in your loop
        # increment i

        lenght = tf.shape(a)[1] - index_steps
        
        a_sliced = tf.slice(a, [0, i, 0], [a.shape[0], lenght ,a.shape[2]])
        a_numpy = a.numpy()
        
        interpolation = tf.add(interpolation, a_sliced)
        interpolation_numpy = interpolation.numpy()
        return [tf.add(i, 1), interpolation, a]

    # do the loop:
    index, result, a = tf.while_loop(while_condition, body, [i, interpolation, a])

    result = result/number_of_steps
    
    return result



# interpolation = (
#     a[3:] + a[2:-1] + a[1:-2] + a[:-3]
#     )/number_of_steps

# length = tf.cast(tf.shape(a), tf.float32)
# mask_every_second = tf.math.mod(tf.range(0.0, length, 1), 2)


a = tf.constant([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0],[3.0, 3.0],[4.0, 4.0], [5.0, 5.0]]])
result =interpolate_mean(a, 2)

print("result_numpy",result.numpy())
exit()

dt=0.04
intermediate_steps=4,
num_rollouts = 2000

next_step_predictor = next_state_predictor_ODE_tf(dt, intermediate_steps, disable_individual_compilation=True)

# x1: x position in global coordinates
# x2: y position in global coordinates
# x3: steering angle of front wheels
# x4: velocity in x direction
# x5: yaw angle
# x6: yaw rate
# x7: slip angle at vehicle center
        
test_initial_state = [5, 6, 0.3, 5, 0, 0, 0]

# Steering velocity / Acceleration
test_control_input = [0.,1.0]


number_of_steps = 1

# Conventional model:

planner = MppiPlanner()
planner.car_state = test_initial_state
next_state = test_initial_state
for  i in range(number_of_steps):
    next_state = planner.simulate_step(next_state ,test_control_input)


result = np.around(next_state, 3)
print("Conventional Next state: ", result)


initial_state = test_initial_state
initial_state = np.tile(initial_state, tf.constant([num_rollouts, 1]))
initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float32)

control_input = [test_control_input[1],test_control_input[0]] 
control_input = np.tile(control_input, tf.constant([num_rollouts, 1]))
control_input = tf.convert_to_tensor(control_input, dtype=tf.float32)

for  i in range(number_of_steps):
    next_state = next_step_predictor.step(initial_state, control_input, 0)

result = next_state.numpy()
result = np.around(result, 3)
print("Tensorflow Next State",result[0])

exit()

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