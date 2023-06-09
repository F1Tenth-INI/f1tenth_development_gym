import math
import numpy as np
from utilities.state_utilities import *
# Obstcle is defined as array
# [x, y]

OBS_X_IDX = 0
OBS_Y_IDX = 1

class ObstacleDetector:
    
    number_of_fixed_length_array = 10
    
    def __init__(self):
        
        self.scan_angles = np.linspace(-2.35,2.35, 1080)
        self.clip_indices = 200
        self.clip_distance = 10 # [m]
        self.delta_well = 1.0 # [m]
        
        
        
    
    def get_obstacles(self,scans, car_state):
        
        start_index = self.clip_indices
        end_index = 1080 - self.clip_indices
        
        scans = np.array(scans)
        scans = scans[start_index : end_index]
        scans = np.clip(scans, 0, self.clip_distance)
        scan_angles = self.scan_angles[start_index : end_index]
    
        
        obstacle_down = False
        obstacle_down_index = 0
        obstacle_up = False
        obstacle_up_index = 0
        
        obstacles = []
        
        # plt.clf()
        # plt.title("Lidar Data")
        # plt.plot(scans)
        # plt.savefig('lll.png')
        
        wells = []
        for i in range (0, len(scans)-2):
            delta = scans[i+2]-scans[i]
        
            if(delta <= -self.delta_well):
                obstacle_down = True
                obstacle_down_index = i
            if(delta >self.delta_well) and obstacle_down and scans[obstacle_down_index+2] < 6: # and i <=obstacle_down_index+200 :
                obstacle_down = False
                obstacle_up = True
                obstacle_up_index = i
                well = [obstacle_down_index, obstacle_up_index]
                wells.append(well)
        
        for i, well in enumerate(wells):
            a_open = scan_angles[well[0]]
            a_close = scan_angles[well[1]]
            a_center = (a_open + a_close)/2
            
            d_open = scans[well[0]+2]
            d_close = scans[well[1]-2]
            d_center = (d_open + d_close) / 2
            
            
            
            obstacle = [
                car_state[POSE_X_IDX] + d_center * math.cos(a_center + car_state[POSE_THETA_IDX]), 
                car_state[POSE_Y_IDX] + d_center* math.sin(a_center  + car_state[POSE_THETA_IDX])]
                
            obstacles.append(obstacle)
        
        obstacles = np.array(obstacles)
        return obstacles
    
    @staticmethod
    def get_fixed_length_obstacle_array(obstacles):
        obstacles_fixed_length =np.full((ObstacleDetector.number_of_fixed_length_array, 2), 10000, dtype=np.float32)
        for i, obstacle in enumerate(obstacles):
            obstacles_fixed_length[i] = obstacle
            
        return obstacles_fixed_length
            
        
            