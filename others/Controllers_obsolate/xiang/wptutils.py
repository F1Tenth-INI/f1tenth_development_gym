##################################################################
import matplotlib.pyplot as plt
import pandas as pd
import os
import pyglet
import numpy as np
import math
# ---Deng, Xiang, dxiang@ini.ethz.ch
################################################################

class WptUtil:

    def __init__(self):
        print("WptUtil initialized")
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        print(self.dir_path)
        self.wpts = []
        self.wpt_ref = []
        self.loadWpts()
        self.trajectory=np.append(self.wpts,[self.wpts[0,:]], axis=0)
        ########################################################
        self.diffs = self.trajectory[1:, :] - self.trajectory[:-1, :]
        self.l2s = self.diffs[:, 0]**2 + self.diffs[:, 1]**2

        self.wpts_opt=[]
    def loadWpts(self,filename='/Oschersleben_map_wpts'):
        path = self.dir_path+filename
        waypoints = pd.read_csv(path+'.csv', header=None).to_numpy()
        self.wpts = waypoints
        if True:
            # path2 = self.dir_path+'/nicholas_icra_global_wpnts'
            # waypoints2 = pd.read_csv(path2+'.csv', header=None).to_numpy()
            # waypoints2=waypoints2[0:-1:10,1:3] 

            path2 = self.dir_path+'/Oschersleben_map_wpts_dense800_60'
            waypoints2 = pd.read_csv(path2+'.csv', header=None).to_numpy()
            waypoints2=waypoints2[0:-1:1,1:3] 
            self.wpts_opt=waypoints2 
            self.wpts=self.wpts_opt
    def normalizeAngle(self,theta):
        """return angle in [-pi, pi]
        """
        while theta>np.pi:
            theta=theta-2.0*np.pi
        while theta<-np.pi:
            theta=theta+2.0*np.pi

        return theta     
    def find_nearest_point_on_trajectory(self,point, trajectory):
            """ Acknowledgement: adapted from the original f1tenth code base##########################
            Return the nearest wpt given a point 2x1 dim
            trajectory: waypoints, Nx2
            """
            # print(trajectory.shape)
            ########################################################      
            
            dots = np.empty((self.trajectory.shape[0]-1, ))
            tmp = point - self.trajectory   
            dots = np.sum(np.multiply(tmp[range(dots.shape[0]),:],self.diffs[:,:]), axis=1)
            # deprecated, these stupid for-loop significantly slow down the code
            # for i in range(dots.shape[0]):
            #     dots[i] = np.dot(tmp[i,:], self.diffs[i, :])
            t = dots / self.l2s
            t[t < 0.0] = 0.0
            t[t > 1.0] = 1.0
            projections = self.trajectory[:-1, :] + (t*self.diffs.T).T
            tmp=point-projections
            distssq=np.sum(np.multiply(tmp,tmp),axis=1)
            # deprecated, these stupid for-loop significantly slow down the code
            # for i in range(dists.shape[0]):
            #     temp = point - projections[i]
            #     dists[i] = np.sqrt(np.sum(temp*temp))
            min_dist_segment = np.argmin(distssq)
            return projections[min_dist_segment], np.sqrt(distssq[min_dist_segment]), t[min_dist_segment], min_dist_segment
    def get_wpt_ref(self,pose_x,pose_y,pose_theta, min_dist_segment,K=3):
        next_wpt_id = min_dist_segment+K
        if next_wpt_id >= self.wpts.shape[0]:
            next_wpt_id = next_wpt_id-self.wpts.shape[0]
        self.wpt_ref = self.wpts[next_wpt_id,:]
        wpt_diff=self.wpt_ref-[pose_x,pose_y]
        self.theta2wpt=self.normalizeAngle(np.arctan2(wpt_diff[1],wpt_diff[0])-pose_theta)
        return self.wpt_ref, self.theta2wpt
    def angleDistance(self, theta1, theta2):
        return np.abs(np.arctan2(np.sin(theta1-theta2),np.cos(theta1-theta2)))


    def suggestGap(self,gaps,largest_gap_index,distances,angles, steering_angle, max_distance,theta2wpt): 
        if len(gaps)>0 and True:
            index_gapII=largest_gap_index
            if len(gaps)>1:
                gap_thetas = np.zeros((len(gaps),))
                gap_widths = np.zeros((len(gaps),))
                for i in range(len(gaps)):
                    gap=gaps[i]
                    gap_thetas[i]=(gap[0]+gap[1])/2
                    gap_widths[i] = gap[5]
                
                theta_dist=self.angleDistance(gap_thetas,theta2wpt)
                vals=np.divide(theta_dist,gap_widths)
                index_gapII=np.argmin(vals) 
                gap=gaps[index_gapII]
                gap_starting_index=gap[2]
                gap_closing_index=gap[3]
                distances_sub=distances[gap_starting_index+1:gap_closing_index] 

                max_distance=np.nanmean(distances_sub) 
                steering_angle=gap_thetas[index_gapII] 
                good_gap = gaps[index_gapII]
            else:
                gap=gaps[index_gapII]
                gap_starting_index=gap[2]
                gap_closing_index=gap[3]
                distances_sub=distances[gap_starting_index+1:gap_closing_index]
                if np.array(distances_sub).size==0:
                    distances_sub=distances[gap_starting_index:gap_closing_index+1]
                # print(distances_sub)
                max_distance=np.nanmean(distances_sub)
            gap=gaps[index_gapII]
            gap_starting_index=gap[2]
            gap_closing_index=gap[3]
            distances_sub=distances[gap_starting_index+1:gap_closing_index]
            if np.array(distances_sub).size==0:
                gap_starting_index-=1
                gap_closing_index+=1
            angles_sub=angles[gap_starting_index+1:gap_closing_index]
            distances_sub=distances[gap_starting_index+1:gap_closing_index]
            theta_dist=self.angleDistance(angles_sub,theta2wpt)
            isel= np.argmin(theta_dist)
            steering_angleII=angles_sub[isel]
            steering_angle=(steering_angleII+3*steering_angle)/4
            # max_distance=distances_sub[isel]
        return steering_angle, max_distance        
    def normalizedProjection(self,pose_x,pose_y,targets,wpt_ref):
        diff_tar = targets-[pose_x,pose_y]
        diff_ref = wpt_ref-[pose.x,pose.y]
        diff_tar_norm = np.sqrt(np.sum(np.multiply(diff_tar,diff_tar),axis=1))
        diff_ref_norm = np.sqrt(np.sum(np.multiply(diff_ref,diff_ref),axis=1))
        diff_ref /= diff_ref_norm
        diff_tar = np.divide(diff_tar, [diff_tar_norm,diff_tar_norm])
        projection = np.sum(np.multiply(np.repeat(diff_ref,),diff_tar),axis=1)