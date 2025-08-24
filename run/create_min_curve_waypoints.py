import os
import sys 
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


import cv2
import math
import numpy as np
import csv
import yaml
import shutil

from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from scipy.signal import savgol_filter


from matplotlib import pyplot as plt
import trajectory_planning_helpers as tph


from utilities.Settings import Settings
from utilities.minimum_curvature_optimization import main_globaltraj


class GlobalPlanner:
    
    def __init__(self):
        
        # TODO: Check for which map_origin position the planner works...
        
        
        # Global Planer Settings
        self.safety_width = Settings.MIN_CURV_SAFETY_WIDTH
        self.show_plots = False
        
        # More Settings (Dont touch)
        map_config = yaml.load(open(Settings.MAP_CONFIG_FILE, "r"), Loader=yaml.FullLoader)
        self.map_name = Settings.MAP_NAME
        self.map_dir = os.path.join("utilities", "maps", self.map_name )
        self.map_origin = lambda: None
        position = lambda: None
        position.x = map_config["origin"][0]
        position.y = map_config["origin"][1]
        self.map_origin.position = position
        self.map_resolution = map_config["resolution"]
        
        self.initial_position = [0.,0., 0.]
        self.watershed = True  # use watershed algorithm
        
        self.map_info_str = ""
        
        print("Global Planner initialized")

    def compute_global_trajectory(self, reverse=False) -> bool:
        """
        Compute the global optimized trajectory of a map.

        Calculate the centerline of the track and compute global optimized trajectory with minimum curvature
        optimization.
        Publish the markers and waypoints of the global optimized trajectory.
        A waypoint has the following form: [s_m, x_m, y_m, d_right, d_left, psi_rad, vx_mps, ax_mps2]

        Parameters
        ----------
        cent_length
            Approximate length of the centerline
            
        Returns
        -------
        bool
            True if successfully computed the global waypoints
        """
        ################################################################################################################
        # Create a filtered black and white image of the map
        ################################################################################################################

        if reverse: self.initial_position=[0.,0., math.pi]
        
        img_path = os.path.join(self.map_dir, self.map_name + '_wp_min_curve.png')
        img_path_original = os.path.join(self.map_dir, self.map_name + '.png')
        
        print(f"loading map from {img_path}...")
        
        if(not os.path.isdir(os.path.join(self.map_dir,'data'))):
            os.makedirs(os.path.join(self.map_dir,'data' ))
        # check if file exists
        if(not os.path.isfile(img_path)):
            if(not os.path.isfile(img_path_original)):
                print("No valid map file found. Make sure there exists an image of the track including a closed contour at utilities/maps/" + Settings.MAP_NAME + "/" + Settings.MAP_NAME + ".png")
                exit()
            
            print("No map file found at utilities/maps/" + Settings.MAP_NAME + "/" + Settings.MAP_NAME + "wp_min_curve.png... Creating copy from original imgage. Use wp_min_curve for editing")
            shutil.copyfile(img_path_original, img_path)  
            
        bw = cv2.flip(cv2.imread(img_path, 0), 0)

        # Filtering with morphological opening
        kernel1 = np.ones((9, 9), np.uint8)
        opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel1, iterations=2)

        # get morphological skeleton of the map
        skeleton = skeletonize(opening, method='lee')
        path = os.path.join(self.map_dir,'data','skeleton.png')
        
        skeleton = skeleton.astype(np.uint8) * 255
        cv2.imwrite(path, skeleton)


        ################################################################################################################
        # Extract centerline from filtered occupancy grid map
        ################################################################################################################
        try:
            centerline = self.extract_centerline(skeleton=skeleton, cent_length=0)
        except IOError:
            print("No closed contours found! Check and edit image...")
            return False
        except ValueError:
            print("Couldn't find a closed contour with similar length as driven path!")
            return False
        
 
        centerline_smooth = self.smooth_centerline(centerline)
 

        # convert centerline from cells to meters
        centerline_meter = np.zeros(np.shape(centerline_smooth))
        centerline_meter[:, 0] = centerline_smooth[:, 0] * self.map_resolution + self.map_origin.position.x
        centerline_meter[:, 1] = centerline_smooth[:, 1] * self.map_resolution + self.map_origin.position.y

        # interpolate centerline to 0.1m stepsize: less computation needed later for distance to track bounds
        centerline_meter = np.column_stack((centerline_meter, np.zeros((centerline_meter.shape[0], 2))))
        centerline_meter_int = self.interp_track(reftrack=centerline_meter, stepsize_approx=0.1)[:, :2]
        
        
        # get distance to initial position for every point on centerline
        cent_distance = np.sqrt(np.power(centerline_meter_int[:, 0] - self.initial_position[0], 2)
                                + np.power(centerline_meter_int[:, 1] - self.initial_position[1], 2))

        min_dist_ind = np.argmin(cent_distance)

        cent_direction = np.angle([complex(centerline_meter_int[min_dist_ind, 0] -
                                           centerline_meter_int[min_dist_ind - 1, 0],
                                           centerline_meter_int[min_dist_ind, 1] -
                                           centerline_meter_int[min_dist_ind - 1, 1])])



        
        if self.show_plots:
            print("Direction of the centerline: ", cent_direction[0])
            print("Direction of the initial car position: ", self.initial_position[2])
            plt.plot(centerline_meter_int[:, 0], centerline_meter_int[:, 1], 'ko', label='Centerline interpolated')
            plt.plot(centerline_meter_int[min_dist_ind - 1, 0], centerline_meter_int[min_dist_ind - 1, 1], 'ro',
                     label='First point')
            plt.plot(centerline_meter_int[min_dist_ind, 0], centerline_meter_int[min_dist_ind, 1], 'bo',
                     label='Second point')
            plt.legend()
            plt.show()
            
     
        # flip centerline if necessary
        if not self.compare_direction(cent_direction, self.initial_position[2]):
            centerline_smooth = np.flip(centerline_smooth, axis=0)
            centerline_meter_int = np.flip(centerline_meter_int, axis=0)
            print("Centerline flipped")

     
        # create reversed centerline   
        # centerline_smooth_reverse = np.flip(centerline_smooth, axis=0)
        # centerline_meter_int_reverse = np.flip(centerline_meter_int, axis=0)

        # extract track bounds
        if self.watershed:
            try:
                bound_r_water, bound_l_water = self.extract_track_bounds(centerline_smooth, opening, save_img=False)
                # bound_r_water_rev, bound_l_water_rev = self.extract_track_bounds(centerline_smooth_reverse, opening)
                dist_transform = None
                print("Using watershed for track bound extraction...")
            except IOError:
                print("More than two track bounds detected with watershed algorithm")
                print("Trying with simple distance transform...")
                self.watershed = False
                bound_r_water = None
                bound_l_water = None
                # bound_r_water_rev = None
                # bound_l_water_rev = None
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
      

        ################################################################################################################
        # Compute global trajectory with mincurv_iqp optimization
        ################################################################################################################

        cent_with_dist = self.add_dist_to_cent(centerline_smooth=centerline_smooth,
                                               centerline_meter=centerline_meter_int,
                                               dist_transform=dist_transform,
                                               bound_r=bound_r_water,
                                               bound_l=bound_l_water)

        # Write centerline in a csv file and get a marker array of it
        self.write_centerline(cent_with_dist)


        print("Start Global Trajectory optimization with iterative minimum curvature...")
        path = os.path.join(self.map_dir, 'centerline')

        global_trajectory_iqp, bound_r_iqp, bound_l_iqp, est_t_iqp = main_globaltraj. \
            main_globaltraj(track_name=path,
                            curv_opt_type='mincurv_iqp',
                            safety_width=self.safety_width,
                            plot=(self.show_plots))

        self.map_info_str += f'IQP estimated lap time: {round(est_t_iqp, 4)}s; '
        self.map_info_str += f'IQP maximum speed: {round(np.amax(global_trajectory_iqp[:, 5]), 4)}m/s; '
    
        
        
        # do not use bounds of optimizer if the one's from the watershed algorithm are available
        if self.watershed:
            bound_r_iqp = bound_r_water
            bound_l_iqp = bound_l_water

        
        # self.publish_track_bounds(bound_r_iqp, bound_l_iqp, reverse=False)
        path = os.path.join(self.map_dir, 'data', self.map_name+'_bound_l_iqp.csv')
        np.savetxt( path,bound_l_iqp,delimiter=',', fmt='%f', header='x,y')
        
        path = os.path.join(self.map_dir, 'data', self.map_name+'bound_r_iqp.csv')
        np.savetxt( path,bound_r_iqp,delimiter=',', fmt='%f', header='x,y')
       

        d_right_iqp, d_left_iqp = self.dist_to_bounds(trajectory=global_trajectory_iqp,
                                                      bound_r=bound_r_iqp,
                                                      bound_l=bound_l_iqp,
                                                      centerline=centerline_meter_int)

        # global_traj_wpnts_iqp, global_traj_markers_iqp = self.create_wpnts_markers(trajectory=global_trajectory_iqp,
        #                                                                            d_right=d_right_iqp,
        #                                                                            d_left=d_left_iqp)
        path = os.path.join(self.map_dir,'data',  self.map_name+'_d_right_iqp.csv')
        np.savetxt( path,d_right_iqp,delimiter=',', fmt='%f', header='d')
        
        path = os.path.join(self.map_dir,'data',  self.map_name+'_d_left_iqp.csv')
        np.savetxt( path,d_left_iqp,delimiter=',', fmt='%f', header='d')
        
        # publish global trajectory markers and waypoints
        # self.vis_wpnt_global_iqp_pub.publish(global_traj_markers_iqp)
        # self.wpnt_global_iqp_pub.publish(global_traj_wpnts_iqp)
        
        # Save as _wp.csv or _wp_reverse.csv
        suffix = '_wp_reverse.csv' if reverse else '_wp.csv'
        path = os.path.join(self.map_dir, self.map_name + suffix)
        # Concatenate d_right_iqp and d_left_iqp as columns to global_trajectory_iqp
        global_trajectory_with_bounds = np.column_stack((global_trajectory_iqp, d_right_iqp, d_left_iqp))
        np.savetxt(
            path,
            global_trajectory_with_bounds,
            delimiter=',',
            fmt='%f',
            header='s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_right_iqp,d_left_iqp',
            comments=''
        )
    
        # global_trajectory_iqp[:,3] += 0.5 * np.pi
        
        # np.savetxt( path,np.array(global_trajectory_iqp),delimiter=',', fmt='%f', header='s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2', comments='')
        
        # Save image of track including waypoints
        plt.clf()
        x,y = bound_r_iqp.T
        plt.scatter(x, y, color='black')
        x,y = bound_l_iqp.T
        plt.scatter(x, y ,color='black')
        
        x = global_trajectory_iqp[:,1]
        y = global_trajectory_iqp[:,2]
        plt.scatter(x, y ,color='red')

        plt.axis('equal')
        
        
        plt.savefig(os.path.join(self.map_dir, 'data', 'waypoints.png'))
        
        
        print("Done. Waypouints saved. Drive carefully :)")
        
    
    def extract_centerline(self, skeleton, cent_length: float) -> np.ndarray:
        """
        Extract the centerline out of the skeletonized binary image.

        This is done by finding closed contours and comparing these contours to the approximate centerline
        length (which is known because of the driven path).

        Parameters
        ----------
        skeleton
            The skeleton of the binarised and filtered map
        cent_length : float
            Expected approximate centerline length

        Returns
        -------
        centerline : np.ndarray
            The centerline in form [[x1,y1],...] and in cells not meters

        Raises
        ------
        IOError
            If no closed contour is found
        ValueError
            If all found contours do not have a similar line length as the centerline (can only happen if
            {self.test_on_car} is True)
        """
        # get contours from skeleton
        contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # save all closed contours
        closed_contours = []
        for i, elem in enumerate(contours):
            opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
            if not opened:
                closed_contours.append(elem)

        # if we have no closed contour, we can't calculate a global trajectory
        if len(closed_contours) == 0:
            raise IOError("No closed contours")

        # calculate the line length of every contour to get the real centerline
        line_lengths = [math.inf] * len(closed_contours)
        for i, cont in enumerate(closed_contours):
            line_length = 0
            for k, pnt in enumerate(cont):
                line_length += np.sqrt((pnt[0][0] - cont[k - 1][0][0]) ** 2 +
                                       (pnt[0][1] - cont[k - 1][0][1]) ** 2)

            line_length *= self.map_resolution  # length in meters not cells
      
            line_lengths[i] = line_length

        # take the shortest line
        min_line_length = min(line_lengths)

        if min_line_length == math.inf:
            raise ValueError("Only invalid closed contour line lengths")

        min_length_index = line_lengths.index(min_line_length)
        # print(line_lengths)
        smallest_contour = np.array(closed_contours[min_length_index]).flatten()

        # reshape smallest_contours from the shape [x1,y1,x2,y2,...] to [[x1,y1],[x2,y2],...]
        # this will be the centerline
        len_reshape = int(len(smallest_contour) / 2)
        centerline = smallest_contour.reshape(len_reshape, 2)

        return centerline
    
    
    @staticmethod
    def smooth_centerline(centerline: np.ndarray) -> np.ndarray:
        """
        Smooth the centerline with a Savitzky-Golay filter.

        Notes
        -----
        The savgol filter doesn't ensure a smooth transition at the end and beginning of the centerline. That's why
        we apply a savgol filter to the centerline with start and end points on the other half of the track.
        Afterwards, we take the results of the second smoothed centerline for the beginning and end of the
        first centerline to get an overall smooth centerline

        Parameters
        ----------
        centerline : np.ndarray
            Unsmoothed centerline

        Returns
        -------
        centerline_smooth : np.ndarray
            Smooth centerline
        """
        # centerline_smooth = centerline
        # smooth centerline with a Savitzky Golay filter
        # filter_length = 20
        centerline_length = len(centerline)
        # print("Number of centerline points: ", centerline_length)

        if centerline_length > 2000:
            filter_length = int(centerline_length / 200) * 10 + 1
        elif centerline_length > 1000:
            filter_length = 81
        elif centerline_length > 500:
            filter_length = 41
        else:
            filter_length = 21
        centerline_smooth = savgol_filter(centerline, filter_length, 3, axis=0)

        # cen_len is half the length of the centerline
        cen_len = int(len(centerline) / 2)
        centerline2 = np.append(centerline[cen_len:], centerline[0:cen_len], axis=0)
        centerline_smooth2 = savgol_filter(centerline2, filter_length, 3, axis=0)

        # take points from second (smoothed) centerline for first centerline
        centerline_smooth[0:filter_length] = centerline_smooth2[cen_len:(cen_len + filter_length)]
        centerline_smooth[-filter_length:] = centerline_smooth2[(cen_len - filter_length):cen_len]

        return centerline_smooth
    
    
    @staticmethod
    def smooth_centerline(centerline: np.ndarray) -> np.ndarray:
        """
        Smooth the centerline with a Savitzky-Golay filter.

        Notes
        -----
        The savgol filter doesn't ensure a smooth transition at the end and beginning of the centerline. That's why
        we apply a savgol filter to the centerline with start and end points on the other half of the track.
        Afterwards, we take the results of the second smoothed centerline for the beginning and end of the
        first centerline to get an overall smooth centerline

        Parameters
        ----------
        centerline : np.ndarray
            Unsmoothed centerline

        Returns
        -------
        centerline_smooth : np.ndarray
            Smooth centerline
        """
        # centerline_smooth = centerline
        # smooth centerline with a Savitzky Golay filter
        # filter_length = 20
        centerline_length = len(centerline)
        # print("Number of centerline points: ", centerline_length)

        if centerline_length > 2000:
            filter_length = int(centerline_length / 200) * 10 + 1
        elif centerline_length > 1000:
            filter_length = 81
        elif centerline_length > 500:
            filter_length = 41
        else:
            filter_length = 21
        centerline_smooth = savgol_filter(centerline, filter_length, 3, axis=0)

        # cen_len is half the length of the centerline
        cen_len = int(len(centerline) / 2)
        centerline2 = np.append(centerline[cen_len:], centerline[0:cen_len], axis=0)
        centerline_smooth2 = savgol_filter(centerline2, filter_length, 3, axis=0)

        # take points from second (smoothed) centerline for first centerline
        centerline_smooth[0:filter_length] = centerline_smooth2[cen_len:(cen_len + filter_length)]
        centerline_smooth[-filter_length:] = centerline_smooth2[(cen_len - filter_length):cen_len]

        return centerline_smooth


    @staticmethod
    def interp_track(reftrack: np.ndarray,
                    stepsize_approx: float = 1.0) -> np.ndarray:
        """
        Created by:
        Alexander Heilmeier

        Documentation:
        Use linear interpolation between track points to create new points with equal distances.

        Inputs:
        reftrack:           array containing the track information that shell be interpolated [x, y, w_tr_right, w_tr_left].
        stepsize_approx:    desired stepsize for the interpolation

        Outputs:
        reftrack_interp:    interpolated reference track (unclosed)
        """

        # ------------------------------------------------------------------------------------------------------------------
        # FUNCTION BODY ----------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------

        reftrack_cl = np.vstack((reftrack, reftrack[0]))

        # calculate element lengths (euclidian distance)
        el_lenghts = np.sqrt(np.sum(np.power(np.diff(reftrack_cl[:, :2], axis=0), 2), axis=1))

        # sum up total distance (from start) to every element
        dists_cum = np.cumsum(el_lenghts)
        dists_cum = np.insert(dists_cum, 0, 0.0)

        # calculate desired lenghts depending on specified stepsize (+1 because last element is included)
        no_points_interp = math.ceil(dists_cum[-1] / stepsize_approx) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_points_interp)

        # interpolate closed track points
        reftrack_interp_cl = np.zeros((no_points_interp, 4))
        reftrack_interp_cl[:, 0] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 0])
        reftrack_interp_cl[:, 1] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 1])
        reftrack_interp_cl[:, 2] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 2])
        reftrack_interp_cl[:, 3] = np.interp(dists_interp, dists_cum, reftrack_cl[:, 3])

        # remove closed points
        reftrack_interp = reftrack_interp_cl[:-1]

        return reftrack_interp

    
    def extract_track_bounds(self, centerline: np.ndarray, filtered_bw: np.ndarray, save_img=False):
        """
        Extract the boundaries of the track.

        Use the watershed algorithm with the centerline as marker to extract the boundaries of the filtered black
        and white image of the map.

        Parameters
        ----------
        centerline : np.ndarray
            The centerline of the track (in cells not meters)
        filtered_bw : np.ndarray
            Filtered black and white image of the track
        save_img : bool
            Only saves sim map when specifically asked for because the function is called in reverse too
        Returns
        -------
        bound_right, bound_left : tuple[np.ndarray, np.ndarray]
            Points of the track bounds right and left in meters

        Raises
        ------
        IOError
            If there were more (or less) than two track bounds found
        """
        # create a black and white image of the centerline
        cent_img = np.zeros((filtered_bw.shape[0], filtered_bw.shape[1]), dtype=np.uint8)
        cv2.drawContours(cent_img, [centerline.astype(int)], 0, 255, 2, cv2.LINE_8)

        # create markers for watershed algorithm
        _, cent_markers = cv2.connectedComponents(cent_img)

        # apply watershed algorithm to get only the track (without any lidar beams outside the track)
        dist_transform = cv2.distanceTransform(filtered_bw, cv2.DIST_L2, 5)
        labels = watershed(-dist_transform, cent_markers, mask=filtered_bw)

        closed_contours = []

        for label in np.unique(labels):
            if label == 0:
                continue

            # Create a mask, the mask should be the track
            mask = np.zeros(filtered_bw.shape, dtype="uint8")
            mask[labels == label] = 255

            if self.show_plots:
                plt.imshow(mask, cmap='gray')
                plt.show()

          
            # Find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            # save all closed contours
            for i, cont in enumerate(contours):
                opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
                if not opened:
                    closed_contours.append(cont)

            # there must not be more (or less) than two closed contour
            if len(closed_contours) != 2:
                raise IOError("More than two track bounds detected! Check input")
            # draw the boundary into the centerline image
            cv2.drawContours(cent_img, closed_contours, 0, 255, 4)
            cv2.drawContours(cent_img, closed_contours, 1, 255, 4)

        # the longest closed contour is the outer boundary
        bound_long = max(closed_contours, key=len)
        bound_long = np.array(bound_long).flatten()

        # reshape from the shape [x1,y1,x2,y2,...] to [[x1,y1],[x2,y2],...]
        len_reshape = int(len(bound_long) / 2)
        bound_long = bound_long.reshape(len_reshape, 2)
        # convert to meter
        bound_long_meter = np.zeros(np.shape(bound_long))
        bound_long_meter[:, 0] = bound_long[:, 0] * self.map_resolution + self.map_origin.position.x
        bound_long_meter[:, 1] = bound_long[:, 1] * self.map_resolution + self.map_origin.position.y

        # inner boundary is the shorter one
        bound_short = min(closed_contours, key=len)
        bound_short = np.array(bound_short).flatten()

        # reshape from the shape [x1,y1,x2,y2,...] to [[x1,y1],[x2,y2],...]
        len_reshape = int(len(bound_short) / 2)
        bound_short = bound_short.reshape(len_reshape, 2)
        # convert to meter
        bound_short_meter = np.zeros(np.shape(bound_short))
        bound_short_meter[:, 0] = bound_short[:, 0] * self.map_resolution + self.map_origin.position.x
        bound_short_meter[:, 1] = bound_short[:, 1] * self.map_resolution + self.map_origin.position.y

        # get distance to initial position for every point on the outer bound to figure out if it is the right
        # or left boundary
        bound_distance = np.sqrt(np.power(bound_long_meter[:, 0] - self.initial_position[0], 2)
                                 + np.power(bound_long_meter[:, 1] - self.initial_position[1], 2))

        min_dist_ind = np.argmin(bound_distance)

        bound_direction = np.angle([complex(bound_long_meter[min_dist_ind, 0] - bound_long_meter[min_dist_ind - 1, 0],
                                            bound_long_meter[min_dist_ind, 1] - bound_long_meter[min_dist_ind - 1, 1])])

        norm_angle_right = self.initial_position[2] - math.pi
        if norm_angle_right < -math.pi:
            norm_angle_right = norm_angle_right + 2 * math.pi

        if self.compare_direction(norm_angle_right, bound_direction):
            bound_right = bound_long_meter
            bound_left = bound_short_meter
        else:
            bound_right = bound_short_meter
            bound_left = bound_long_meter

        if self.show_plots:
            plt.imshow(cent_img, cmap='gray')
            fig1, ax1 = plt.subplots()
            ax1.plot(bound_right[:, 0], bound_right[:, 1], 'b', label='Right bound')
            ax1.plot(bound_left[:, 0], bound_left[:, 1], 'g', label='Left bound')
            ax1.plot(centerline[:, 0] * self.map_resolution + self.map_origin.position.x,
                     centerline[:, 1] * self.map_resolution + self.map_origin.position.y, 'r', label='Centerline')
            ax1.legend()
            plt.show()

        return bound_right, bound_left


    @staticmethod
    def compare_direction(alpha: float, beta: float) -> bool:
        """
        Compare the direction of two points and check if they point in the same direction.

        Parameters
        ----------
        alpha : float
            direction angle in rad
        beta : float
            direction angle in rad

        Returns
        -------
        bool
            True if alpha and beta point in the same direction
        """
        delta_theta = math.fabs(alpha - beta)

        if delta_theta > math.pi:
            delta_theta = 2 * math.pi - delta_theta

        return delta_theta < math.pi / 2

    def add_dist_to_cent(self, centerline_smooth: np.ndarray,
                         centerline_meter: np.ndarray, dist_transform=None,
                         bound_r: np.ndarray = None, bound_l: np.ndarray = None) -> np.ndarray:
        """
        Add distance to track bounds to the centerline points.

        Parameters
        ----------
        centerline_smooth : np.ndarray
            Smooth centerline in cells (not meters)
        centerline_meter : np.ndarray
            Smooth centerline in meters (not cells)
        dist_transform : Any, default=None
            Euclidean distance transform of the filtered black and white image
        bound_r : np.ndarray, default=None
            Points of the right track bound in meters
        bound_l : np.ndarray, default=None
            Points of the left track bound in meters

        Returns
        -------
        centerline_comp : np.ndarray
            Complete centerline with distance to right and left track bounds for every point
        """
        centerline_comp = np.zeros((len(centerline_meter), 4))

        if dist_transform is not None:
            width_track_right = dist_transform[centerline_smooth[:, 1].astype(int),
                                               centerline_smooth[:, 0].astype(int)] * self.map_resolution
            if len(width_track_right) != len(centerline_meter):
                width_track_right = np.interp(np.arange(0, len(centerline_meter)), np.arange(0, len(width_track_right)),
                                              width_track_right)
            width_track_left = width_track_right
        elif bound_r is not None and bound_l is not None:
            width_track_right, width_track_left = self.dist_to_bounds(centerline_meter, bound_r, bound_l,
                                                                      centerline=centerline_meter)
        else:
            raise IOError("Error! Missing arguments, check inputs...")

        centerline_comp[:, 0] = centerline_meter[:, 0]
        centerline_comp[:, 1] = centerline_meter[:, 1]
        centerline_comp[:, 2] = width_track_right
        centerline_comp[:, 3] = width_track_left

        return centerline_comp
    
    def dist_to_bounds(self, trajectory: np.ndarray, bound_r, bound_l, centerline: np.ndarray):
        """
        Calculate the distance to track bounds for every point on a trajectory.

        Parameters
        ----------
        trajectory : np.ndarray
            A trajectory in form [s_m, x_m, y_m, psi_rad, vx_mps, ax_mps2] or [x_m, y_m]
        bound_r
            Points in meters of boundary right
        bound_l
            Points in meters of boundary left
        centerline : np.ndarray
            Centerline only needed if global trajectory is given and plot of it is wanted

        Returns
        -------
        dists_right, dists_left : tuple[np.ndarray, np.ndarray]
            Distances to the right and left track boundaries for every waypoint
        """
        # check format of trajectory
        if len(trajectory[0]) > 2:
            help_trajectory = trajectory[:, 1:3]
            trajectory_str = "Global Trajectory"
        else:
            help_trajectory = trajectory
            trajectory_str = "Centerline"

        # interpolate track bounds
        bound_r_tmp = np.column_stack((bound_r, np.zeros((bound_r.shape[0], 2))))
        bound_l_tmp = np.column_stack((bound_l, np.zeros((bound_l.shape[0], 2))))

        bound_r_int = self.interp_track(reftrack=bound_r_tmp,
                                                                      stepsize_approx=0.1)
        bound_l_int = self.interp_track(reftrack=bound_l_tmp,
                                                                      stepsize_approx=0.1)

        # find the closest points of the track bounds to global trajectory waypoints
        n_wpnt = len(help_trajectory)
        dists_right = np.zeros(n_wpnt)  # contains (min) distances between waypoints and right bound
        dists_left = np.zeros(n_wpnt)  # contains (min) distances between waypoints and left bound

        # print(f"Calculating distance from {trajectory_str} to track bounds...")
        for i, wpnt in enumerate(help_trajectory):
            dists_bound_right = np.sqrt(np.power(bound_r_int[:, 0] - wpnt[0], 2)
                                        + np.power(bound_r_int[:, 1] - wpnt[1], 2))
            dists_right[i] = np.amin(dists_bound_right)

            dists_bound_left = np.sqrt(np.power(bound_l_int[:, 0] - wpnt[0], 2)
                                       + np.power(bound_l_int[:, 1] - wpnt[1], 2))
            dists_left[i] = np.amin(dists_bound_left)

        # print(f"Done calculating distance from {trajectory_str} to track bounds")

        if self.show_plots and trajectory_str == "Global Trajectory":
            width_veh_real = 0.3
            normvec_normalized_opt = tph.calc_normal_vectors.calc_normal_vectors(trajectory[:, 3])

            veh_bound1_virt = trajectory[:, 1:3] + normvec_normalized_opt * self.safety_width / 2
            veh_bound2_virt = trajectory[:, 1:3] - normvec_normalized_opt * self.safety_width / 2

            veh_bound1_real = trajectory[:, 1:3] + normvec_normalized_opt * width_veh_real / 2
            veh_bound2_real = trajectory[:, 1:3] - normvec_normalized_opt * width_veh_real / 2

            # plot track including optimized path
            fig, ax = plt.subplots()

            # ax.plot(refline[:, 0], refline[:, 1], "k--", linewidth=0.7, label="Reference line")
            ax.plot(bound_r[:, 0], bound_r[:, 1], "k-", linewidth=1.0, label="Track bounds")
            ax.plot(bound_l[:, 0], bound_l[:, 1], "k-", linewidth=1.0)
            ax.plot(centerline[:, 0], centerline[:, 1], "k--", linewidth=1.0, label='Centerline')

            ax.plot(veh_bound1_virt[:, 0], veh_bound1_virt[:, 1], "b", linewidth=0.5, label="Vehicle width with safety")
            ax.plot(veh_bound2_virt[:, 0], veh_bound2_virt[:, 1], "b", linewidth=0.5)
            ax.plot(veh_bound1_real[:, 0], veh_bound1_real[:, 1], "c", linewidth=0.5, label="Real vehicle width")
            ax.plot(veh_bound2_real[:, 0], veh_bound2_real[:, 1], "c", linewidth=0.5)

            ax.plot(trajectory[:, 1], trajectory[:, 2], 'tab:orange', linewidth=2.0, label="Global trajectory")

            plt.grid()
            ax1 = plt.gca()

            point1_arrow = np.array([trajectory[0, 1], trajectory[0, 2]])
            point2_arrow = np.array([trajectory[5, 1], trajectory[5, 2]])
            vec_arrow = (point2_arrow - point1_arrow)
            ax1.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1], width=0.05,
                      head_width=0.3, head_length=0.3, fc='g', ec='g')

            ax.set_aspect("equal", "datalim")
            plt.xlabel("x-distance from origin [m]")
            plt.ylabel("y-distance from origin [m]")
            plt.title(f"Global trajectory with vehicle width")
            plt.legend()

            plt.show()

        return dists_right, dists_left
    
    def write_centerline(self, centerline: np.ndarray) -> bool:
        """
        Create a csv file with centerline data.

        The centerline data includes position and width to the track bounds in meter, so that it can be used in the
        global trajectory optimizer. The file has the following format: x_m, y_m, w_tr_right_m, w_tr_left_m .

        Parameters
        ----------
        centerline : np.ndarray
            The centerline in form [x_m, y_m, w_tr_right_m, w_tr_left_m]
        sp_bool : bool, default=False
            Used for shortest path optimization when another centerline csv should be created

        """
        path = os.path.join(self.map_dir, 'centerline.csv')
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            id_cnt = 0  # for marker id

            for row in centerline:
                x_m = row[0]
                y_m = row[1]
                width_tr_right_m = row[2]
                width_tr_left_m = row[3]
                writer.writerow([x_m, y_m, width_tr_right_m, width_tr_left_m])


        return True
    
    
global_planner = GlobalPlanner()
global_planner.compute_global_trajectory( reverse=False)
global_planner.compute_global_trajectory( reverse=True)