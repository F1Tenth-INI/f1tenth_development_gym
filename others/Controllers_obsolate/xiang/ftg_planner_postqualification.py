# from pyglet.gl import GL_POINTS
# import pyglet
import numpy as np
import math
import scipy.signal as sp
# import matplotlib.pyplot as plt

##################################################################
# import matplotlib.pyplot as plt
from xiang.wptutils import WptUtil
# from pkg.FollowTheGap.ftg_planner import FollowTheGapPlannerSafeMode

"""
    Adapted FTG with wpt guidence
    current lapse time is 53.6099

    authors: Florian and Xiang
"""
################################################################

class FollowTheGapPlanner:
    """
    Qualification Planner
    """

    def __init__(self):

        print("Controller initialized")
        #Settings
        self.speed_fraction = 2.05
        self.safe_mode_speed_fraction = 1.3
        self.lidar_lookahead_distance = 6.2

        self.kickdown_factor = 1.2
        self.double_kickdiwn_factor = 1.3

        self.max_number_of_gaps_for_safe_mode = 1
        self.min_eucledian_gap_distance_for_safe_mode = 88
        self.angle_penalty = 10
        self.safe_mode_angle_penality = 15

        self.lidar_border_points = 1080 * [[0, 0]]
        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35, 2.35, 1080)
        self.simulation_index = 0
        self.plot_lidar_data = False
        self.draw_lidar_data = True
        self.lidar_visualization_color = (0, 0, 0)
        self.lidar_live_gaps = []
        self.min_gap_dist = 10000

        self.kickdown = False
        self.last_steering = [0, 0]

        # Safe mode (in case of obstacles)
        self.safe_mode = False
        self.safe_planner = FollowTheGapPlannerSafeMode()

        self.translational_control = None
        self.angular_control = None

        ######################################################
        self.refined_ranges=[];
        self.sat_ranges=[]

        self.largest_gap_index=-1
        self.max_gap=[];
        self.free_space_ranges=[]
        self.selected_angles=[]

        self.sat_pts=1080 * [[0,0]]
        self.pose_x=0
        self.pose_y=0
        self.pose_theta=0

        self.wptutil=WptUtil()
        self.wptutil.loadWpts()
        self.wpts=self.wptutil.wpts
        self.wpt_ref=[self.pose_x,self.pose_y]

        self.pts_free=[]
        ############################################################

    ##############################################################################

    def range2pos(self,pose_x,pose_y,pose_theta,scans):
        angles = []
        distances = []
        lidar_border_points=[]
        max_dist = 0


        for i in range(200-100, 880+100,5): # Only front cone, not all 180 degree (use 0 to 108 for all lidar points)
            index = i # Only use every 10th lidar point

            p1 = pose_x + scans[index] * math.cos(self.lidar_scan_angles[index] + pose_theta)
            p2 = pose_y + scans[index] * math.sin(self.lidar_scan_angles[index] + pose_theta)
            #points.append((p1,p2))
            angles.append(self.lidar_scan_angles[index])
            distances.append(scans[index])
            lidar_border_points.append([50* p1, 50* p2])
            if( scans[index] > max_dist):
                max_dist = scans[index]
        return max_dist,lidar_border_points,angles,distances
    def range2pos_default(self,pose_x,pose_y,pose_theta,scans,scan_angles):
        angles = []
        distances = []
        lidar_border_points=[]
        max_dist = 0

        for i in range(len(scans)):  
            index = i #  
            if not np.isnan(scans[index]) and scans[index]>0:
                p1 = pose_x + scans[index] * math.cos(scan_angles[index] + pose_theta)
                p2 = pose_y + scans[index] * math.sin(scan_angles[index] + pose_theta)
                #points.append((p1,p2))
                angles.append(scan_angles[index])
                distances.append(scans[index])
                lidar_border_points.append([50* p1, 50* p2])
                if( scans[index] > max_dist):
                    max_dist = scans[index]
        return  lidar_border_points 



    ##############################################################################



    def switch_to_safe_mode(self):
        
        return
        print ("There are obstacles on the track: switching to save mode.")
        self.safe_mode = True
        self.speed_fraction= self.safe_mode_speed_fraction
        self.angle_penalty = self.safe_mode_angle_penality

    def check_if_obstacles(self, ranges, gaps):
        if(not self.safe_mode):

            # find number of peaks of raw lidar data for long range obstacle detection
            peaks = sp.find_peaks(ranges, prominence=1)
            peaks = peaks[1]['prominences']
            number_of_peaks = len(peaks)
            if(number_of_peaks > 1):
                print("too many peaks", number_of_peaks)
                self.switch_to_safe_mode()

            if(len(gaps) == self.max_number_of_gaps_for_safe_mode ):
                eucledian_gap_dist = gaps[0][6]

                # Depending on map: 
                # !! Look for minimal gap dist without obstacles and set as thresohold for obstacle detection !!

                # if(eucledian_gap_dist < self.min_gap_dist):
                #     self.min_gap_dist = eucledian_gap_dist
                #     print("min gap dist", self.min_gap_dist)

                if(eucledian_gap_dist < self.min_eucledian_gap_distance_for_safe_mode):
                    print("Gapdist too small, ", eucledian_gap_dist)
                    self.switch_to_safe_mode()

            if(len(gaps) > 1):
                self.switch_to_safe_mode()
                print("Too many gaps", len(gaps) )




    def render(self, e):

        if not self.draw_lidar_data:
             return
        ###########################################################################
        e.wpts=self.wpts #pass information to rendering

        e.wpt_x=50*self.wpt_ref[0]
        e.wpt_y=50*self.wpt_ref[1]
        e.lidar_pts=self.lidar_border_points
        if len(self.max_gap)>0 :
            gap_goal=self.max_gap[2]
            gap_goal_dist=self.max_gap[3];
            px = self.pose_x + gap_goal_dist * math.cos(gap_goal + self.pose_theta)
            py = self.pose_y + gap_goal_dist * math.sin(gap_goal + self.pose_theta)
            line=50*np.asarray([[px,py]])
            line=line.flatten()
            e.goal_x=50*px ## local goal, this will be drawed by the rendring.py
            e.goal_y=50*py
            pts= self.lidar_border_points[self.max_gap[0]:self.max_gap[1]+1]
            e.gap_pts=pts;
            e.max_gap=self.max_gap
            self.pts_free =self.range2pos_default(self.pose_x,self.pose_y,self.pose_theta,self.free_space_ranges,self.selected_angles)
            e.pts_free=self.pts_free

            e.pose_x=50*self.pose_x
            e.pose_y=50*self.pose_y
        #####################################################################


    def process_observation(self, ranges=None, ego_odom=None):
        """
        gives actuation given observation
        @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
        @ ego_odom: A dict with following indices:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
            'linear_vel_x': float,
            'linear_vel_y': float,
            'angular_vel_z': float,
        }
        """

        # @Xiang: I have to turn off your part of the controller for the safe mode. 
        # Apparently the obstacle avoidance already worked with the old controller with speed fraction 1.3 but it didnt after the modification
        # I just use the git commit from then as safemode controller
        if(self.safe_mode):
            return self.safe_planner.process_observation(ranges, ego_odom)

        #######################Xiang ####################################
        steering_angle=0
        speed=0
        self.max_gap=[]
        self.pose_x = ego_odom['pose_x']
        self.pose_y = ego_odom['pose_y']
        self.pose_theta = ego_odom['pose_theta']

        K = 4
        if self.kickdown:
            K = 6

        if True:
            projection, dist, t, min_dist_segment= self.wptutil.find_nearest_point_on_trajectory([self.pose_x,self.pose_y], self.wpts)
            self.wpt_ref, theta2wpt=self.wptutil.get_wpt_ref(self.pose_x,self.pose_y,self.pose_theta,min_dist_segment, K = K)
            self.theta2wpt=theta2wpt
        self.largest_gap_index=-1
        self.max_gap=[];
        self.lidar_border_points = []
        # Take into account size of car
        scans = [max(0.0,x - 0.3) for x in ranges]
        max_dist,self.lidar_border_points,angles,distances=self.range2pos(self.pose_x,self.pose_y,self.pose_theta,scans)
        self.selected_angles=angles


        ####################################################
        ###########################################################

        pose_x = ego_odom['pose_x']
        pose_y = ego_odom['pose_y']
        pose_theta = ego_odom['pose_theta'] 
        angles_unfiltered = angles.copy()
        distances_unfiltered = distances.copy()

        closest_distance = 10000
        closest_distance_index = 0

        # Filter distances

        # ignore close points:
        for i in range(len(distances)):

            if(distances[i] < 3):
                distances[i] = 0

            if (distances[i] > self.lidar_lookahead_distance):
                distances[i] = self.lidar_lookahead_distance

            # Set points near closest distance to 0
            if(distances[i] < closest_distance):
                closest_distance = distances[i]
                closest_distance_index = i

        # IGNORE neighbors of closest point
        for i in range(closest_distance_index - 3, closest_distance_index + 3):
            if(i < len(distances)):
                distances[i] = 0

        # Find gaps
        gaps = []
        gap_open = False
        gap_opening_angle = 0
        gap_starting_index = 0
        gap_treshold = 1.499
        max_distance = 0
        gap_found = False
        gap_integral = 0

        for i in range(len(distances) - 1):
            # Rising
            if(not gap_open):
                if(distances[i] < distances[i+1] - gap_treshold):
                    # + math.sin(0.05) * distances[i]
                    gap_opening_angle = angles[i+1]
                    gap_starting_index = i+1
                    gap_open = True
                if(distances[i+1] > 6):
                    gap_opening_angle = angles[i+1]
                    gap_starting_index = i+1
                    gap_open = True

            # Falling
            if(gap_open):
                gap_integral += distances_unfiltered[i]  # Integrating over gap
                if(max_distance < distances[i]):
                    max_distance = distances[i]

                if(distances[i] > distances[i+1] + gap_treshold):

                    # Find out gap width:
                    gap_width = i - gap_starting_index                  
                    gap_closing_angle = angles[i]
                    gap_closing_index = i

                    # Get eucledian distance of gap (distance from opening point to cloding point)
                    gap_start_point = self.lidar_border_points[gap_starting_index]
                    gap_end_point = self.lidar_border_points[gap_closing_index] 
                    gap_eucledian_distances = math.sqrt(((gap_start_point[0]- gap_end_point[0])**2 )+ ((gap_start_point[1]- gap_end_point[1])**2))


                    # gap: [open angle, closing angle, starting index of distances, closing index of distances, gap integral, gap width, gap_eucledian_distances]
                    gap = [gap_opening_angle,  gap_closing_angle,
                           gap_starting_index, gap_closing_index, gap_integral, gap_width, gap_eucledian_distances]

                    # The gap has to have a certain area that we recognize it as gap (avoid traps)
                    if(gap_integral > 1 and gap_width > 1 and gap_eucledian_distances > 0.6):
                        gaps.append(gap)

                    gap_open = False
                    gap_found = True

        self.lidar_live_gaps = gaps

        # Find largest Gap
        largest_gap_angle = 0
        largest_gap_index = 0
        largest_gap_center = 0
        largest_gap_integral = 0
        largest_gap_width = 0
        for i in range(len(gaps)):
            gap = gaps[i]
            gap_angle = abs(gap[1] - gap[0])
            if(gap_angle) > largest_gap_angle:
                largest_gap_angle = gap_angle
                largest_gap_index = i
                largest_gap_center = (gap[0] + gap[1]) / 2
                largest_gap_integral = gap[4]
                largest_gap_width = gap[5]

        if(self.simulation_index > 50): # Avoid falling in safe mode at initial position
            self.check_if_obstacles(distances_unfiltered, gaps)

        ################## Xiang  #######################
        steering_angle = largest_gap_center
        # You can comment out this line of code for testing #


        steering_angle, max_distance=self.wptutil.suggestGap(gaps,largest_gap_index,distances, angles,steering_angle, max_distance, self.theta2wpt)

        self.max_gap = [0,0,steering_angle,max_distance] #good_gap[2],good_gap[3]

        #########################################

        # Speed Calculation
        speed = self.speed_fraction * max_distance
        if(speed < 0.1):
            speed = 0.1  # Dont stand still 


        speed = speed - self.angle_penalty * abs(steering_angle)

        # print("Speed", speed)
        # Emergency Brake
        if(not gap_found):
            speed = 0.1
            #################
            steering_angle=0.0
            #################
            print("Emergency Brake")

        distances_unfiltered_center_index = int(len(distances_unfiltered)/2)

        if True:
            # Kickdown: If we are on a straight line, we should accelerate faster
            if(not self.safe_mode and distances_unfiltered[distances_unfiltered_center_index] > 11):
                self.kickdown = True
                speed = speed * self.kickdown_factor
                steering_angle = 0.7 * steering_angle
                # If we are on a ong straight line, accelerate even more
                if(distances_unfiltered[distances_unfiltered_center_index] > 20):
                  speed = speed * self.double_kickdiwn_factor
            else:
                self.kickdown = False

            # Kickdown at beginning
            if(not self.safe_mode and self.simulation_index<20):
                speed = 100

        # steering_angle= theta2wpt

        self.simulation_index += 1
        self.translational_control = speed
        self.angular_control = steering_angle
        return speed, steering_angle




class FollowTheGapPlannerSafeMode:
    """
    This is a safe mode planner which kicks in in case of obstacles. 
    From the qualification trials we know that it is able to complete the track with obstacles
    """

    def __init__(self, speed_fraction=1.3):

        print("Safe Controller initialized")

        self.lidar_border_points = 1080 * [[0, 0]]
        self.lidar_live_points = []
        self.lidar_scan_angles = np.linspace(-2.35, 2.35, 1080)
        self.simulation_index = 0
        self.speed_fraction = speed_fraction
        self.plot_lidar_data = False
        self.draw_lidar_data = False
        self.lidar_visualization_color = (0, 0, 0)
        self.lidar_live_gaps = []

        self.kickdown = False
        self.last_steering = [0, 0]
        self.translational_control = None
        self.angular_control = None



    def process_observation(self, ranges=None, ego_odom=None):
        """
        gives actuation given observation
        @ranges: an array of 1080 distances (ranges) detected by the LiDAR scanner. As the LiDAR scanner takes readings for the full 360°, the angle between each range is 2π/1080 (in radians).
        @ ego_odom: A dict with following indices:
        {
            'pose_x': float,
            'pose_y': float,
            'pose_theta': float,
            'linear_vel_x': float,
            'linear_vel_y': float,
            'angular_vel_z': float,
        }
        """
        pose_x = ego_odom['pose_x']
        pose_y = ego_odom['pose_y']
        pose_theta = ego_odom['pose_theta']

        points = []
        angles = []
        distances = []
        self.lidar_border_points = []

        # Take into account size of car
        scans = [x - 0.3 for x in ranges]

        # Use only a part
        max_dist = 0
        # Only front cone, not all 180 degree (use 0 to 108 for all lidar points)
        for i in range(20, 88):
            index = 10*i  # Only use every 10th lidar point

            p1 = pose_x + scans[index] * \
                math.cos(self.lidar_scan_angles[index] + pose_theta)
            p2 = pose_y + scans[index] * \
                math.sin(self.lidar_scan_angles[index] + pose_theta)
            points.append((p1, p2))
            angles.append(self.lidar_scan_angles[index])
            distances.append(scans[index])
            self.lidar_border_points.append([50 * p1, 50 * p2])
            self.lidar_live_points.append([50 * p1, 50 * p2])
            if(scans[index] > max_dist):
                max_dist = scans[index]

        angles_unfiltered = angles.copy()
        distances_unfiltered = distances.copy()

        closest_distance = 10000
        closest_distance_index = 0

        # Filter distances

        # ignore close points:
        for i in range(len(distances)):

            if(distances[i] < 3):
                distances[i] = 0

            if (distances[i] > 6):
                distances[i] = 6

            # Set points near closest distance to 0
            if(distances[i] < closest_distance):
                closest_distance = distances[i]
                closest_distance_index = i

        # IGNORE neighbors of closest point
        for i in range(closest_distance_index - 3, closest_distance_index + 3):
            if(i < len(distances)):
                distances[i] = 0

        # Find gaps
        gaps = []
        gap_open = False
        gap_opening_angle = 0
        gap_starting_index = 0
        gap_treshold = 1.499
        max_distance = 0
        gap_found = False
        gap_integral = 0

        for i in range(len(distances) - 1):
            # Rising
            if(not gap_open):
                if(distances[i] < distances[i+1] - gap_treshold):
                    # + math.sin(0.05) * distances[i]
                    gap_opening_angle = angles[i+1]
                    gap_starting_index = i+1
                    gap_open = True
                if(distances[i+1] > 6):
                    gap_opening_angle = angles[i+1]
                    gap_starting_index = i+1
                    gap_open = True

            # Falling
            if(gap_open):
                gap_integral += distances[i]  # Integrating over gap
                if(max_distance < distances[i]):
                    max_distance = distances[i]

                if(distances[i] > distances[i+1] + gap_treshold):

                    # Find out gap width:
                    gap_width = i - gap_starting_index
                    # print("gap_width",gap_width)
                    # if(gap_width > 2):
                    # - math.sin(0.05) * distances[i]
                    gap_closing_angle = angles[i]
                    gap_closing_index = i

                    # gap: [open angle, closing angle, starting index of distances, closing index of distances, gap integral, gap width]
                    gap = [gap_opening_angle,  gap_closing_angle,
                           gap_starting_index, gap_closing_index, gap_integral, gap_width]

                    # The gap has to have a certain area that we recognize it as gap (avoid traps)
                    if(gap_integral > 30):
                        gaps.append(gap)

                    gap_open = False
                    gap_found = True

        self.lidar_live_gaps = gaps

        # Find largest Gap
        largest_gap_angle = 0
        largest_gap_index = 0
        largest_gap_center = 0
        largest_gap_integral = 0
        largest_gap_width = 0
        for i in range(len(gaps)):
            gap = gaps[i]
            gap_angle = abs(gap[1] - gap[0])
            if(gap_angle) > largest_gap_angle:
                largest_gap_angle = gap_angle
                largest_gap_index = i
                largest_gap_center = (gap[0] + gap[1]) / 2
                largest_gap_integral = gap[4]
                largest_gap_width = gap[5]

        # Speed Calculation
        speed = self.speed_fraction * max_distance
        if(speed < 0.1):
            speed = 0.1  # Dont stand still



        speed = speed - 8 * abs(largest_gap_center)
        # print("Speed", speed)
        # Emergency Brake
        if(not gap_found):
            speed = 0.1
            print("Emergency Brake")

        # Only front cone, not all 180 degree (use 0 to 108 for all lidar points)
        for i in range(50, 58):
            index = 10*i  # Only use every 10th lidar point
            if(scans[index] < 0.5):
                speed = 0.0
                print("Emergency Brake: Obstacle in front")


        steering_angle = largest_gap_center

        self.simulation_index += 1
        self.translational_control = speed
        self.angular_control = steering_angle
        return speed, steering_angle