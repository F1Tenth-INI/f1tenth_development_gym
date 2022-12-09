import pandas as pd

'''
A File to Remove every nth line of a waypoint file to make it leaner
'''

FILENAME = 'example_waypoints_adapted'
PATH_WAYPOINT_FILE = 'utilities/maps_files/waypoints/'
general_path = '/home/marcin/PycharmProjects/f1tenth_development_gym_Jago/'

waypoints = pd.read_csv(general_path + PATH_WAYPOINT_FILE + FILENAME + '.csv', header=None).to_numpy()                          #import csv as np array
waypoints = waypoints[0:-1:2]                                                                                                   #only every second/third line is kept
pd.DataFrame(waypoints).to_csv(general_path + PATH_WAYPOINT_FILE + FILENAME + '_lean.csv', header = False, index = False)       #export pd df to csv




