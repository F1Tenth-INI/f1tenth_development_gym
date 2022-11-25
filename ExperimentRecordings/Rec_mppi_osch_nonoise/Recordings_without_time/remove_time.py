import pandas as pd

'''
A File to Remove the time column
'''

FILENAME = 'F1TENTH_Oschers-MPPI-simple-9__2022-11-14_17-18-18'
PATH_WAYPOINT_FILE = 'SI_Toolkit_ASF/Experiments/Experiment-MPPI-Imitator/Recordings/Train/'
general_path = '/home/marcin/PycharmProjects/f1tenth_development_gym_Jago/'

training_data = pd.read_csv(general_path + PATH_WAYPOINT_FILE + FILENAME + '.csv', sep=',', header=8)                          #import csv as pd dataframe
training_data = training_data.iloc[:,1:]                                                                                               #remove first column
pd.DataFrame(training_data).to_csv(general_path + PATH_WAYPOINT_FILE + FILENAME + '_timeless.csv', header = 8, index = False)           #export pd df to csv
