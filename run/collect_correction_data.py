""" Run the race! """
from run.run_simulation import RacingSimulation
from utilities.Settings import Settings

import numpy as np
if __name__ == '__main__':
    
    Settings.EXPERIMENT_LENGTH = 600
    Settings.NUMBER_OF_EXPERIMENTS = 1
    Settings.DATASET_NAME = "correction"
    Settings.CONTROLLER = 'mpc'
    Settings.SAVE_RECORDINGS = True
    Settings.SAVE_PLOTS = False    
    
    # Read initial state from Test.csv
    initial_state = np.genfromtxt('Test.csv', delimiter=',')

    # create multiple copies of initial states with noise
    initial_states = np.tile(initial_state, (10, 1))
    noise = np.random.normal(0, 0.1, initial_states.shape)
    initial_states += noise
    # print(initial_states)
    
    for i in range(len(initial_states)):
        simulation = RacingSimulation()
        simulation.run_experiments(initial_states=[initial_state])