""" Run the race! """
import os
import numpy as np

from utilities.Settings import Settings

def run_experiment():

    Settings.EXPERIMENT_LENGTH = 30000
    Settings.NUMBER_OF_EXPERIMENTS = 1
    Settings.DATASET_NAME = "trial"
    Settings.CONTROLLER = 'nni-lite'
    Settings.SAVE_RECORDINGS = True
    Settings.SAVE_PLOTS = True    
    Settings.RENDER_MODE = None
    Settings.RECORDING_FOLDER = './ExperimentRecordings/'

    
    from run.run_simulation import RacingSimulation
    simulation = RacingSimulation()
    simulation.run_experiments()


def collect_correction_data():
    from utilities.Settings import Settings
    from run.run_simulation import RacingSimulation

    Settings.EXPERIMENT_LENGTH = 600
    Settings.NUMBER_OF_EXPERIMENTS = 1
    Settings.DATASET_NAME = "correction"
    Settings.CONTROLLER = 'mpc'
    Settings.SAVE_RECORDINGS = True
    Settings.SAVE_PLOTS = False    
    Settings.RENDER_MODE = None

    Settings.RECORDING_FOLDER = Settings.RECORDING_FOLDER = './ExperimentRecordings/correction/'

    
    # Read initial state from Test.csv
    initial_state = np.genfromtxt('Test.csv', delimiter=',')

    # create multiple copies of initial states with noise
    number_of_initializations = 3
    initial_states = np.tile(initial_state, (number_of_initializations, 1))
    noise = np.random.normal(0, 0.1, initial_states.shape)
    initial_states += noise
    # print(initial_states)
    
    for i in range(len(initial_states)):
        simulation = RacingSimulation()
        try:
            simulation.run_experiments(initial_states=[initial_state])
        except Exception as e:
            print(f"Error during experiment {i}: {e}")

def finetune_network():
    from TrainingLite.mpc_immitator_mu.torch_train import ImmitationTraining
    model_name = "04_08_RCA1_noise"
    dataset_name = "correction"
    immitation_training = ImmitationTraining(model_name, dataset_name)
    immitation_training.load_model()
    immitation_training.load_and_normalize_dataset()
    immitation_training.train_network()


if __name__ == '__main__':


    for i in range(100):
        print(f"Running experiment {i+1}...")
        try:
            run_experiment()
        except Exception as e:
            print(f"Error during experiment: {e}")

        print("Collecting correction data...")
        collect_correction_data()
        print("Data collection complete.")
        print("Retraining network...")

        # Move all files form ExperimentData/correction/ to TrainingLite/Datasets/correction/
        if not os.path.exists('TrainingLite/Datasets/correction/'):
            os.makedirs('TrainingLite/Datasets/correction/')
        # Move files
        os.system('mv ExperimentRecordings/correction/* TrainingLite/Datasets/correction/')


        finetune_network()
        # print("Network retraining complete.")
