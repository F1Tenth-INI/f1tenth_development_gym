# Neural Imitator

## Training
Collect experiment recordings with a controller of choise (fe. MPC - MPPI).
 - For a higher variance data, set control noise up to 0.5

NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.5 
NOISE_LEVEL_ANGULAR_CONTROL = 0.5  

 - Tune the controller for robustness ( it needs to be able to complete laps reliably )
 - Delete all old experiment recordings in ExperimentRecordings/
 - Set EXPERIMENT_LENGTH such that the car completes more than 2 laps
 - set NUMBER_OF_EXPERIMENTS >= 10 depending on how much data you want to have
 - Run experiments

 - Create the following folders: 
    - SI_Toolkit_ASF/Experiments/[Controller Name]/Recordings/Train
    - SI_Toolkit_ASF/Experiments/[Controller Name]/Recordings/Test
    - SI_Toolkit_ASF/Experiments/[Controller Name]/Recordings/Validate
 - Distribute the experiment's CSV files into these 3 folders ( each 80%, 10%, 10% of the data points)
 - in config_training.yml set path_to_experiment to [Controller Name]

 - Create normalization file:
```bash
python SI_Toolkit_ASF/run/Create_normalization_file.py
```
 - Check the histograms if training data makes sense
 - in config_training.yml set NET_NAME, inputs and training settings
 - Train Network:
```bash
 python SI_Toolkit_ASF/run/Train_Network.py 
```
 - create a file at SI_Toolkit_ASF/Experiments/[Controller Name]/Models/[Model Name]/notes.txt and write minimal documentation about the network (Maps, Controller, Settings, thoughts etc...)
 Congratulations, the Neural Controller is now ready to use.

 ## Run Neural Imitator
- in Settings.py, select the neural controller and the model name
- Deactivate the control noise
- Deactivate control averaging (NN does not like it )
```python
CONTROLLER = 'neural'
...
PATH_TO_MODELS = 'SI_Toolkit_ASF/Experiments/[Controller Name]/Models/'
NET_NAME = '[Model Name]'
...
NOISE_LEVEL_TRANSLATIONAL_CONTROL = 0.0
NOISE_LEVEL_ANGULAR_CONTROL = 0.0  
...
CONTROL_AVERAGE_WINDOW = (1, 1)
...
```
- Make sure that the control_inputs in config_training.yml and nni_planner.py match. (Otherwise correct them in nni_planner)
- Run experiment
Enjoy your realtime neural network MPPI imipator ( or how we call it: the INItator ).

## Brunton Test
Check config_testting.yml: 
 - Select a file in experiment recordings for reference
 - Select the network you want to test
```bash
python SI_Toolkit_ASF/run/Run_Brunton_Test.py 
```
