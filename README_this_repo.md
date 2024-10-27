To use this repo and replicate my results or to work further with this repo you can use this short guide.
All the runs can be done in terminal.

If you are running from terminal, please run all python scripts from the project's root folder. You might want to export the Python Path env variable:

```bash
export PYTHONPATH=./
```

To test any type of controller on a specific map, you can run the following code.

```bash
python run.py
```

To be able to modify your controller, you can change the Settings.py file.
This file is located in ./utilities/Settings.py

If you want to train a neural imitator, there is a procedere, which you can follow to train the network.
IMPORTANT to maintain. Please setup everything like mentioned in the README.md with the working enviroment and with the submodules !!!

First you have to collect training data, which can be done by running several times the run.py code or we made a script, which runs with the desired settings (different velocities, adding noise, different surface_frictions, etc.)

For running the data_collection.py script, make sure that your map matches the same as the in the Settings.py. After you modified the data_collection.py file you can run it like:

```bash
python run/data_collection.py
```

This command creates in ./ExperimentRecordings folder the training data with the corresponding configs by default.

If you take a look into the training data and see that the mu doesn't match your configs, then you can run the script:

```bash
python run/modify_csv.py
```

This script modifys the desired mu to the value you wanted to like and set in the file.

If the training data looks good, you can modify the data in a further step to introduce a delay. This can be done by shifting the applied controls with the corresponding timesteps. If the delay is constant this command can be scripted and can be directly done during training process by enabling in the config_training.yml the shift_labels by any integer number. But if you want to have different delays inside a training process the training data has to be manipulated. This can be done with the script:

```bash
python run/shift_csvfiles.py
```

After the training data is ready in the ./ExperimentRecodings folder if everthing is let default, then you can distribute the files as you like into the training pipeline of the SI_Toolit. This is explained well in the README.md but here is a short recap of it. To distribute with a certain probability you like, you can use the script:

```bash
python run/experiment_data_distribution.py
```

After the distribution you will find it in SI_Toolkit_ASF folder, where you set the location in the script experiment_data_distribution.py.

Here a short recap of SI_Toolkit training.

First set every parameter in the file ./SI_Toolkit_ASF/config_training.yml.
Important parameter to change:
- NET_NAME: 
- PATH_TO_EXPERIMENT_FOLDERS
- control_inputs
- SHIFT_LABELS  As described before instead of running the shift_csvfiles.py

After setting all the params, you can run the following script, which will normalize through all the training data you created for training.

```bash
python SI_Toolkit_ASF/run/Create_normalization_file.py
```
After this the training can be started with the command:

```bash
python SI_Toolkit_ASF/run/Train_Network.py
```

Here you go, you have then a model with your desired settings. If you want to run the trained model and test if it can complete a track, then you have to modify some files:

1. Set in ./Control_Toolkit_ASF/config_controllers.yml the PATH_TO_MODELS and net_name as it is in your Setting.

2. Then change in the ./utilities/Settings.py the parameter CONTROLLER = 'neural' and you can test your trained network :)