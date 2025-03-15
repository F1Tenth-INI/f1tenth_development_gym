# RL Racing

This is an example for an RL training thats able to race in our F1TENTH gym. The whole simulator is wrapped in a new Gymnasium env called RacingEnv.
RacingEnv.simulator has access to all data during the simulation, like car_state, waypoint_utils, lidar, etc.

## Training

In Settings.py set OPTIMIZE_FOR_RL = True
Then run the training script:
'''console
python TrainingLite/rl_racing/train_model.py
'''

Checkout the RewardCalculator.py for the reward function.

## Tensorboard

'''console
tensorboard --logdir=TrainingLite/rl_racing/models/#EXPERIMENT_NAME#/tensorboard_logs/
'''
