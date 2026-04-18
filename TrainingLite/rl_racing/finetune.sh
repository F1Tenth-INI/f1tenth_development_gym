
# bash TrainingLite/rl_racing/finetune.sh


# # Evaluate (informed) MPC performance on the original car dynamics
# python run.py --CONTROLLER rpgd-lite-jax --ENV_CAR_PARAMETER_FILE gym_car_parameters.yml --CONTROLLER_CAR_PARAMETER_FILE gym_car_parameters.yml --SAVE_RECORDINGS True --SAVE_VIDEOS True
# # Baseline mpc: 20.7s

# # Evaluate (informed) MPC performance on the dinetune car dynamics
# python run.py --CONTROLLER rpgd-lite-jax --ENV_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml --CONTROLLER_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml --SAVE_RECORDINGS True --SAVE_VIDEOS True
# # Baseline MPC: 24.3


# # original ( base model ) training
# python TrainingLite/rl_racing/run_training.py \
#   --auto-start-client \
#   --CONTROLLER sac_agent \
#   --SAVE_RECORDINGS False \
#   --MAP_NAME RCA1 \
#   --SAC_TARGET_UDT 0.25 \
#   --batch-size 64 \
#   --SAC_CHECKPOINT_FREQUENCY 50000 \
#   --SIMULATION_LENGTH 250000 \
#   --MAX_SIM_FREQUENCY 250 \
#   --ENV_CAR_PARAMETER_FILE gym_car_parameters.yml \
#   --save_replay_buffer False \
#   --save-model-name RCA1-1 \
#   --USE_CUSTOM_SAC_SAMPLING False \
  

# finetune model in different car dynamics
python TrainingLite/rl_racing/run_training.py \
  --auto-start-client \
  --CONTROLLER sac_agent \
  --SAVE_RECORDINGS False \
  --MAP_NAME RCA1 \
  --SAC_TARGET_UDT 0.25 \
  --batch-size 64 \
  --SAC_CHECKPOINT_FREQUENCY 50000 \
  --SIMULATION_LENGTH 50000 \
  --MAX_SIM_FREQUENCY 200 \
  --ENV_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml \
  --save_replay_buffer True \
  --load-model-name RCA1-1 \
  --save-model-name RCA1-1c_moreState08 \
  --USE_CUSTOM_SAC_SAMPLING True \

python TrainingLite/rl_racing/run_training.py \
  --auto-start-client \
  --CONTROLLER sac_agent \
  --SAVE_RECORDINGS False \
  --MAP_NAME RCA1 \
  --SAC_TARGET_UDT 0.25 \
  --batch-size 64 \
  --SAC_CHECKPOINT_FREQUENCY 50000 \
  --SIMULATION_LENGTH 50000 \
  --MAX_SIM_FREQUENCY 200 \
  --ENV_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml \
  --save_replay_buffer True \
  --load-model-name RCA1-1 \
  --save-model-name RCA1-1c_noCustom \
  --USE_CUSTOM_SAC_SAMPLING False \