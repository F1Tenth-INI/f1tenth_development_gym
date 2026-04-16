
# Evaluate (informed) MPC performance on the original car dynamics
python run.py --CONTROLLER rpgd-lite-jax --ENV_CAR_PARAMETER_FILE gym_car_parameters.yml --CONTROLLER_CAR_PARAMETER_FILE gym_car_parameters.yml
 
# Evaluate (informed) MPC performance on the dinetune car dynamics
python run.py --CONTROLLER rpgd-lite-jax --ENV_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml --CONTROLLER_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml

# original ( base model ) training
python TrainingLite/rl_racing/run_training.py \
  --auto-start-client \
  --CONTROLLER sac_agent \
  --SAVE_RECORDINGS False \
  --MAP_NAME RCA1 \
  --SAC_TARGET_UDT 1.0 \
  --batch-size 64 \
  --SAC_CHECKPOINT_FREQUENCY 50000 \
  --SIMULATION_LENGTH 250000 \
  --MAX_SIM_FREQUENCY 250 \
  --ENV_CAR_PARAMETER_FILE gym_car_parameters.yml \
  --save_replay_buffer False \
  --save-model-name RCA1-1
  

# finetune model in different car dynamics
python TrainingLite/rl_racing/run_training.py \
  --auto-start-client \
  --CONTROLLER sac_agent \
  --SAVE_RECORDINGS False \
  --MAP_NAME RCA1 \
  --SAC_TARGET_UDT 1.0 \
  --batch-size 64 \
  --SAC_CHECKPOINT_FREQUENCY 50000 \
  --SIMULATION_LENGTH 50000 \
  --MAX_SIM_FREQUENCY 250 \
  --ENV_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml \
  --save_replay_buffer True \
  --load-model-name RCA1-1 \
  --save-model-name RCA1-1a