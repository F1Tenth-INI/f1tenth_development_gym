
  # original ( base model ) training
python TrainingLite/rl_racing/run_training.py \
  --auto-start-client \
  --CONTROLLER sac_agent \
  --SAVE_RECORDINGS False \
  --MAP_NAME TheTrack1 \
  --SAC_TARGET_UTD 1.0 \
  --batch-size 64 \
  --SAC_CHECKPOINT_FREQUENCY 50000 \
  --SIMULATION_LENGTH 200000 \
  --MAX_SIM_FREQUENCY 250 \
  --RANDOM_WAYPOINT_VEL_FACTOR True \
  --save_replay_buffer False \
  --save-model-name Physical-42


# # finetune model in different car dynamics
python TrainingLite/rl_racing/run_training.py \
  --SAC_MAX_UTD 4 \
  --batch-size 256 \
  --learning-rate 3e-4 \
  --SAC_CHECKPOINT_FREQUENCY 10000 \
  --save_replay_buffer True \
  --load_replay_buffer False \
  --load-model-name Physical-42 \
  --save-model-name Physical-42b