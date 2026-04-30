#Original
cd gym_bridge/f1tenth_development_gym/


python3 TrainingLite/rl_racing/run_training.py \
  --SAC_MAX_UTD 4 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --SAC_CHECKPOINT_FREQUENCY 10000 \
  --save_replay_buffer True \
  --load_replay_buffer False \
  --load-model-name Physical-20 \
  --save-model-name Physical-20b


  # Nikita 1st try
  python3 TrainingLite/rl_racing/run_training.py \
                  --save_replay_buffer True \
                  --SAC_MAX_UTD 4 \
                  --batch-size 64 \
                  --SAC_CHECKPOINT_FREQUENCY 5000 \
                  --load-model-name Physical-20\
                  --save-model-name Physical-20b_TD_A_06 \
                  --USE_CUSTOM_SAC_SAMPLING True \
                  --SAC_USE_BATCH_RECENCY_PRIORITIZATION False \
                  --SAC_RECENCY_ONLY_MODE False \
                  --SAC_CUSTOM_ACTOR_INVERT_TD False \
                  --SAC_PRIORITY_FACTOR 0.6 \
                  --SAC_STATE_TO_TD_RATIO 0.0 \
                  --SAC_CUSTOM_UNIFORM_CRITIC False \
                  --SAC_CUSTOM_SEPARATE_BATCHES_ACTOR_CRITIC False

# 2nd run
  python3 TrainingLite/rl_racing/run_training.py \
                  --save_replay_buffer True \
                  --SAC_MAX_UTD 4 \
                  --batch-size 64 \
                  --SAC_CHECKPOINT_FREQUENCY 5000 \
                  --load-model-name Physical-20\
                  --save-model-name Physical-20-CustomUniform-a \
                  --USE_CUSTOM_SAC_SAMPLING True \
                  --SAC_USE_BATCH_RECENCY_PRIORITIZATION False \
                  --SAC_RECENCY_ONLY_MODE False \
                  --SAC_CUSTOM_ACTOR_INVERT_TD False \
                  --SAC_PRIORITY_FACTOR 0.0 \
                  --SAC_STATE_TO_TD_RATIO 0.0 \
                  --SAC_CUSTOM_UNIFORM_CRITIC False \
                  --SAC_CUSTOM_SEPARATE_BATCHES_ACTOR_CRITIC False

# 3rd run

  python3 TrainingLite/rl_racing/run_training.py \
                  --save_replay_buffer True \
                  --SAC_MAX_UTD 4 \
                  --batch-size 64 \
                  --SAC_CHECKPOINT_FREQUENCY 5000 \
                  --load-model-name Physical-20\
                  --save-model-name Physical-20-CustomUniform-a \
                  --USE_CUSTOM_SAC_SAMPLING True \
                  --SAC_USE_BATCH_RECENCY_PRIORITIZATION False \
                  --SAC_RECENCY_ONLY_MODE False \
                  --SAC_CUSTOM_ACTOR_INVERT_TD False \
                  --SAC_PRIORITY_FACTOR 0.0 \
                  --SAC_STATE_TO_TD_RATIO 0.0 \
                  --SAC_CUSTOM_UNIFORM_CRITIC False \
                  --SAC_CUSTOM_SEPARATE_BATCHES_ACTOR_CRITIC False





REWARD_WEIGHT_LIST_ALL=(10.0)
  D_WEIGHT_LIST_ALL=(10.0) # lateral offset
  E_WEIGHT_LIST_ALL=(3.0) # heading error

python3 TrainingLite/rl_racing/run_training.py \
    --save_replay_buffer False \
    --SAC_TARGET_UDT 4 \
    --batch-size 64 \
    --SAC_CHECKPOINT_FREQUENCY 5000 \
    --load-model-name Physical-20 \
    --save-model-name Physical-20-State_Wrew_10_Wd_10_We_3_fixed \
    --USE_CUSTOM_SAC_SAMPLING True \
    --SAC_USE_BATCH_RECENCY_PRIORITIZATION False \
    --SAC_RECENCY_ONLY_MODE False \
    --SAC_CUSTOM_ACTOR_INVERT_TD False \
    --SAC_PRIORITY_FACTOR 0.6 \
    --SAC_STATE_TO_TD_RATIO 0.6 \
    --SAC_REWARD_WEIGHT 10.0 \
    --SAC_WP_OFFSET_WEIGHT 10.0 \
    --SAC_WP_HEADING_ERROR_WEIGHT 3.0 \
    --SAC_CUSTOM_UNIFORM_CRITIC False \
    --SAC_CUSTOM_SEPARATE_BATCHES_ACTOR_CRITIC True \
    --SAC_CRITIC_PURE_TD True





python3 TrainingLite/rl_racing/run_training.py \
                  --save_replay_buffer True \
                  --SAC_MAX_UTD 4 \
                  --batch-size 64 \
                  --SAC_CHECKPOINT_FREQUENCY 5000 \
                  --load-model-name Physical-20\
                  --save-model-name Physical-20c_TD_A_06 \
                  --USE_CUSTOM_SAC_SAMPLING True \
                  --SAC_USE_BATCH_RECENCY_PRIORITIZATION False \
                  --SAC_RECENCY_ONLY_MODE False \
                  --SAC_CUSTOM_ACTOR_INVERT_TD False \
                  --SAC_PRIORITY_FACTOR 0.6 \
                  --SAC_STATE_TO_TD_RATIO 0.0 \
                  --SAC_CUSTOM_UNIFORM_CRITIC False \
                  --SAC_CUSTOM_SEPARATE_BATCHES_ACTOR_CRITIC False



python3 TrainingLite/rl_racing/run_training.py \
                  --save_replay_buffer True \
                  --SAC_MAX_UTD 4 \
                  --batch-size 64 \
                  --SAC_CHECKPOINT_FREQUENCY 5000 \
                  --load-model-name Physical-20\
                  --save-model-name Physical-20d_TD_A_06 \
                  --USE_CUSTOM_SAC_SAMPLING True \
                  --SAC_USE_BATCH_RECENCY_PRIORITIZATION False \
                  --SAC_RECENCY_ONLY_MODE False \
                  --SAC_CUSTOM_ACTOR_INVERT_TD False \
                  --SAC_PRIORITY_FACTOR 0.6 \
                  --SAC_STATE_TO_TD_RATIO 0.0 \
                  --SAC_CUSTOM_UNIFORM_CRITIC False \
                  --SAC_CUSTOM_SEPARATE_BATCHES_ACTOR_CRITIC False