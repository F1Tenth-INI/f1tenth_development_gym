#!/bin/bash

# ALPHAS=(0.4 0.6 0.8) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
# BETAS=(0.2 0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
# RATIOS=(0.0 0.25 0.5) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only

ALPHAS=(0.0 0.4 0.8) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0 0.5 1.0) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
NSTEPS=(1)
# CURRICULUM_START=(0.5 0.3 0.7)
CURRICULUM_T2=(0.6 1.0)
BASELINE_T2=0.8

SOURCE_MODEL="Example-1"
SOURCE_MODEL_SHORT="Ex1"
NEW_MAP_NAME="RCA2"

# SOURCE_MODEL="Curriculum1PC"
# SOURCE_MODEL_SHORT="Cur1"
# NEW_MAP_NAME="RCA2"

CURRCULUM_START=0.5

# CURRICULUM_TYPE=()

USE_CUSTOM_SAMPLING=True
USE_SPEED_CURRICULUM_LIST=(True False)

RANK_BASED_LIST=(False)

LOG_SQUISH_LIST=(True False)

IS_WEIGHTS_FOR_ACTOR_LIST=(True False)

# for alpha in "${ALPHAS[@]}"; do
#   for rank_based in "${RANK_BASED_LIST[@]}"; do
#     for index in 1 2; do
#       # MODEL_NAME="Sweep_Cur_from_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_CUR${CURRICULUM_T2}"
#       MODEL_NAME="2002_from_${SOURCE_MODEL_SHORT}_A${alpha}_Rank_${rank_based}_Run${index}"
    
#       # MODEL_NAME="Sweep_nstep_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_N${}"
      
#       echo "=================================================="
#       echo " STARTING: $MODEL_NAME"
#       echo " Alpha: $alpha | Rank Based: $rank_based"
#       echo "=================================================="

#       python -u TrainingLite/rl_racing/run_training.py \
#         --auto-start-client \
#         --USE_CUSTOM_SAC_SAMPLING "$USE_CUSTOM_SAMPLING" \
#         --device cpu \
#         --SIMULATION_LENGTH 75000 \
#         --load-model-name "$SOURCE_MODEL" \
#         --save-model-name "$MODEL_NAME" \
#         --MAP_NAME "$NEW_MAP_NAME" \
#         --alpha $alpha \
#         --SAC_RANK_BASED_SAMPLING $rank_based \

#       sleep 5
#     done
#   done
# done

# Loop
for alpha in "${ALPHAS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    for cur_t2 in "${CURRICULUM_T2[@]}"; do
      # MODEL_NAME="Sweep_Cur_from_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_CUR${CURRICULUM_T2}"
      MODEL_NAME="2602_${SOURCE_MODEL_SHORT}_A${alpha}_R_${ratio}_T2_${cur_t2}"
    
      # MODEL_NAME="Sweep_nstep_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_N${}"
      
      echo "=================================================="
      echo " STARTING: $MODEL_NAME"
      echo " Alpha: $alpha | Cur T2: $cur_t2 | Ratio: $ratio"
      echo "=================================================="

      cmd=(python -u TrainingLite/rl_racing/run_training.py
        --auto-start-client
        --USE_CUSTOM_SAC_SAMPLING True
        --device cpu
        --SIMULATION_LENGTH 75000
        --load-model-name "$SOURCE_MODEL"
        --save-model-name "$MODEL_NAME"
        --MAP_NAME "$NEW_MAP_NAME"
        --alpha "$alpha"
        --td_ratio "$ratio"
        --SAC_CURRICULUM_T2 "$cur_t2"
        --replay-capacity 50000
      )
      "${cmd[@]}"
      sleep 5
    done
  done
done

# CAR_STATE_NOISE="[0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]"
# CONTROL_NOISE="[0.35, 0.7]"

# USE_SPEED_CURRICULUM_LIST=(True True True False False)
# SPEED_MODE=("speed_cap" "vel_factor" "speed_cap" None None)
# USE_WIDTH_CUR=(False False True True False)
# USE_NOISE_CUR=(False False False False True)


# for main_index in 1 2 3 4 5; do
#   for list_index in 0 1 2 3 4; do
#     MODEL_NAME="30Jan_Sweep_S_${SPEED_MODE[$list_index]}_W_${USE_WIDTH_CUR[$list_index]}_N_${USE_NOISE_CUR[$list_index]}_Run${main_index}"
#     echo "=================================================="
#     echo " STARTING: $MODEL_NAME"
#     echo " Alpha: 0.0 | Beta: 0.4 | Ratio: 0.0"
#     echo "=================================================="

#     python -u TrainingLite/rl_racing/run_training.py \
#       --auto-start-client \
#       --USE_CUSTOM_SAC_SAMPLING False \
#       --device cpu \
#       --SIMULATION_LENGTH 75000 \
#       --load-model-name "$SOURCE_MODEL" \
#       --save-model-name "$MODEL_NAME" \
#       --MAP_NAME "$NEW_MAP_NAME" \
#       --alpha 0.0 \
#       --beta_start 0.4 \
#       --td_ratio 0.0 \
#       --SAC_CURRICULUM_SPEED "${USE_SPEED_CURRICULUM_LIST[$list_index]}" \
#       --SAC_CURRICULUM_SPEED_ADJUST_MODE "${SPEED_MODE[$list_index]}" \
#       --SAC_CURRICULUM_SPEED_LIMIT_MAX 15 \
#       --SAC_CURRICULUM_SPEED_LIMIT 15 \
#       --SAC_CURRICULUM_TRACK_WIDTH_SCALING "${USE_WIDTH_CUR[$list_index]}" \
#       --SAC_CURRICULUM_TRACK_WIDTH_FACTOR 1.0 \
#       --SAC_CURRICULUM_NOISE_SCALING "${USE_NOISE_CUR[$list_index]}" \
#       --SAC_NOISE_LEVEL_CAR_STATE_MAX "$CAR_STATE_NOISE" \
#       --SAC_NOISE_LEVEL_CONTROL_MAX "$CONTROL_NOISE" \

#     sleep 5
#   done
# done