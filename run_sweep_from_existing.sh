#!/bin/bash

# ALPHAS=(0.4 0.6 0.8) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
# BETAS=(0.2 0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
# RATIOS=(0.0 0.25 0.5) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only

ALPHAS=(0.0 0.6) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
NSTEPS=(1)
# CURRICULUM_START=(0.5 0.3 0.7)
CURRICULUM_T2=(0.5 0.8)

# SOURCE_MODEL="Example-1"
# SOURCE_MODEL_SHORT="Ex1"
# NEW_MAP_NAME="RCA2"

SOURCE_MODEL="Curriculum1PC"
SOURCE_MODEL_SHORT="Cur1"
NEW_MAP_NAME="RCA2"
CURRICULUM_START=0.5

# Loop
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for t2 in "${CURRICULUM_T2[@]}"; do
        for index in 1 2 3; do
          # MODEL_NAME="Sweep_Cur_from_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_CUR${CURRICULUM_T2}"
          MODEL_NAME="Sweep_BETTER_Cur_from_${SOURCE_MODEL_SHORT}_A${alpha}_CUR_T2${t2}_Run${index}"
        
          # MODEL_NAME="Sweep_nstep_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_N${}"
          
          echo "=================================================="
          echo " STARTING: $MODEL_NAME"
          echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
          echo "=================================================="

          python -u TrainingLite/rl_racing/run_training.py \
            --auto-start-client \
            --USE_CUSTOM_SAC_SAMPLING True \
            --device cpu \
            --SIMULATION_LENGTH 50000 \
            --load-model-name "$SOURCE_MODEL" \
            --save-model-name "$MODEL_NAME" \
            --MAP_NAME "$NEW_MAP_NAME" \
            --alpha $alpha \
            --beta_start $beta \
            --td_ratio $ratio \
            --SAC_SPEED_CURRICULUM_LEARNING True \
            --SAC_CURRICULUM_T2 $t2 \

          sleep 5
        done
      done
    done
  done
done
 
for index in 1 2 3 4 5 6; do
  MODEL_NAME="Sweep_BETTER_Cur_from_${SOURCE_MODEL_SHORT}_A0.0_CUR_T20.0__Run${index}"
  echo "=================================================="
  echo " STARTING: $MODEL_NAME"
  echo " Alpha: 0.0 | Beta: 0.4 | Ratio: 0.0"
  echo "=================================================="

  python -u TrainingLite/rl_racing/run_training.py \
    --auto-start-client \
    --USE_CUSTOM_SAC_SAMPLING True \
    --device cpu \
    --SIMULATION_LENGTH 50000 \
    --load-model-name "$SOURCE_MODEL" \
    --save-model-name "$MODEL_NAME" \
    --MAP_NAME "$NEW_MAP_NAME" \
    --alpha 0.0 \
    --beta_start 0.4 \
    --td_ratio 0.0 \
    --SAC_SPEED_CURRICULUM_LEARNING False \
    --SAC_CURRICULUM_T2 $CURRICULUM_START \

  sleep 5
done