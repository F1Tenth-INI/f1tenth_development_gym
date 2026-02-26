#!/bin/bash

ALPHAS=(0.0) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
# NSTEPS=(1 2 3 5)
CURRICULUM=(True False)


# Loop
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for cur in "${CURRICULUM[@]}"; do
      
        MODEL_NAME="Sweep_BETTER_fresh_cur_A${alpha}_B${beta}_Cur${cur}"
        
        echo "=================================================="
        echo " STARTING: $MODEL_NAME"
        echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
        echo "=================================================="

        cmd=(python -u TrainingLite/rl_racing/run_training.py
          --auto-start-client
          --USE_CUSTOM_SAC_SAMPLING True
          --device cpu
          --SIMULATION_LENGTH 250000
          --model-name "$MODEL_NAME"
          --alpha "$alpha"
          --beta_start "$beta"
          --td_ratio "$ratio"
          --SAC_SPEED_CURRICULUM_LEARNING "$cur"
        )
        "${cmd[@]}"
        sleep 5
      done
    done
  done
done


for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for cur in "${CURRICULUM[@]}"; do
      
        MODEL_NAME="Sweep_BETTER_fresh_cur_A${alpha}_B${beta}_Cur${cur}_pt2"
        
        echo "=================================================="
        echo " STARTING: $MODEL_NAME"
        echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
        echo "=================================================="

        cmd=(python -u TrainingLite/rl_racing/run_training.py
          --auto-start-client
          --USE_CUSTOM_SAC_SAMPLING True
          --device cpu
          --SIMULATION_LENGTH 250000
          --model-name "$MODEL_NAME"
          --alpha "$alpha"
          --beta_start "$beta"
          --td_ratio "$ratio"
          --SAC_SPEED_CURRICULUM_LEARNING "$cur"
        )
        "${cmd[@]}"
        sleep 5
      done
    done
  done
done