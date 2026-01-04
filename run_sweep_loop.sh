#!/bin/bash

ALPHAS=(0.0 0.6) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
NSTEPS=(1 2 3 5)

# Loop
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for nstep in "${NSTEPS[@]}"; do
      
        MODEL_NAME="Sweep_fresh_nstep_A${alpha}_B${beta}_R${ratio}_N${nstep}"
        
        echo "=================================================="
        echo " STARTING: $MODEL_NAME"
        echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
        echo "=================================================="

        python -u TrainingLite/rl_racing/run_training.py \
          --auto-start-client \
          --USE_CUSTOM_SAC_SAMPLING True \
          --device cpu \
          --SIMULATION_LENGTH 250000 \
          --model-name "$MODEL_NAME" \
          --alpha $alpha \
          --beta_start $beta \
          --td_ratio $ratio \
          --SAC_N_STEP $nstep

        sleep 5
      done
    done
  done
done