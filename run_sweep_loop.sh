#!/bin/bash

ALPHAS=(0.2 0.4 0.6 0.8)
BETAS=(0.2 0.4 0.6)
RATIOS=(0.0 0.25 0.5)

# Define your grid
ALPHAS=(0.4)
BETAS=(0.4 0.6)
RATIOS=(0.0 0.25 0.5)

# ALPHAS=(0.8)
# BETAS=(0.4 0.6)
# RATIOS=(0.0 0.25 0.5)

# Loop
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      
      MODEL_NAME="Sweep_A${alpha}_B${beta}_R${ratio}"
      
      echo "=================================================="
      echo " STARTING: $MODEL_NAME"
      echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
      echo "=================================================="

      python -u TrainingLite/rl_racing/run_training.py \
        --auto-start-client \
        --USE_CUSTOM_SAC_SAMPLING True \
        --device cpu \
        --SIMULATION_LENGTH 200000 \
        --model-name "$MODEL_NAME" \
        --alpha $alpha \
        --beta_start $beta \
        --td_ratio $ratio

      sleep 5

    done
  done
done