#!/bin/bash

# ALPHAS=(0.2 0.4 0.6 0.8)
# BETAS=(0.2 0.4 0.6)
# RATIOS=(0.0 0.25 0.5)

# Define your grid
# ALPHAS=(0.0 0.4 0.8)
# BETAS=(0.4 0.6)
# RATIOS=(0.0 0.25 0.5)

ALPHAS=(0.8 0.6 0.0)
BETAS=(0.4)
RATIOS=(0.0)

SOURCE_MODEL="Example-1"
SOURCE_MODEL_SHORT="Ex1"
NEW_MAP_NAME="hangar16"
# ALPHAS=(0.8)
# BETAS=(0.4 0.6)
# RATIOS=(0.0 0.25 0.5)

# Loop
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      
      MODEL_NAME="Sweep_lap_debug_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}"
      
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
        --td_ratio $ratio

      sleep 5

    done
  done
done