#!/usr/bin/env bash
# Run finetune training (same flags as finetune.sh) for several save names: RCA1-1a, RCA1-1b, ...
# Usage: from repo root: ./TrainingLite/rl_racing/finetune_loop.sh
# Optional: FINETUNE_SUFFIXES="a b c" ./TrainingLite/rl_racing/finetune_loop.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SUFFIXES="${FINETUNE_SUFFIXES:-a}"

for suffix in ${SUFFIXES}; do
  echo "=== Finetune  ==="
  python TrainingLite/rl_racing/run_training.py \
    --auto-start-client \
    --CONTROLLER sac_agent \
    --SAVE_RECORDINGS False \
    --MAP_NAME RCA1 \
    --SAC_TARGET_UDT 1 \
    --learning-rate 1e-4 \
    --batch-size 64 \
    --SAC_CHECKPOINT_FREQUENCY 10000 \
    --SIMULATION_LENGTH 50000 \
    --MAX_SIM_FREQUENCY 250 \
    --ENV_CAR_PARAMETER_FILE gym_car_parameters_finetune.yml \
    --SAC_TERMINATE_BELOW_LAPTIME 23.0 \
    --save_replay_buffer True \
    --load_replay_buffer True \
    --load-model-name Example-1a \
    --save-model-name "Example-1a${suffix}"
    
done
