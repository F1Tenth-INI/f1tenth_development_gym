#!/bin/bash
set -e  # exit if any command fails

PREFIX="SAC_RCA1_auto_"
counter=1

GRAD_STEPS_LIST=(128 256 512 1024 2048 4096)
BATCH_SIZE_LIST=(128 256 512 1024 2048 4096 8192)

for grad in "${GRAD_STEPS_LIST[@]}"; do
    for bs in "${BATCH_SIZE_LIST[@]}"; do
        MODEL_NAME="${PREFIX}${counter}"

        echo "======================================"
        echo " Run #$counter"
        echo "   Model: $MODEL_NAME"
        echo "   Gradient steps: $grad"
        echo "   Batch size: $bs"
        echo "======================================"

        python TrainingLite/rl_racing/learner_server.py \
            --model-name "$MODEL_NAME" \
            --gradient-steps "$grad" \
            --batch-size "$bs"

        echo " Finished run $counter / $(( ${#GRAD_STEPS_LIST[@]} * ${#BATCH_SIZE_LIST[@]} ))"
        echo
        counter=$((counter+1))
    done
done

echo "All experiments completed."
