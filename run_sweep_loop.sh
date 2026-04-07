#!/bin/bash

ALPHAS=(0.8 0.6 0.4) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.7 0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0 0.4 0.75) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
PP_PREFILL=(False True)
CRITIC_UNIFORM=(True False)


# Loop
# for alpha in "${ALPHAS[@]}"; do
#   for beta in "${BETAS[@]}"; do
#     for ratio in "${RATIOS[@]}"; do
#       for critic_uniform in "${CRITIC_UNIFORM[@]}"; do
#         for idx in 1 2 3; do
#           MODEL_NAME="Sweep_BETTER_fresh_cur_A${alpha}_B${beta}_R${ratio}_CritU${critic_uniform}_idx${idx}"
          
#           echo "=================================================="
#           echo " STARTING: $MODEL_NAME"
#           echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio" 
#           echo "| CritUniform: $critic_uniform"
#           echo "=================================================="

#           cmd=(python -u TrainingLite/rl_racing/run_training.py
#             --auto-start-client
#             --USE_CUSTOM_SAC_SAMPLING True
#             --device cpu
#             --SIMULATION_LENGTH 250000
#             --model-name "$MODEL_NAME"
#             --alpha "$alpha"
#             --beta_start "$beta"
#             --td_ratio "$ratio"
#             --SAC_CUSTOM_UNIFORM_CRITIC "$critic_uniform"
#           )
#           "${cmd[@]}"
#           sleep 5
#         done
#       done
#     done
#   done
# done



CUSTOM_ON=(True False)

# for custom in "${CUSTOM_ON[@]}"; do
#   for idx in 1 2; do
#     MODEL_NAME="0406_MyExample-3_250k_custom_${custom}_seed${idx}"
  
#     echo "=================================================="
#     echo " STARTING: $MODEL_NAME"
#     echo " Custom Sampling: $custom "
#     echo "=================================================="

#     cmd=(python -u TrainingLite/rl_racing/run_training.py
#       --auto-start-client
#       --USE_CUSTOM_SAC_SAMPLING True
#       --device cpu
#       --SIMULATION_LENGTH 250000
#       --model-name "$MODEL_NAME"
#       --SAC_PRIORITY_FACTOR 0.0
#       --USE_CUSTOM_SAC_SAMPLING "$custom"
#     )
#     "${cmd[@]}"
#     sleep 1
#   done
# done

BC_EPOCH_LENGTHS=(10 25 50)
PP_PREFILL_AMOUNT=(10000 50000 100000)

for epochs in "${BC_EPOCH_LENGTHS[@]}"; do
  for prefill_amount in "${PP_PREFILL_AMOUNT[@]}"; do
    for custom in "${CUSTOM_ON[@]}"; do
      MODEL_NAME="0406_prefill_BC_e${epochs}_p${prefill_amount}_custom_${custom}"

      echo "=================================================="
      echo " STARTING: $MODEL_NAME"
      echo " Custom Sampling: $custom "
      echo "=================================================="

      cmd=(python -u TrainingLite/rl_racing/run_training.py
        --auto-start-client
        --USE_CUSTOM_SAC_SAMPLING True
        --device cpu
        --SIMULATION_LENGTH 250000
        --model-name "$MODEL_NAME"
        --SAC_PRIORITY_FACTOR 0.0
        --USE_CUSTOM_SAC_SAMPLING "$custom"
        --SAC_PREFILL_BUFFER_WITH_PP True
        --SAC_PREFILL_BEHAVIOR_CLONING_EPOCHS "$epochs"
        --SAC_PREFILL_BUFFER_WITH_PP_AMOUNT "$prefill_amount"
      )
      "${cmd[@]}"
      sleep 1
    done
  done
done