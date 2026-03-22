#!/bin/bash

ALPHAS=(0.8 0.6 0.4 0.2 0.0) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.7) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0 0.5) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
ANNEALING_RATIOS=(0.4 0.8)
# NSTEPS=(1 2 3 5)
CURRICULUM=(True False)


# Loop
# for alpha in "${ALPHAS[@]}"; do
#   for beta in "${BETAS[@]}"; do
#     for ratio in "${RATIOS[@]}"; do
#       for cur in "${CURRICULUM[@]}"; do
      
#         MODEL_NAME="Sweep_BETTER_fresh_cur_A${alpha}_B${beta}_Cur${cur}"
        
#         echo "=================================================="
#         echo " STARTING: $MODEL_NAME"
#         echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
#         echo "=================================================="

#         cmd=(python -u TrainingLite/rl_racing/run_training.py
#           --auto-start-client
#           --USE_CUSTOM_SAC_SAMPLING True
#           --device cpu
#           --SIMULATION_LENGTH 250000
#           --model-name "$MODEL_NAME"
#           --alpha "$alpha"
#           --beta_start "$beta"
#           --td_ratio "$ratio"
#           --SAC_SPEED_CURRICULUM_LEARNING "$cur"
#         )
#         "${cmd[@]}"
#         sleep 5
#       done
#     done
#   done
# done


# for alpha in "${ALPHAS[@]}"; do
#   for beta in "${BETAS[@]}"; do
#     for ratio in "${RATIOS[@]}"; do
#       for anneal in "${ANNEALING_RATIOS[@]}"; do
#         MODEL_NAME="1203_A${alpha}_B${beta}_R${ratio}_Anneal${anneal}"
        
#         echo "=================================================="
#         echo " STARTING: $MODEL_NAME"
#         echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio" "| Anneal: $anneal"
#         echo "=================================================="

#         cmd=(python -u TrainingLite/rl_racing/run_training.py
#           --auto-start-client
#           --USE_CUSTOM_SAC_SAMPLING True
#           --device cpu
#           --SIMULATION_LENGTH 350000
#           --model-name "$MODEL_NAME"
#           --SAC_PRIORITY_FACTOR "$alpha"
#           --SAC_IMPORANCE_SAMPLING_CORRECTOR "$beta"
#           --SAC_STATE_TO_TD_RATIO "$ratio"
#           --SAC_BETA_ANNEALING_RATIO "$anneal"
#         )
#         "${cmd[@]}"
#         sleep 1
#       done
#     done
#   done
# done

STATE_TO_TD_RATIOS=(0.0 0.1 0.25 0.5 0.75 0.9 1.0)
PP_BUFFER_SIZE=(10000 30000 50000 70000)
for pp_size in "${PP_BUFFER_SIZE[@]}"; do
  MODEL_NAME="0321_actorInverseTD_critiUniform_PP_prefillSize${pp_size}"
  
  echo "=================================================="
  echo " STARTING: $MODEL_NAME"
  echo " PP Buffer Size: $pp_size "
  echo "=================================================="

  cmd=(python -u TrainingLite/rl_racing/run_training.py
    --auto-start-client
    --USE_CUSTOM_SAC_SAMPLING True
    --device cpu
    --SIMULATION_LENGTH 300000
    --model-name "$MODEL_NAME"
    --SAC_CUSTOM_UNIFORM_CRITIC True
    --SAC_PREFILL_BUFFER_WITH_PP True
    --SAC_PREFILL_BUFFER_WITH_PP_AMOUNT "$pp_size"
  )
  "${cmd[@]}"
  sleep 1
done