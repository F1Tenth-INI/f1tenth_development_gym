ALPHAS=(0.8) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.6) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.8) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
CRITIC_UNIFORM=(True)
CUSTOM_SAMPLING=(True)

OFFSET_WEIGHTS=(0.0 3.0)
HEADINF_ERROR_WEIGHTS=(0.0 3.0)
REWARD_WEIGHTS=(5.0)
VELOCITY_WEIGHTS=(0.0 3.0)

# Loop
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for custom_on in "${CUSTOM_SAMPLING[@]}"; do
        for offset_w in "${OFFSET_WEIGHTS[@]}"; do
          for heading_err_w in "${HEADINF_ERROR_WEIGHTS[@]}"; do
            for reward_w in "${REWARD_WEIGHTS[@]}"; do
              for velocity_w in "${VELOCITY_WEIGHTS[@]}"; do
                MODEL_NAME="0412_SimFr200_maxUDT025_custom_${custom_on}_A${alpha}_R${ratio}_OW${offset_w}_HW${heading_err_w}_RW${reward_w}_VW${velocity_w}"

                echo "=================================================="
                echo " STARTING: $MODEL_NAME"
                echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
                echo " Custom Sampling: $custom_on"
                echo " Offset W: $offset_w | Heading Err W: $heading_err_w | Reward W: $reward_w | Velocity W: $velocity_w"
                echo "=================================================="

                cmd=(python -u TrainingLite/rl_racing/run_training.py
                  --auto-start-client
                  --USE_CUSTOM_SAC_SAMPLING "$custom_on"
                  --device cpu
                  --SIMULATION_LENGTH 250000
                  --model-name "$MODEL_NAME"
                  --alpha "$alpha"
                  --beta_start "$beta"
                  --td_ratio "$ratio"
                  --SAC_CUSTOM_UNIFORM_CRITIC True
                  --MAX_SIM_FREQUENCY 200
                  --SAC_WP_OFFSET_WEIGHT "$offset_w"
                  --SAC_WP_HEADING_ERROR_WEIGHT "$heading_err_w"
                  --SAC_REWARD_WEIGHT "$reward_w"
                  --SAC_VELOCITY_WEIGHT "$velocity_w"
                )
                "${cmd[@]}"
                sleep 5
              done
            done
          done
        done
      done
    done
  done
done

# # Loop
# for alpha in "${ALPHAS[@]}"; do
#   for beta in "${BETAS[@]}"; do
#     for ratio in "${RATIOS[@]}"; do
#       for custom_on in "${CUSTOM_SAMPLING[@]}"; do
#         for idx in 1; do
#           MODEL_NAME="0411_SimFr200_maxUDT025_custom_${custom_on}_A${alpha}_R${ratio}_idx${idx}"
          
#           echo "=================================================="
#           echo " STARTING: $MODEL_NAME"
#           echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio" 
#           echo "| Custom Sampling: $custom_on"
#           echo "=================================================="

#           cmd=(python -u TrainingLite/rl_racing/run_training.py
#             --auto-start-client
#             --USE_CUSTOM_SAC_SAMPLING "$custom_on"
#             --device cpu
#             --SIMULATION_LENGTH 250000
#             --model-name "$MODEL_NAME"
#             --alpha "$alpha"
#             --beta_start "$beta"
#             --td_ratio "$ratio"
#             --SAC_CUSTOM_UNIFORM_CRITIC True
#             --MAX_SIM_FREQUENCY 200
#             --SAC_WP_OFFSET_WEIGHT = 0.0
#             --SAC_WP_HEADING_ERROR_WEIGHT = 0.0
#             --SAC_REWARD_WEIGHT = 5.0
#             --SAC_VELOCITY_WEIGHT = 0.0
#           )
#           "${cmd[@]}"
#           sleep 5
#         done
#       done
#     done
#   done
# done