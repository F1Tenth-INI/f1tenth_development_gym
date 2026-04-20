ALPHAS=(0.8 0.7 0.6 0.5 0.4 0.3 0.2) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
BETAS=(0.9 0.7 0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0 0.2 0.4 0.6 0.8) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only
# PP_PREFILL=(False True)
CRITIC_UNIFORM=(True False)
ACTOR_INVERSE_TD_ERROR=(True False)


# Loop
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      for critic_uniform in "${CRITIC_UNIFORM[@]}"; do
        for actor_inverse_td_error in "${ACTOR_INVERSE_TD_ERROR[@]}"; do
          for idx in 1 2 3; do
            MODEL_NAME="Sweep_BETTER_fresh_cur_A${alpha}_B${beta}_R${ratio}_CritU${critic_uniform}_ActorInvTDE${actor_inverse_td_error}_idx${idx}"

            echo "=================================================="
            echo " STARTING: $MODEL_NAME"
          echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio" 
          echo "| CritUniform: $critic_uniform"
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
            --SAC_CUSTOM_UNIFORM_CRITIC "$critic_uniform"
          )
          "${cmd[@]}"
          sleep 5
        done
      done
    done
  done
done