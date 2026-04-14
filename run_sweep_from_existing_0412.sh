#!/bin/bash

# ALPHAS=(0.4 0.6 0.8) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
# BETAS=(0.2 0.4) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
# RATIOS=(0.0 0.25 0.5) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only



SOURCE_MODEL="Example-1-Nikita"
SOURCE_MODEL_SHORT="Ex1_Nik"
NEW_MAP_NAME="RCA2"

ALPHAS=(0.7) # How much to mix with uniform -> 0.0 = uniform only, 1.0 = priority only
# BETAS=(0.4 0.6 0.8) # Starting beta for importance sampling bias correction-> 0.0 = no IS, 1.0 = full IS
RATIOS=(0.0 0.5 0.8) # Ratio of TD to state error -> 0.0 = TD only, 1.0 = state only

CRITIC_INVERT_TD_LIST=(False)
ACTOR_INVERT_TD_LIST=(True False)

CRITIC_UNIFORM_LIST=(False True)

# CURRICULUM_TYPE=()

# USE_CUSTOM_SAMPLING=(True False)

# USE_SPEED_CURRICULUM_LIST=(True False)


for alpha in "${ALPHAS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    for crit_inv in "${CRITIC_INVERT_TD_LIST[@]}"; do
      for act_inv in "${ACTOR_INVERT_TD_LIST[@]}"; do
        for critic_uni in "${CRITIC_UNIFORM_LIST[@]}"; do
            MODEL_NAME="from_${SOURCE_MODEL_SHORT}_0412_SimFr200_maxUDT025_A${alpha}_R${ratio}_CriU_${critic_uni}_CriInv_${crit_inv}_ActInv_${act_inv}"

            echo "=================================================="
            echo " STARTING: $MODEL_NAME"
            echo " Alpha: $alpha | Beta: $beta | Ratio: $ratio"
            echo " Critic Uniform: $critic_uni | Critic Invert TD: $crit_inv | Actor Invert TD: $act_inv | Rank Based: $rank_based"
            echo "=================================================="

            cmd=(python -u TrainingLite/rl_racing/run_training.py
              --auto-start-client
              --device cpu
              --SIMULATION_LENGTH 50000
              --load-model-name "$SOURCE_MODEL"
              --save-model-name "$MODEL_NAME"
              --MAP_NAME "$NEW_MAP_NAME"
              --alpha "$alpha"
              --td_ratio "$ratio"
              --SAC_CUSTOM_UNIFORM_CRITIC "$critic_uni"
              --SAC_CUSTOM_CRITIC_INVERT_TD "$crit_inv"
              --SAC_CUSTOM_ACTOR_INVERT_TD "$act_inv"
              --MAX_SIM_FREQUENCY 200
            )
            "${cmd[@]}"
            sleep 5
          done
        done
      done
    done
  done
done

# for respawn_timestep in "${RESPAWN_SETBACK_TIMESTEPS_LIST[@]}"; do
#   for respawn_probability in "${RESPAWN_PROBABILITY_LIST[@]}"; do
#     for index in 1 2; do
#       # MODEL_NAME="Sweep_Cur_from_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_CUR${CURRICULUM_T2}"
#       MODEL_NAME="0314_from_${SOURCE_MODEL_SHORT}_ResProb_${respawn_probability}_ResStep_${respawn_timestep}_Run${index}"
    
#       # MODEL_NAME="Sweep_nstep_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_N${}"
      
#       echo "=================================================="
#       echo " STARTING: $MODEL_NAME"
#       echo " Respawn Probability: $respawn_probability | Respawn Setback Timesteps: $respawn_timestep"
#       echo "=================================================="

#       python -u TrainingLite/rl_racing/run_training.py \
#         --auto-start-client \
#         --USE_CUSTOM_SAC_SAMPLING "$USE_CUSTOM_SAMPLING" \
#         --device cpu \
#         --SIMULATION_LENGTH 75000 \
#         --load-model-name "$SOURCE_MODEL" \
#         --save-model-name "$MODEL_NAME" \
#         --MAP_NAME "$NEW_MAP_NAME" \
#         --RESPAWN_SETBACK_TIMESTEPS $respawn_timestep \
#         --RESPAWN_PROBABILITY $respawn_probability \

#       sleep 5
#     done
#   done
# done

# Loop
# for alpha in "${ALPHAS[@]}"; do
#   for ratio in "${RATIOS[@]}"; do
#     for cur_t2 in "${CURRICULUM_T2[@]}"; do
#       for index in 1 2 3; do
#         # MODEL_NAME="Sweep_Cur_from_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_CUR${CURRICULUM_T2}"
#         MODEL_NAME="2602_A${alpha}_R_${ratio}_T2_${cur_t2}"
      
#         # MODEL_NAME="Sweep_nstep_${SOURCE_MODEL_SHORT}_A${alpha}_B${beta}_R${ratio}_N${}"
        
#         echo "=================================================="
#         echo " STARTING: $MODEL_NAME"
#         echo " Alpha: $alpha | Cur T2: $cur_t2 | Ratio: $ratio"
#         echo "=================================================="

#         cmd=(python -u TrainingLite/rl_racing/run_training.py
#           --auto-start-client
#           --USE_CUSTOM_SAC_SAMPLING True
#           --device cpu
#           --SIMULATION_LENGTH 350000
#           --load-model-name "$SOURCE_MODEL"
#           --save-model-name "$MODEL_NAME"
#           --MAP_NAME "$NEW_MAP_NAME"
#           --alpha "$alpha"
#           --td_ratio "$ratio"
#           --SAC_CURRICULUM_T2 "$cur_t2"
#           --replay-capacity 50000
#         )
#         "${cmd[@]}"
#         sleep 5
#       done
#     done
#   done
# done

# CAR_STATE_NOISE="[0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]"
# CONTROL_NOISE="[0.35, 0.7]"

# USE_SPEED_CURRICULUM_LIST=(True True True False False)
# SPEED_MODE=("speed_cap" "vel_factor" "speed_cap" None None)
# USE_WIDTH_CUR=(False False True True False)
# USE_NOISE_CUR=(False False False False True)


# for main_index in 1 2 3 4 5; do
#   for list_index in 0 1 2 3 4; do
#     MODEL_NAME="30Jan_Sweep_S_${SPEED_MODE[$list_index]}_W_${USE_WIDTH_CUR[$list_index]}_N_${USE_NOISE_CUR[$list_index]}_Run${main_index}"
#     echo "=================================================="
#     echo " STARTING: $MODEL_NAME"
#     echo " Alpha: 0.0 | Beta: 0.4 | Ratio: 0.0"
#     echo "=================================================="

#     python -u TrainingLite/rl_racing/run_training.py \
#       --auto-start-client \
#       --USE_CUSTOM_SAC_SAMPLING False \
#       --device cpu \
#       --SIMULATION_LENGTH 75000 \
#       --load-model-name "$SOURCE_MODEL" \
#       --save-model-name "$MODEL_NAME" \
#       --MAP_NAME "$NEW_MAP_NAME" \
#       --alpha 0.0 \
#       --beta_start 0.4 \
#       --td_ratio 0.0 \
#       --SAC_CURRICULUM_SPEED "${USE_SPEED_CURRICULUM_LIST[$list_index]}" \
#       --SAC_CURRICULUM_SPEED_ADJUST_MODE "${SPEED_MODE[$list_index]}" \
#       --SAC_CURRICULUM_SPEED_LIMIT_MAX 15 \
#       --SAC_CURRICULUM_SPEED_LIMIT 15 \
#       --SAC_CURRICULUM_TRACK_WIDTH_SCALING "${USE_WIDTH_CUR[$list_index]}" \
#       --SAC_CURRICULUM_TRACK_WIDTH_FACTOR 1.0 \
#       --SAC_CURRICULUM_NOISE_SCALING "${USE_NOISE_CUR[$list_index]}" \
#       --SAC_NOISE_LEVEL_CAR_STATE_MAX "$CAR_STATE_NOISE" \
#       --SAC_NOISE_LEVEL_CONTROL_MAX "$CONTROL_NOISE" \

#     sleep 5
#   done
# done