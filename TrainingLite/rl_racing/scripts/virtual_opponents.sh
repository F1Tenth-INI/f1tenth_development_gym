
# python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 1024 --SAC_TARGET_UTD None --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --save-model-name Example-2


python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 1024 \
 --learning-rate 3e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 150000 \
 --save_replay_buffer True \
 --save-model-name VirtualOpponents-1 \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1


python TrainingLite/rl_racing/run_training.py  --auto-start-client  --batch-size 1024  --learning-rate 3e-4  --discount-factor 0.98  --SAC_TARGET_UTD None  --SAVE_RECORDINGS False  --SAC_CHECKPOINT_FREQUENCY 500000  --CONTROLLER sac_agent  --SIMULATION_LENGTH 250000  --save_replay_buffer True --load-model-name VirtualOpponents-1  --save-model-name VirtualOpponents-1b  --NUMBER_OF_VIRTUAL_OPPONENTS 1

python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-1b --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True




# 2. with w_proximity = 1.0
python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 1024 \
 --learning-rate 3e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --save_replay_buffer True \
 --save-model-name VirtualOpponents-2 \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1

python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-2 --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True

 # Safe but no longer overtaking...



# 3. with w_proximity = 1.0 and w_lateral_error = 0.0, also smaller opponents
python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 1024 \
 --learning-rate 3e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --save_replay_buffer True \
 --save-model-name VirtualOpponents-3 \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1

 python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-3 --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True

# Almost reliable.. but not quite yet

# 4. with w_proximity = 1.0 and w_lateral_error = 0.0, also smaller opponents, also slip penalty
python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 1024 \
 --learning-rate 3e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --save_replay_buffer True \
 --save-model-name VirtualOpponents-4 \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1

 python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-4 --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True --MAX_EPISODE_LENGTH 10000


# First reliably working example... try finetune


# Try finetuning with smaller batch size / lr 

python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4b \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1

 python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-4b --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True --MAX_EPISODE_LENGTH 10000



 # Try finetuning with larger opponents
python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4c \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1

 python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-4b --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True --MAX_EPISODE_LENGTH 10000



 

  # Try finetuning with faster opponents (0.8)?
python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4d \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1



# Even faster opponents (0.9)
 python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 100000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4e \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1

 #That was too fast: 0.85 now
 python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 100000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4f \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1



 python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-4f --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True --MAX_EPISODE_LENGTH 10000


# 0.87 increase max episode length for better trailing...
 python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --MAX_EPISODE_LENGTH 10000 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 100000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4g \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1 \
 --VIRTUAL_OPPONENT_VEL_FACTORS "[0.87]"


 python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-4g --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True --MAX_EPISODE_LENGTH 10000 --VIRTUAL_OPPONENT_VEL_FACTORS "[0.87]"



# Virtual opponent with random speeeds per sposode (( 0.5,1.0))
 python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --MAX_EPISODE_LENGTH 4000 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 250000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4i \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1 \
 --EPISODE_RANDOMIZATION_FILE utilities/episode_randomization.yaml

# WORKS WELL FOR MULTIPLE SPEED
  python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-4i --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True --MAX_EPISODE_LENGTH 10000 --VIRTUAL_OPPONENT_VEL_FACTORS "[0.7]"




# Virtual opponent with random speeeds per sposode (( 0.5,1.0)) - same as before but crash penality higher : 25
 python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --MAX_EPISODE_LENGTH 4000 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 250000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4l \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1 \
 --EPISODE_RANDOMIZATION_FILE utilities/episode_randomization.yaml


python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME VirtualOpponents-4l --RENDER_MODE human_fast --MAX_SIM_FREQUENCY 50 --NUMBER_OF_VIRTUAL_OPPONENTS 1 --SIMULATION_LENGTH 10000 --SAVE_RECORDINGS True --MAX_EPISODE_LENGTH 10000 --VIRTUAL_OPPONENT_VEL_FACTORS "[0.7]"



# Crash is fault of car behind.
 python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --MAX_EPISODE_LENGTH 4000 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4j \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1 \
 --EPISODE_RANDOMIZATION_FILE utilities/episode_randomization.yaml


 # Active reward for overtaking, first try

 # Crash is fault of car behind.
 python TrainingLite/rl_racing/run_training.py \
 --auto-start-client \
 --WEB_RENDER_AUTO_OPEN False \
 --batch-size 512 \
 --learning-rate 1e-4 \
 --MAX_EPISODE_LENGTH 4000 \
 --discount-factor 0.98 \
 --SAC_TARGET_UTD None \
 --SAVE_RECORDINGS False \
 --SAC_CHECKPOINT_FREQUENCY 500000 \
 --CONTROLLER sac_agent \
 --SIMULATION_LENGTH 500000 \
 --load_replay_buffer True \
 --save_replay_buffer True \
 --load-model-name VirtualOpponents-4 \
 --save-model-name VirtualOpponents-4k \
 --NUMBER_OF_VIRTUAL_OPPONENTS 1 \
 --EPISODE_RANDOMIZATION_FILE utilities/episode_randomization.yaml