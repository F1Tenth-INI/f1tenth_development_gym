python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 1024 --learning-rate 3e-4 --discount-factor 0.99 --SAC_TARGET_UTD None --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --save-model-name Example-1


python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 1024 --learning-rate 3e-4 --discount-factor 0.99 --SAC_TARGET_UTD None --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --save-model-name Example-3 --NUMBER_OF_VIRTUAL_OPPONENTS 1
python run.py --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME Example-3 --MAX_SIM_FREQUENCY 50 --SIMULATION_LENGTH 5000 --SAVE_RECORDINGS True --NUMBER_OF_VIRTUAL_OPPONENTS 1






# Exampel Nik
python TrainingLite/rl_racing/run_training.py \
--auto-start-client \
--batch-size 1024 \
--learning-rate 3e-4 \
--discount-factor 0.99 \
--SAC_TARGET_UTD None \
--SAVE_RECORDINGS False \
--SAC_CHECKPOINT_FREQUENCY 500000 \
--CONTROLLER sac_agent \
--save_replay_buffer True \
--SIMULATION_LENGTH 150000 \
--WEB_RENDER_AUTO_OPEN False \
--save-model-name Example-11

# Retrain with 100 lap finish reward
python TrainingLite/rl_racing/run_training.py \
--auto-start-client \
--batch-size 1024 \
--learning-rate 3e-4 \
--discount-factor 0.99 \
--SAC_TARGET_UTD None \
--SAVE_RECORDINGS False \
--SAC_CHECKPOINT_FREQUENCY 500000 \
--CONTROLLER sac_agent \
--load-replay-buffer True \
--SIMULATION_LENGTH 250000 \
--WEB_RENDER_AUTO_OPEN False \
--load-model-name Example-11 \
--save-model-name Example-11b

# Continuous lap finish reward
python TrainingLite/rl_racing/run_training.py \
--auto-start-client \
--batch-size 1024 \
--learning-rate 3e-4 \
--discount-factor 0.99 \
--SAC_TARGET_UTD None \
--SAVE_RECORDINGS False \
--SAC_CHECKPOINT_FREQUENCY 500000 \
--CONTROLLER sac_agent \
--load-replay-buffer True \
--SIMULATION_LENGTH 250000 \
--WEB_RENDER_AUTO_OPEN False \
--load-model-name Example-11 \
--save-model-name Example-11c


# Sector finish reward
python TrainingLite/rl_racing/run_training.py \
--auto-start-client \
--batch-size 1024 \
--learning-rate 3e-4 \
--discount-factor 0.99 \
--SAC_TARGET_UTD None \
--SAVE_RECORDINGS False \
--SAC_CHECKPOINT_FREQUENCY 500000 \
--CONTROLLER sac_agent \
--load-replay-buffer True \
--SIMULATION_LENGTH 250000 \
--WEB_RENDER_AUTO_OPEN False \
--load-model-name Example-11 \
--save-model-name Example-11d


# Less sectors, more reward
python TrainingLite/rl_racing/run_training.py \
--auto-start-client \
--batch-size 1024 \
--learning-rate 3e-4 \
--discount-factor 0.99 \
--SAC_TARGET_UTD None \
--SAVE_RECORDINGS False \
--SAC_CHECKPOINT_FREQUENCY 500000 \
--CONTROLLER sac_agent \
--load-replay-buffer True \
--SIMULATION_LENGTH 250000 \
--WEB_RENDER_AUTO_OPEN False \
--load-model-name Example-11d \
--save-model-name Example-11e