# python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 256 --learning-rate 3e-4 --discount-factor 0.96 --SAC_TARGET_UTD None --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 250000 --WEB_RENDER_AUTO_OPEN False --save-model-name Example-2i



#  python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 1024 --learning-rate 3e-4 --discount-factor 0.99 --SAC_TARGET_UTD None --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --save-model-name Example-2e


#  python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 1024 --learning-rate 3e-4 --discount-factor 0.99 --SAC_TARGET_UTD None --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --SAC_N_STEP 3 --save-model-name Example-2l


# python run.py --CONTROLLER sac_agent --SAVE_RECORDINGS False --MAX_SIM_FREQUENCY 50 --SAC_INFERENCE_MODEL Example-2e 


 python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 256 --learning-rate 3e-4 --discount-factor 0.98 --SAC_TARGET_UTD 1 --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --SAC_N_STEP 1 --save-model-name Example-TD1
 python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 256 --learning-rate 3e-4 --discount-factor 0.98 --SAC_TARGET_UTD 1 --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --SAC_N_STEP 2 --save-model-name Example-TD2
 python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 256 --learning-rate 3e-4 --discount-factor 0.98 --SAC_TARGET_UTD 1 --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --SAC_N_STEP 3 --save-model-name Example-TD3
 python TrainingLite/rl_racing/run_training.py --auto-start-client --batch-size 256 --learning-rate 3e-4 --discount-factor 0.98 --SAC_TARGET_UTD 1 --SAVE_RECORDINGS False --SAC_CHECKPOINT_FREQUENCY 500000 --CONTROLLER sac_agent --SIMULATION_LENGTH 150000 --WEB_RENDER_AUTO_OPEN False --SAC_N_STEP 4 --save-model-name Example-TD4
