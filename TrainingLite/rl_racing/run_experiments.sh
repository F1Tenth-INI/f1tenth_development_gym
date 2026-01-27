
python TrainingLite/rl_racing/run_training.py --auto-start-client --CONTROLLER sac_agent --SAVE_RECORDINGS False --SIMULATION_LENGTH 500000 --MAP_NAME RCA2 --load-model-name Example-1 --save-model-name Example-1d
python run.py --RENDER_MODE human_fast --SIMULATION_LENGTH 1000 --SAVE_RECORDINGS True --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME Example-1