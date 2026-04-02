

python TrainingLite/rl_racing/run_training.py --auto-start-client --SAVE_RECORDINGS False --CONTROLLER sac_agent --SIMULATION_LENGTH 200000 --model-name Example-1
python run.py --RENDER_MODE human_fast --SIMULATION_LENGTH 3000 --MAX_SIM_FREQUENCY 1000 --SAVE_RECORDINGS True --CONTROLLER sac_agent --SAC_INFERENCE_MODEL_NAME Example-1
