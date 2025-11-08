python TrainingLite/rl_racing/run_training.py --auto-start-client --SIMULATION_LENGTH 300000 --model-name OriginalReward1
python run.py --RENDER_MODE human_fast --SIMULATION_LENGTH 2000 --SAVE_RECORDINGS True --SAC_INFERENCE_MODEL_NAME OriginalReward1

python TrainingLite/rl_racing/run_training.py --auto-start-client --SIMULATION_LENGTH 300000 --model-name OriginalReward2
python run.py --RENDER_MODE human_fast --SIMULATION_LENGTH 2000 --SAVE_RECORDINGS True --SAC_INFERENCE_MODEL_NAME OriginalReward2

python TrainingLite/rl_racing/run_training.py --auto-start-client --SIMULATION_LENGTH 300000 --model-name OriginalReward3
python run.py --RENDER_MODE human_fast --SIMULATION_LENGTH 2000 --SAVE_RECORDINGS True --SAC_INFERENCE_MODEL_NAME OriginalReward3

python TrainingLite/rl_racing/run_training.py --auto-start-client --SIMULATION_LENGTH 300000 --model-name OriginalReward4
python run.py --RENDER_MODE human_fast --SIMULATION_LENGTH 2000 --SAVE_RECORDINGS True --SAC_INFERENCE_MODEL_NAME OriginalReward4
