#!/bin/bash
# Single-file mode: processes DEFAULT_INPUT_ROOT from the Python script.
# For array jobs with indexed files, uncomment --array and use -i $SLURM_ARRAY_TASK_ID
#SBATCH --mem-per-cpu=2G         # Request memory per CPU
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --time=2:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out   # Output file

# Uncomment for array jobs (indexed files like *_001.csv, *_002.csv, etc.):
# #SBATCH --array=1-17
# Then change output to: --output=./scripts/EulerCluster/out/slurm-%A_%a.out
# And add -i $SLURM_ARRAY_TASK_ID to the python command below.

mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t_tf15  # Change to f1t_tf15 if that's the env name on Euler

export PYTHONPATH=$HOME/f1tenth_development_gym/
cd $HOME/f1tenth_development_gym/

# Single file mode (uses DEFAULT_INPUT_ROOT from Python script)
python ./SI_Toolkit_ASF/run/PreprocessData_Add_Control_Along_Trajectories.py

# For array jobs with indexed files, use:
# python ./SI_Toolkit_ASF/run/PreprocessData_Add_Control_Along_Trajectories.py -i $SLURM_ARRAY_TASK_ID

