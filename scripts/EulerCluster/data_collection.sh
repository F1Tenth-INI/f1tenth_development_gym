#!/bin/bash
#SBATCH --array=0-377           # 0-len(feature_A)xlen(feature_B)xnum_repetitions - 1
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --mem-per-cpu=3G        # Request memory per CPU
#SBATCH --time=2:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=$HOME/f1tenth_development_gym:$PYTHONPATH
cd $HOME/f1tenth_development_gym/

# Run the Python script with the speed factor
python run/data_collection.py -i $SLURM_ARRAY_TASK_ID

