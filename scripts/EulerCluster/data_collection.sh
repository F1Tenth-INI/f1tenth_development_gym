#!/bin/bash
#SBATCH --array=0-335           # 0-55 indices × 6 repetitions = 336 jobs (for ~672 files)
                                 # For single sweep: use --array=0-55 (produces 112 files)
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

# Map array task ID to euler_index (0-55) using modulo
# This allows multiple repetitions: 0-335 maps to 0-55 six times
EULER_INDEX=$((SLURM_ARRAY_TASK_ID % 56))

# Run the Python script with the euler index
python run/data_collection.py -i $EULER_INDEX

