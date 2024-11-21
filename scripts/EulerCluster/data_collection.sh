#!/bin/bash
#SBATCH --array=25-48             # Create an array job with task IDs from 1 to 12
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --time=8:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=/cluster/home/bollif/f1tenth_development_gym/
cd $HOME/f1tenth_development_gym/

# Use SLURM_ARRAY_TASK_ID for the model index directly since it ranges from 1 to 50
i=$SLURM_ARRAY_TASK_ID

# Run the Python script with the specific index
python run/data_collection.py -i $i

