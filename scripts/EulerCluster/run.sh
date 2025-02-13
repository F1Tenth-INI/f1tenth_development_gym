#!/bin/bash
#SBATCH --array=25-25             # Create an array job with task IDs from 1 to 12
#SBATCH --cpus-per-task=12        # Assign the required number of CPUs per task
#SBATCH --mem=4G  # Request 4GB of memory for the job
#SBATCH --time=8:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=/cluster/home/bollif/f1tenth_development_gym/
cd $HOME/f1tenth_development_gym/

# Use SLURM_ARRAY_TASK_ID for the model index directly since it ranges from 1 to 50
i=$SLURM_ARRAY_TASK_ID

# Run the Python script with the specific index
python run.py -i $i

