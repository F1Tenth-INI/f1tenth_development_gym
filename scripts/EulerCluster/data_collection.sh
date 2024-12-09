#!/bin/bash
#SBATCH --array=0-47
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --mem-per-cpu=4G        # Request 2GB of memory per CPU
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

# Define the speed factors and repetitions
speed_factors=(0.4 0.6 0.8 1.0)
repetitions=12

# Calculate the speed factor and repetition index
speed_factor_index=$((SLURM_ARRAY_TASK_ID / repetitions))
repetition_index=$((SLURM_ARRAY_TASK_ID % repetitions))

# Get the speed factor
speed_factor=${speed_factors[$speed_factor_index]}

# Run the Python script with the speed factor
python run/data_collection.py -s $speed_factor -i $i

