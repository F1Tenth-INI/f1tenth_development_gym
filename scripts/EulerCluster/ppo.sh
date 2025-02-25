#!/bin/bash
#SBATCH --array=0-0
#SBATCH --cpus-per-task=16        # Assign the required number of CPUs per task
#SBATCH --mem-per-cpu=2G        # Request 2GB of memory per CPU
#SBATCH --time=8:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/ppo/slurm-%A_%a.out   # Output file
#SBATCH --error=./scripts/EulerCluster/out/ppo/slurm-%A_%a.err    # Error file


# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/out/ppo/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=/cluster/home/bollif/f1tenth_development_gym/
cd $HOME/f1tenth_development_gym/

# Run the Python script with the specific index
python TrainingLite/ppo_racing/ppo_test.py 

