#!/bin/bash
#SBATCH --array=0-0
#SBATCH --cpus-per-task=4        # Assign the required number of CPUs per task
#SBATCH --mem-per-cpu=4G        # Request 2GB of memory per CPU
#SBATCH --time=8:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out   # Output file
#SBATCH --error=./scripts/EulerCluster/out/slurm-%A_%a.err    # Error file


# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=/cluster/home/bollif/f1tenth_development_gym/
cd $HOME/f1tenth_development_gym/

# Run the Python script with the specific index
python SI_Toolkit_ASF/run/Train_Network.py

