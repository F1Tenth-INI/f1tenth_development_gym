#!/bin/bash
#SBATCH --array=0-16             # 17 jobs: 1 velocity × 17 friction values
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --mem-per-cpu=3G         # Request memory per CPU
#SBATCH --time=2:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=$HOME/f1tenth_development_gym:$PYTHONPATH
cd $HOME/f1tenth_development_gym/

# Run the Python script with the euler index
# For repetitions, use --array=0-N where N = (17 × num_reps - 1) and uncomment modulo line:
# EULER_INDEX=$((SLURM_ARRAY_TASK_ID % 17))
python run/data_collection.py -i $SLURM_ARRAY_TASK_ID

