#!/bin/bash
# SBATCH --array=1-756            # Create an array job with task IDs 1-756
#SBATCH --mem-per-cpu=3G         # Request memory per CPU
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --time=2:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=/cluster/home/paluchm/f1tenth_development_gym/
cd $HOME/f1tenth_development_gym/

# Use SLURM_ARRAY_TASK_ID for the model index directly since it ranges from 1 to 50
i=$SLURM_ARRAY_TASK_ID

# Run the Python script with the specific index
python ./SI_Toolkit_ASF/run/PreprocessData_Add_Control_Along_Trajectories.py -i $i

