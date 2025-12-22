#!/bin/bash
#SBATCH --array=0-16             # Create an array job with task IDs 0-16 (17 jobs: 1 velocity × 17 friction values)
#SBATCH --mem-per-cpu=2G         # Request memory per CPU
#SBATCH --cpus-per-task=1        # Assign the required number of CPUs per task
#SBATCH --time=2:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out   # Output file

# Create output and error directories if they do not exist
mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=/cluster/home/paluchm/f1tenth_development_gym/
cd $HOME/f1tenth_development_gym/

# Run the Python script with the specific index
python ./SI_Toolkit_ASF/run/PreprocessData_Add_Control_Along_Trajectories.py -i $SLURM_ARRAY_TASK_ID

