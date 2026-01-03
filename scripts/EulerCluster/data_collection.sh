#!/bin/bash
# 1190 jobs = 7 velocity factors × 170 repetitions
# Each job samples mu uniformly from [0.3, 1.1]
#SBATCH --array=0-1189
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=2:00:00
#SBATCH --output=./scripts/EulerCluster/out/slurm-%A_%a.out

mkdir -p ./scripts/EulerCluster/out/

source $HOME/miniconda3/bin/activate
conda activate f1t

export PYTHONPATH=$HOME/f1tenth_development_gym:$PYTHONPATH
cd $HOME/f1tenth_development_gym/

# Each array task ID directly maps to euler_index (no modulo needed)
python run/data_collection.py -i $SLURM_ARRAY_TASK_ID

