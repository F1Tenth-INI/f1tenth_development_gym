#!/bin/bash
#SBATCH --array=0-0           # 0-len(feature_A)xlen(feature_B)xnum_repetitions - 1
#SBATCH --cpus-per-task=8        # Assign the required number of CPUs per task
#SBATCH --mem-per-cpu=4G        # Request memory per CPU
#SBATCH --time=2:00:00           # Set the maximum job time
#SBATCH --output=./scripts/EulerCluster/out/data_collection/slurm-%A_%a.out   # Output file

# ZIP ExperimentRecordings/Datasetname

experiment_name="04_08_RCA1and2_noise"

cd $HOME/f1tenth_development_gym/
zip -r ExperimentRecordings/${experiment_name}.zip ExperimentRecordings/${experiment_name}