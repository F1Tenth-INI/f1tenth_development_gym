# Euler Cluster

## Run jobs
```console
sbatch scripts/EulerCluster/data_collection.sh 
```
For Training immitator
## Run jobs
```console
sbatch scripts/EulerCluster/train_network.sh
```


For training RL
## Run jobs
```console
sbatch scripts/EulerCluster/ppo.sh
```


## Otehr commands

Show status of job:
```console
squeue
```
Cancel job
```console
scancel [jobid]
```