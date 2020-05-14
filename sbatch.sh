#!/bin/bash  -l

#SBATCH -t 00:05:00
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --constraint=gpu
#SBATCH -G 1
#SBATCH -A m844
#m1248
#m2956
#m844

ulimit -c unlimited
srun -n 1 python3 train.py --epochs=1 --batch_size=8
