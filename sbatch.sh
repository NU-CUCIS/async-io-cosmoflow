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

cd /global/homes/s/slz839/cosmo/tf2-cosmoflow
srun -n 1 python3 train.py --epochs=1 --batch_size=8

cd /global/homes/s/slz839/cosmo/cosmoflow
srun -n 1 python3 ../cosmoflow/train_CosmoFlow.py --steps=16 --epochs=1 --batch_size=8 --noHorovod
