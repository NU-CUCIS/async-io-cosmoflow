#!/bin/bash  -l

#SBATCH -t 04:00:00
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH -G 8
#SBATCH --exclusive
#SBATCH -A m844

ulimit -c unlimited

srun -n 8 -c 10 python3 main.py --epochs=100 --batch_size=4 --overlap=1 --checkpoint=1 --cache_size=32
