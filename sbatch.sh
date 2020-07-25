#!/bin/bash  -l

#SBATCH -t 00:10:00
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --constraint=gpu
#SBATCH -G 32
#SBATCH --exclusive
#SBATCH -A m844

ulimit -c unlimited

srun -n 32 -c 10 python3 main.py --epochs=3 --batch_size=8 --overlap=1 --checkpoint=0 --cache_size=0 --file_shuffle=1
