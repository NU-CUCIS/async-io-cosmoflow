#!/bin/bash  -l

#SBATCH -t 00:10:00
#SBATCH --qos=regular
#SBATCH --nodes=8
#SBATCH --constraint=gpu
#SBATCH -G 64
#SBATCH --exclusive
#SBATCH -A m844

ulimit -c unlimited

srun -n 64 -c 10 python3 main.py --epochs=3 \
                                 --batch_size=4 \
                                 --overlap=1 \
                                 --checkpoint=0 \
                                 --cache_size=0 \
                                 --file_shuffle=1 \
                                 --record_acc=0 \
                                 --evaluate=0
