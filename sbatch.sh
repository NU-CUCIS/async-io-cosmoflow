#!/bin/bash  -l

#SBATCH -t 00:05:00
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH -G 8
#SBATCH -A m844
#SBATCH --exclusive
#m1248
#m2956
#m844

ulimit -c unlimited
#export KMP_AFFINITY="granularity=fine,compact"
#export KMP_BLOCKTIME=0

#cd /global/homes/s/slz839/cosmo/cosmoflow
#srun -n 1 python3 ../cosmoflow/train_CosmoFlow.py --steps=16 --epochs=1 --batch_size=8 --noHorovod
#srun -n 2 python3 ../cosmoflow/train_CosmoFlow.py --steps=16 --epochs=1 --batch_size=8

cd /global/homes/s/slz839/cosmo/tf2-cosmoflow
#srun -n 1 -c 80 python3 main.py --epochs=3 --batch_size=16
srun -n 4 -c 20 python3 main.py --epochs=1 --batch_size=4 --overlap=0

#module unload gcc/8.3.0
#module unload cuda/10.1.243
#module load pytorch/v1.4.0-gpu
#cd /global/homes/s/slz839/cosmo/Cosmoflow-notes/pytorch_implementation
#srun -n 1 python3 train_cosmoflow.py cosmo6_dataSplit_peter_tryG_c1.yaml 8
