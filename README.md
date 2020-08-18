# tf2-cosmoflow
This program is a CosmoFlow implementation with TensorFlow 2.
The training is parallelized using Horovod.
For scalable parallel training, an asynchronous I/O module has been implemented based on python multiprocessing package.

## Software Requirements
  * TensorFlow > 2.0.0 (2.2.0 is recommended)
  * Horovod > 0.16

## Instructions to Run on Cori
1. Clone the code first.
```
git clone https://github.com/swblaster/tf2-cosmoflow
```

2. Load the modules for using TensorFlow, Horovod, and GPUs on Cori.
Note that Horovod is embedded in TensorFlow module.
```
module load tensorflow/gpu-2.2.0-py37
module load esslurm
```

3. Modify the hyper-parameters appropriately and start training.

## Questions/Comments
  * Sunwoo Lee <sunwoolee1.2014@u.northwestern.edu>
  * Wei-keng Liao <wkliao@northwestern.edu>
  * Alex Sim <asim@lbl.gov>
  * John Wu <kwu@lbl.gov>
