# tf2-cosmoflow
This is a cosmoflow implementation with TensorFlow 2.1.
The training is parallelized using Horovod.

## Software Requirements
  * TensorFlow > 2.0.0
  * Horovod > 0.16

## Instructions to Run on Cori
1. Clone the code first.
```
git clone https://github.com/swblaster/tf2-cosmoflow
```

2. Load the modules for using TensorFlow, Horovod, and GPUs on Cori.
Note that Horovod is embedded in TensorFlow module.
```
module load tensorflow/gpu-2.1.0-py37
module load esslurm
```

3. Modify the hyper-parameters appropriately and run the job.

## Questions/Comments
  * Sunwoo Lee <sunwoolee1.2014@u.northwestern.edu>
  * Wei-keng Liao <wkliao@eecs.northwestern.edu>