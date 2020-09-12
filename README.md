# tf2-cosmoflow
This program is a CosmoFlow implementation with TensorFlow 2.
The original CosmoFlow paper has been published in SC18 [1] and the source code is available in [4].
This program implements LBANN model proposed in [2].
Jan Balewski's CosmoFlow repository shows where the Cosmology data files are located in Cori storage space [5].
The data parallel training is parallelized using Horovod [3].
For scalable parallel training, an asynchronous I/O module has been implemented based on python multiprocessing package.

* [1]: [Mathuriya et al., CosmoFlow: Using Deep Learning to Learn the Universe at Scale, SC 18](https://arxiv.org/abs/1808.04728)
* [2]: [Oyama et al., Toward Training a Large 3D Cosmological CNN with Hybrid Parallelization, 2019](https://www.osti.gov/servlets/purl/1548314)
* [3]: [Sergeev et al., Horovod: fast and easy distributed deep learning in TensorFlow](https://github.com/horovod/horovod#citation)
* [4]: [CosmoFlow develeoped at NERSC (based on TensorFlow 1.x)](https://urldefense.com/v3/__https://github.com/NERSC/CosmoFlow__;!!Dq0X2DkFhyF93HkjWTBQKhk!BmY4R1jYNnd3fYBHe8ShstXFYUMmeNTaiF8uHMreftTMDBdIaNvO_a2Pc-XM7JA6NYwlPK8EF2s4JlXm$)
* [5]: [Jan Balewski's CosmoFlow repository](https://bitbucket.org/balewski/cosmoflow/src/master/)

## Software Requirements
  * TensorFlow > 2.0.0 (2.2.0 is recommended)
  * Horovod > 0.16

## Instructions to Run on Cori
1. Clone the code first.
```
git clone https://github.com/swblaster/tf2-cosmoflow
```

2. Modify the file path in `test.yaml` file.
* `frameCnt`: the number of samples in each file.
* `numPar`: the number of parameters to be predicted.
* `sourceDir/prj`: the top directory of the data files.
* `subDir`: the sub-directory under `sourceDir/prj`, where the actual files are located.
* `splitIdx/train`: the indices of the training files.
* `splitIdx/test`: the indices of the test files.

The below is an example `test.yaml` file.
```
frameCnt: 128
numPar: 4
parNames: [Omega_m, sigma_8, N_spec, H_0]
sourceDir: {
  prj: /global/cscratch1/sd/slz839/cosmoflow_c1/,
subDir: multiScale_tryG/
splitIdx:
  test: [100, 101, 102, 103, 104, 105, 106, 107]
  train: [20, 21, 22, 23, 24, 25, 26, 27,
          30, 31, 32, 33, 34, 35, 36, 37,
          40, 41, 42, 43, 44, 45, 46, 47,
          50, 51, 52, 53, 54, 55, 56, 57,
          60, 61, 62, 63, 64, 65, 66, 67,
          70, 71, 72, 73, 74, 75, 76, 77,
          80, 81, 82, 83, 84, 85, 86, 87,
          90, 91, 92, 93, 94, 95, 96, 97]
```

3. Run training with appropriate hyper-parameter settings.
For example, the command can be executed by running `sbatch.sh` on Cori at NERSC.
The `myjob.lsf` is the script for training on Summit at OLCF.

* `--epochs`: the number of epochs for training.
* `--batch_size`: the local batch size (the batch size for each process).
* `--overlap`: (0:off / 1:on) disable/enable the I/O overlap feature.
* `--checkpoint`: (0:off / 1:on) disable/enable the checkpointing.
* `--buffer_size`: the I/O buffer size with respect to the number of samples.
* `--record_acc`: (0:off / 1:on) disable/enable the accuracy recording.
* `--config`: the file path for input data configuration.
* `--enable`: (0:off / 1:on) disable/enable evaluation of the trained model.
* `--async_io`: (0:off / 1:on) disable/enable the asynchronous I/O feature.

The below is an example command for training.
```
python3 main.py --epochs=3 \
                --batch_size=4 \
                --overlap=1 \
                --checkpoint=0 \
                --buffer_size=128 \
                --file_shuffle=1 \
                --record_acc=0 \
                --config="test.yaml" \
                --evaluate=0 \
                --async_io=1
```

## Questions/Comments
  * Sunwoo Lee <sunwoolee1.2014@u.northwestern.edu>
  * Wei-keng Liao <wkliao@northwestern.edu>
  * Alex Sim <asim@lbl.gov>
  * John Wu <kwu@lbl.gov>
