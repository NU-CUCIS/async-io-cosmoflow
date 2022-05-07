# tf2-cosmoflow
This repository contains programs that implement CosmoFlow using TensorFlow 2.x.
The original work CosmoFlow is described in an [SC18 paper](https://dl.acm.org/doi/10.1109/SC.2018.00068).
Its source codes use TensorFlow 1.x and are available on [github](https://github.com/NERSC/CosmoFlow)
and [MLPerf](https://mlcommons.org/en/training-hpc-10/).
Programs in this repo improve CosmoFlow by implementing [LBANN model](https://www.osti.gov/servlets/purl/1548314).
and parallelizing it using [Horovod](https://github.com/horovod/horovod#citation).
To reduce the cost of reading input files and thus improve the end-to-end
training time, we develop an asynchronous I/O module based on python
multiprocessing package to overlap file reads with the computation of model
training.
To obtain input files, users are referrred to [this repo](https://bitbucket.org/balewski/cosmoflow/src/master/)
of continuous software development by Jan Balewski, which points to the location
of input files available on Cori at NERSC.

## Software Requirements
  * TensorFlow > 2.0.0 (2.2.0 is recommended)
  * Horovod > 0.16

## Run Instructions on Cori at NERSC
1. Clone the source codes.
   ```
   git clone https://github.com/swblaster/tf2-cosmoflow
   ```

2. Customize run-time paratemters by modifying the file paths in [./test.yaml](test.yaml).
   * `frameCnt`: the number of samples in each file.
   * `numPar`: the number of parameters to be predicted.
   * `sourceDir/prj`: the top directory of the data files.
   * `subDir`: the sub-directory under `sourceDir/prj`, where the actual files are located.
   * `splitIdx/train`: the indices of the training files.
   * `splitIdx/test`: the indices of the test files.

   Below shows an example of `test.yaml` file.
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

3. Command-line Options
   * `--epochs`: the number of epochs for training.
   * `--batch_size`: the local batch size (the batch size for each process).
   * `--overlap`: (0:off / 1:on) disable/enable the I/O overlap feature.
   * `--checkpoint`: (0:off / 1:on) disable/enable the checkpointing.
   * `--buffer_size`: the I/O buffer size with respect to the number of samples.
   * `--record_acc`: (0:off / 1:on) disable/enable the accuracy recording.
   * `--config`: the file path for input data configuration.
   * `--enable`: (0:off / 1:on) disable/enable evaluation of the trained model.
   * `--async_io`: (0:off / 1:on) disable/enable the asynchronous I/O feature.

4. Start Training
   Jobs of parallel training can be submitted to Cori's batch queue using
   a script file. An example is given in [./sbatch.sh](sbatch.sh).
   File [./myjob.lsf](myjob.lsf) is an example script file for running on
   Summit at OLCF. Below shows an example python command that can be used
   in the job script file.
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

## Publication
* Sunwoo Lee, Qiao Kang, Kewei Wang, Jan Balewski, Alex Sim, Ankit Agrawal, Alok Choudhary, Peter Nugent, Kesheng Wu, and Wei-keng Liao. [Asynchronous I/O Strategy for Large-Scale Deep Learning Applications](https://doi.org/10.1109/HiPC53243.2021.00046). In the 28th International Conference on High-Performance Computing, Data, and Analytics (HiPC), December 2021.

## Development team
  * Northwestern University
    + Sunwoo Lee <<sunwoolee1.2014@u.northwestern.edu>>
    + Kewei Wang <<keweiwang2019@u.northwestern.edu>>
    + Wei-keng Liao <<wkliao@northwestern.edu>>
  * Lawrence Berkeley National Laboratory
    + Alex Sim <<asim@lbl.gov>>
    + Jan Balewski <<balewski@lbl.gov>>
    + Peter Nugent <<penugent@lbl.gov>>
    + John Wu <<kwu@lbl.gov>>

## Questions/Comments
  * Sunwoo Lee <<sunwoolee1.2014@u.northwestern.edu>>
  * Wei-keng Liao <<wkliao@northwestern.edu>>

## Project Funding Supports
This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Scientific Discovery through Advanced Computing ([SciDAC](https://www.scidac.gov)) program. This project is a joint work of Northwestern University and Lawrence Berkeley National Laboratory supported by the [RAPIDS Institute](https://rapids.lbl.gov).
