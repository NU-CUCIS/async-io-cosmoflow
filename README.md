# tf2-cosmoflow
This program is a CosmoFlow implementation with TensorFlow 2.
The original CosmoFlow paper has been published in SC18 [1] and the source code is available in [4].
This program implements LBANN model proposed in [2].
The data parallel training is parallelized using Horovod [3].
For scalable parallel training, an asynchronous I/O module has been implemented based on python multiprocessing package.

* [1]: [Mathuriya et al., CosmoFlow: Using Deep Learning to Learn the Universe at Scale, SC 18](https://arxiv.org/abs/1808.04728)
* [2]: [Oyama et al., Toward Training a Large 3D Cosmological CNN with Hybrid Parallelization, 2019](https://www.osti.gov/servlets/purl/1548314)
* [3]: [Sergeev et al., Horovod: fast and easy distributed deep learning in TensorFlow](https://github.com/horovod/horovod#citation)
* [4]: [CosmoFlow develeoped at NERSC (based on TensorFlow 1.x)](https://urldefense.com/v3/__https://github.com/NERSC/CosmoFlow__;!!Dq0X2DkFhyF93HkjWTBQKhk!BmY4R1jYNnd3fYBHe8ShstXFYUMmeNTaiF8uHMreftTMDBdIaNvO_a2Pc-XM7JA6NYwlPK8EF2s4JlXm$)

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

3. Modify the hyper-parameters appropriately and start training.
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
