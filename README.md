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

2. Modify the file path in `test.yaml` file.

3. Modify the hyper-parameters appropriately and start training.
```
python3 main.py --epochs=3 \
                --batch_size=4 \
                --overlap=1 \
                --checkpoint=0 \
                --cache_size=0 \
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
