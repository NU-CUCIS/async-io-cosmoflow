# tf2-cosmoflow
This program is a CosmoFlow implementation with TensorFlow 2.
The original CosmoFlow paper has been published in SC18 [1].
This program implements LBANN model proposed in [2].
The data parallel training is parallelized using Horovod [3].
For scalable parallel training, an asynchronous I/O module has been implemented based on python multiprocessing package.

* [1]: [Mathuriya et al., CosmoFlow: Using Deep Learning to Learn the Universe at Scale, SC 18](https://arxiv.org/abs/1808.04728)
* [2]: [Oyama et al., Toward Training a Large 3D Cosmological CNN with Hybrid Parallelization, 2019](https://www.osti.gov/servlets/purl/1548314)
* [3]: [Sergeev et al., Horovod: fast and easy distributed deep learning in TensorFlow](https://github.com/horovod/horovod#citation)

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
