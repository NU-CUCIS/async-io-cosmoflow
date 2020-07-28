'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>
Northwestern University
'''
import tensorflow as tf
import time
import argparse
from model import model
from feeder import cosmoflow
from train import Trainer
from io_daemon import IOdaemon
import multiprocessing as mp
import numpy as np
import horovod.tensorflow as hvd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 8,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-o", "--overlap", type = int, default = 1,
                        help = "0: do not overlap I/O with computation, 1: overlap I/O with computation")
    parser.add_argument("-p", "--checkpoint", type = int, default = 0,
                        help = "0: do not checkpoint the model, 1: checkpoint the model")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")
    parser.add_argument("-c", "--cache_size", type = int, default = 0,
                        help = "cache size with respect to the number of samples")
    parser.add_argument("-f", "--file_shuffle", type = int, default = 0,
                        help = "shuffle the files across the processes")
    parser.add_argument("-r", "--record_acc", type = int, default = 0,
                        help = "write the loss and accuracy into output files")
    parser.add_argument("-v", "--evaluate", type = int, default = 0,
                        help = "evaluate the model after every epoch")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()

    # Initialize Horovod.tensorflow.
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # multiprocessing synchronization
    lock = mp.Lock()
    cv = mp.Condition(lock = lock)
    finish = mp.Value('i')
    num_cached_files = mp.Value('i')
    num_cached_samples = mp.Value('i')
    finish.value = 0
    num_cached_files.value = 0
    num_cached_samples.value = 0
    data_buffer_size = 128 * 128 * 128 * 128 * 12
    label_buffer_size = 128 * 4
    if args.overlap == 0:
        num_buffers = 1
    else:
        num_buffers = 2

    print ("File shuffling: " + str(args.file_shuffle))
    data = []
    label = []
    num_samples = []
    for i in range(num_buffers):
        data.append(mp.RawArray('H', data_buffer_size))
        label.append(mp.RawArray('f', label_buffer_size))
        num_samples.append(mp.Value('i'))

    # Initialize model, dataset, and trainer.
    cosmo_model = model()
    dataset = cosmoflow("test.yaml", lock, cv,
                        num_cached_files,
                        num_cached_samples,
                        data, label, num_samples,
                        batch_size = args.batch_size)

    # Initialize the I/O daemon.
    async_io_module = IOdaemon(dataset, args.file_shuffle, args.cache_size)
    trainer = Trainer(cosmo_model,
                      async_io_module,
                      dataset,
                      do_shuffle = args.file_shuffle,
                      num_epochs = args.epochs,
                      do_checkpoint = args.checkpoint,
                      do_record_acc = args.record_acc,
                      do_evaluate = args.evaluate)

    io_process = mp.Process(target = async_io_module.run,
                            args = (lock, cv, finish,
                                    num_cached_files,
                                    num_cached_samples,
                                    data, label, num_samples))
    io_process.start()

    # Start the training.
    trainer.train()

    lock.acquire()
    finish.value = 1
    cv.notify()
    lock.release()
    io_process.join()
    print ("All done!")
