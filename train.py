'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import argparse
import time
import threading
from tqdm import tqdm
import numpy as np
import h5py
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import multiprocessing as mp
from mpi4py import MPI
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from feeder_tf import cosmoflow_tf
from feeder_keras_async import cosmoflow_keras
from model import model
from reader import io_daemon
#import horovod.tensorflow as hvd
#from feeder_keras_sync import cosmoflow_keras
#from feeder_tf import cosmoflow_tf

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 8,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")

    args = parser.parse_args()
    return args

def tf_thread (train_dataset, valid_dataset):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Build a model.
    cosmo_model = model()

    # Perform the training.
    num_epochs = 1
    checkpoint_dir = "./checkpoint"
    compiled_model = cosmo_model.build_model()
    compiled_model.summary()
    lr = PiecewiseConstantDecay(boundaries = [100],
                                values = [1e-4, 5e-5])
    loss = MeanSquaredError()
    opt = Adam(lr = 1e-4)
    opt = hvd.DistributedOptimizer(opt)
    checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0),
                                          model = compiled_model,
                                          optimizer = opt)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint = checkpoint,
                                                    directory = checkpoint_dir,
                                                    max_to_keep = 3)
    checkpoint.model.compile(optimizer = checkpoint.optimizer, loss = 'mse', experimental_run_tf_function = False)

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    ]
    t1 = time.time()
    checkpoint.model.fit(train_dataset,
                         shuffle = False,
                         callbacks = callbacks,
                         max_queue_size = 1,
                         epochs = num_epochs,
                         workers=1, use_multiprocessing=False,  
                         steps_per_epoch = train_dataset.num_batches)
                         #validation_data = valid_dataset,
                         #validation_steps = valid_dataset.num_batches)
    t2 = time.time()
    print ("fit took " + str(t2 - t1) + " and ended at " + str(t2))
    
if __name__ == "__main__":
    args = get_parser()
    hvd.init()

    data_per_file_size = 128 * 128 * 128 * 128 * 12
    label_per_file_size = 128 * 4

    lock = mp.Lock()
    cv = mp.Condition(lock = lock)
    num_files_in_cache = mp.Value('i')
    finish = mp.Value('i')
    buffer_index = mp.Value('i')
    data0 = mp.RawArray('H', data_per_file_size)
    label0 = mp.RawArray('f', label_per_file_size)
    data1 = mp.RawArray('H', data_per_file_size)
    label1 = mp.RawArray('f', label_per_file_size)

    train_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'train',
                                    num_files_in_cache = num_files_in_cache,
                                    buffer_index = buffer_index,
                                    finish = finish,
                                    rank = hvd.rank(),
                                    lock = lock,
                                    cv = cv,
                                    data0 = data0,
                                    label0 = label0,
                                    data1 = data1,
                                    label1 = label1)
    valid_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'valid',
                                    num_files_in_cache = num_files_in_cache,
                                    buffer_index = buffer_index,
                                    finish = finish,
                                    rank = hvd.rank(),
                                    lock = lock,
                                    cv = cv,
                                    data0 = data0,
                                    label0 = label0,
                                    data1 = data1,
                                    label1 = label1)

    daemon = io_daemon(hvd.rank(), train_dataset, valid_dataset)
    
    io_process = mp.Process(target = daemon.run, args = (num_files_in_cache, buffer_index, finish, hvd.rank(), lock, cv, data0, label0, data1, label1))
    io_process.start()

    # NOTE: have no idea why but it hangs when tf_thread is created as Process.
    # So, let's just call it instead of making a separate process.
    tf_thread(train_dataset, valid_dataset)

    train_dataset.finish.value = 1
    io_process.join()
    print ("All done!")
