'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import multiprocessing as mp
from multiprocessing import Process, Manager, Lock
from multiprocessing.managers import BaseManager

import argparse
import time
import threading
from tqdm import tqdm
import numpy as np
from mpi4py import MPI

import horovod.tensorflow.keras as hvd
#import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from feeder_tf import cosmoflow_tf
from feeder_keras_async import cosmoflow_keras
#from feeder_keras_sync import cosmoflow_keras
#from feeder_tf import cosmoflow_tf
from model import model

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
    
class io_daemon:
    def __init__ (self, rank, train_dataset, valid_dataset):
        '''
        These datasets are replicates of them in the main process.
        We get these objects to reference the static values only.
        '''
        self.rank = rank
        self.comm = MPI.COMM_WORLD
        self.rng = np.random.default_rng()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.file_index = 0
        self.shuffle()

    def shuffle (self):
        # Shuffle the files.
        self.shuffled_index = np.arange(len(self.train_dataset.files))
        if self.rank == 0:
            self.rng.shuffle(self.shuffled_index)
        self.comm.Bcast(self.shuffled_index, root = 0) 
        print ("R0 shuffled the files... the first file id is " + str(self.shuffled_index[0]))

    def run (self, num_files_in_cache, buffer_index, finish, rank, lock, cv):
        while 1:
            lock.acquire()
            '''
            Multiprocessing.Condition makes the program hang.
            We will work on it later.
            '''
            if num_files_in_cache.value == 0:
                num_files_in_cache.value += 1
                file_index = self.shuffled_index[self.file_index + self.train_dataset.offset]
                self.file_index += 1
                if self.file_index == self.train_dataset.num_local_files:
                    self.file_index = 0
                buffer_index.value = (self.file_index % 2)

                t = time.time()
                print ("R" + str(rank) + " woke up and file_index: " + str(self.file_index) + \
                       " buffer_index in io_thread: " + str(buffer_index.value) + \
                       " increased num_files_in_cache to " + str(num_files_in_cache.value) + \
                       " reading " + self.train_dataset.files[file_index] + " at " + str(t))

            if finish.value == 1:
                print ("R" + str(rank) + " Okay i will go die...")
                break

            lock.release()

            time.sleep(0.5)

            #start = time.time()
            #f = h5py.File(self.dataset.files[file_index], 'r')
            #self.dataset.cached_data[self.dataset.tail] = f['3Dmap'][:]
            #self.dataset.cached_label[self.dataset.tail] = f['unitPar'][:]
            #f.close()
            #end = time.time()

            ## Update the tail offset.
            #self.dataset.lock.acquire()
            #self.dataset.tail += 1
            #if self.dataset.tail == self.dataset.num_files_to_keep:
            #    self.dataset.tail = 0
            #self.dataset.num_files_in_cache += 1
            #self.dataset.cv.notify()
            #self.dataset.lock.release()

            #print ("R" + str(self.dataset.rank) + " Async reader reads files[" + str(self.file_index + self.dataset.offset) + "] " +\
            #       self.dataset.files[file_index] +\
            #       " now, head: " + str(self.dataset.head) +\
            #       ", tail: " + str(self.dataset.tail) + \
            #       " timing: " + str(end - start))

            ## Remember which file was read just now.
            #self.file_index += 1
            #if self.file_index == self.dataset.num_local_files:
            #    self.file_index = 0

if __name__ == "__main__":
    args = get_parser()
    hvd.init()

    lock = mp.Lock()
    cv = mp.Condition(lock = lock)
    num_files_in_cache = mp.Value('i')
    finish = mp.Value('i')
    buffer_index = mp.Value('i')

    train_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'train',
                                    num_files_in_cache = num_files_in_cache,
                                    buffer_index = buffer_index,
                                    finish = finish,
                                    rank = hvd.rank(),
                                    lock = lock,
                                    cv = cv)
    valid_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'valid',
                                    num_files_in_cache = num_files_in_cache,
                                    buffer_index = buffer_index,
                                    finish = finish,
                                    rank = hvd.rank(),
                                    lock = lock,
                                    cv = cv)

    daemon = io_daemon(hvd.rank(), train_dataset, valid_dataset)
    
    io_process = Process(target = daemon.run, args = (num_files_in_cache, buffer_index, finish, hvd.rank(), lock, cv))
    io_process.start()

    # NOTE: have no idea why but it hangs when tf_thread is created as Process.
    # So, let's just call it instead of making a separate process.
    tf_thread(train_dataset, valid_dataset)

    train_dataset.finish.value = 1
    io_process.join()
    print ("All done!")
