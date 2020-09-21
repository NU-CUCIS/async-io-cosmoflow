'''
Copyright (C) 2020, Northwestern University and Lawrence Berkeley National Laboratory
See COPYRIGHT notice in top-level directory.
'''
import os
import time
import tensorflow as tf
import yaml
import numpy as np
import h5py
import math
from mpi4py import MPI
#import horovod.tensorflow as hvd
import multiprocessing as mp

class cosmoflow_async:
    def __init__ (self, yaml_file, lock, cv,
                  data, label, num_samples,
                  do_shuffle = 0,
                  batch_size = 4,
                  buffer_size = 128):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        #self.size = hvd.size()
        #self.rank = hvd.rank()
        self.lock = lock
        self.cv = cv
        self.data = data
        self.label = label
        self.num_samples = num_samples
        self.num_buffers = len(data)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.read_index = 0
        self.rng = np.random.default_rng()
        self.do_shuffle = do_shuffle
        self.num_cached_train_batches = 0
        self.num_cached_valid_batches = 0
        self.train_file_index = 0
        self.valid_file_index = 0
        self.data_shape = (self.buffer_size, 128, 128, 128, 12)
        self.label_shape = (self.buffer_size, 4)
        self.file_index = 0

        # Parse the given yaml file and get the top dir and file names.
        with open (yaml_file, "r") as f:
            data = yaml.load(f, Loader = yaml.FullLoader)
            for key, value in data.items():
                if key == 'frameCnt':
                    self.samples_per_file = value
                    self.batches_per_file = int(value / self.batch_size)

                if key == 'numPar':
                    self.label_size = value

                if key == 'sourceDir':
                    self.prj = value['prj']

                if key == 'subDir':
                    self.subdir = value

                if key == 'splitIdx':
                    self.train_files = list(value['train'])
                    self.valid_files = list(value['val'])

                    self.train_files = [str(self.prj) + "/" + 
                                        str(self.subdir) + "/" +
                                        "PeterA_2019_05_4parE-rec" +
                                        str(file_name[1]) +
                                        ".h5" for file_name in enumerate(self.train_files)]
                    self.valid_files = [str(self.prj) + "/" +
                                        str(self.subdir) + "/" +
                                        "PeterA_2019_05_4parE-rec" +
                                        str(file_name[1]) +
                                        ".h5" for file_name in enumerate(self.valid_files)]

            print ("Number of samples per file: " + str(self.samples_per_file))
            print ("Label size: " + str(self.label_size))
            print ("sourceDir.prj: " + str(self.prj))
            print ("subDir: " + str(self.subdir))
        print ("Buffer size: " + str(self.buffer_size) + " samples")

        self.num_train_files = len(self.train_files)
        self.offset = int(self.num_train_files / self.size) * self.rank
        self.shared_shuffled_index = mp.RawArray('i', self.num_train_files)

        # First, calculate the number of local files.
        common = int(self.num_train_files / self.size)
        remainder = self.num_train_files % self.size

        if self.rank < remainder:
            self.num_local_train_files = common + 1
        else:
            self.num_local_train_files = common

        self.num_train_batches = int(self.batches_per_file * self.num_local_train_files)

        # Count the number of local files for validaiton.
        num_local_valid_files = int(math.floor(len(self.valid_files) / self.size))
        local_valid_files_off = num_local_valid_files * self.rank
        if self.rank < (len(self.valid_files) % self.size):
            num_local_valid_files += 1
            local_valid_files_off += self.rank
        else:
            local_valid_files_off += (len(self.valid_files) % self.size)
        self.local_valid_files = self.valid_files[local_valid_files_off:
                                                  local_valid_files_off + num_local_valid_files]

        self.num_valid_batches = 0
        for file_path in self.local_valid_files:
            f = h5py.File(file_path, 'r')
            self.num_valid_batches += f['unitPar'].shape[0]
            f.close()
        self.num_valid_batches = int(math.floor(self.num_valid_batches / self.batch_size))

        self.shuffle()

    def shuffle (self):
        # Shuffle the file index.
        self.shuffled_file_index = np.arange(self.num_train_files)
        self.rng.shuffle(self.shuffled_file_index)
        self.comm.Bcast(self.shuffled_file_index, root = 0) 
        self.shared_shuffled_index[:] = self.shuffled_file_index[:]

        self.shuffled_sample_index = np.arange(self.buffer_size)
        self.rng.shuffle(self.shuffled_sample_index)

    '''
    Training dataset
    '''
    def read_train_sample (self, sample_id):
        # Read a random sample from the buffer.
        sample_index = sample_id.numpy() % self.buffer_size
        sample_index = self.shuffled_sample_index[sample_index]

        data_np = np.frombuffer(self.data[self.read_index], dtype = np.uint16).reshape(self.data_shape)
        label_np = np.frombuffer(self.label[self.read_index], dtype = np.float32).reshape(self.label_shape)
        image = data_np[sample_index]
        label = label_np[sample_index]

        return image, label

    def tf_read_train_sample (self, sample_id):
        image, label = tf.py_function(self.read_train_sample, inp=[sample_id], Tout=[tf.float32, tf.float32])
        return image, label

    def train_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_local_train_files * self.samples_per_file))
        dataset = dataset.map(self.tf_read_train_sample)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        #dataset = dataset.prefetch(4)
        #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset.__iter__()

    def pre_batch (self):
        # Wait if the current buffer is empty.
        self.lock.acquire()
        while self.num_samples[self.read_index].value == 0:
            print ("R" + str(self.rank) + " okay, buffer " + str(self.read_index) + " is empty.. I will wait...")
            self.cv.notify()
            self.cv.wait()
        self.lock.release()

        if self.num_cached_train_batches == 0:
            self.num_cached_train_batches = int(self.buffer_size / self.batch_size)

    def post_batch (self):
        self.num_cached_train_batches -= 1
        if self.num_cached_train_batches == 0:
            self.lock.acquire()
            self.num_samples[self.read_index].value -= self.buffer_size
            self.cv.notify()
            self.lock.release()

            self.read_index += 1
            if self.read_index == self.num_buffers:
                self.read_index = 0

    '''
    Validation dataset
    '''
    def read_valid_samples (self, batch_id):
        # Read a new file if there are no cached batches.
        if self.num_cached_valid_batches == 0:
            if self.valid_file_index == len(self.local_valid_files):
                print ("batch_id: " + str(batch_id) + " Invalid valid_file_index! " + str(self.valid_file_index) + "/" + str(len(self.valid_files)))
            f = h5py.File(self.local_valid_files[self.valid_file_index], 'r')
            self.valid_file_index += 1
            self.images = f['3Dmap'][:]
            self.labels = f['unitPar'][:]
            f.close()
            self.num_cached_valid_batches = int(self.images.shape[0] / self.batch_size)

        # Get a mini-batch from the memory buffer.
        index = (self.num_cached_valid_batches - 1) * self.batch_size
        images = self.images[index : index + self.batch_size]
        labels = self.labels[index : index + self.batch_size]
        self.num_cached_valid_batches -= 1
        return images, labels

    def tf_read_valid_samples (self, batch_id):
        images, labels = tf.py_function(self.read_valid_samples, inp=[batch_id], Tout=[tf.float32, tf.float32])
        images.set_shape([self.batch_size, 128,128,128,12])
        labels.set_shape([self.batch_size, 4])
        return images, labels

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_batches))
        dataset = dataset.map(self.tf_read_valid_samples)
        dataset = dataset.repeat()
        return dataset.__iter__()
