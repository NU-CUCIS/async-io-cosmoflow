'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import os
import time
import tensorflow as tf
import yaml
import numpy as np
import h5py
import math
from mpi4py import MPI
import multiprocessing as mp

class cosmoflow_tf:
    def __init__ (self, yaml_file, lock, cv,
                  num_cached_files,
                  num_cached_samples,
                  data, label, num_samples,
                  do_shuffle = 0,
                  batch_size = 4):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.lock = lock
        self.cv = cv
        self.num_cached_files = num_cached_files
        self.num_cached_samples = num_cached_samples
        self.data = data
        self.label = label
        self.num_samples = num_samples
        self.num_buffers = len(data)
        self.batch_size = batch_size
        self.read_index = 0
        self.rng = np.random.default_rng()
        self.do_shuffle = do_shuffle
        self.num_cached_train_batches = 0
        self.num_cached_valid_batches = 0
        self.train_file_index = 0
        self.valid_file_index = 0
        self.data_shape = (128, 128, 128, 128, 12)
        self.label_shape = (128, 4)

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
                    self.cfs = value['cfs']
                    
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
            print ("sourceDir.cfs: " + str(self.cfs))
            print ("subDir: " + str(self.subdir))
            print ("training files")
            for file_path in self.train_files:
                print (file_path)

            print ("validation files")
            for file_path in self.valid_files:
                print (file_path)

        self.num_train_files = len(self.train_files)
        self.shared_shuffled_index = mp.RawArray('i', self.num_train_files)
        num_local_files = int(math.floor(self.num_train_files / self.size))
        self.num_train_batches = int(self.batches_per_file * num_local_files)
        print ("Number of training batches in the given " + str(num_local_files) +
               " files: " + str(self.num_train_batches))

        # Calculate the local file offsets and lengths.
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
        self.num_valid_batches = int(math.floor(self.num_valid_batches / self.batch_size))
        print ("Number of validation batches in the given " + str(len(self.valid_files)) +
               " files: " + str(self.num_valid_batches))
        self.shuffle()

    def shuffle (self):
        # Shuffle the file index.
        shuffled_index = np.arange(self.num_train_files)
        if self.rank == 0:
            self.rng.shuffle(shuffled_index)
        self.comm.Bcast(shuffled_index, root = 0) 
        print ("R" + str(self.rank) + " shuffled the files... the first file ID is " +
               str(shuffled_index[0]))
        self.lock.acquire()
        self.shared_shuffled_index[:] = shuffled_index[:]
        self.cv.notify()
        self.lock.release()

    def read_train_samples (self, batch_id):
        self.lock.acquire()
        while self.num_cached_files.value == 0:
            t = time.time()
            print ("R" + str(self.rank) + " okay, getitem will wait... at " + str(t))
            self.cv.notify()
            self.cv.wait()
        self.lock.release()

        # This condition is for the case where a file has been
        # consumed in the previous iteration.
        # Because a new file has been loaded, let's update the
        # number of batches in the buffer.
        if self.num_cached_train_batches == 0:
            num_samples = self.num_samples[self.read_index].value
            self.num_cached_train_batches = int(num_samples / self.batch_size)

            self.batch_list = np.arange(self.batches_per_file)
            # In case the file contains fewer samples than 128,
            # fill in the memory buffer with the first samples.
            if self.num_cached_train_batches < self.batches_per_file:
                for i in range(self.num_cached_train_batches, self.batches_per_file):
                    self.batch_list[i] = i % self.num_cached_train_batches
                self.num_cached_train_batches = self.batches_per_file
            self.rng.shuffle(self.batch_list)

        self.num_cached_train_batches -= 1
        index = self.batch_list[self.num_cached_train_batches] * self.batch_size

        data_np = np.frombuffer(self.data[self.read_index], dtype = np.uint16).reshape(self.data_shape)
        label_np = np.frombuffer(self.label[self.read_index], dtype = np.float32).reshape(self.label_shape)
        images = data_np[index:index + self.batch_size]
        labels = label_np[index:index + self.batch_size]

        # If the current batch is the last batch of the file,
        # Update the read_index and let I/O module know it.
        if self.num_cached_train_batches == 0:
            self.lock.acquire()
            self.num_cached_files.value -= 1
            self.read_index += 1
            if self.read_index == self.num_buffers:
                self.read_index = 0
            self.cv.notify()
            self.lock.release()
        return images, labels

    def read_valid_samples (self, batch_id):
        # Read a new file if there are no cached batches.
        if self.num_cached_valid_batches == 0:
            if self.valid_file_index == len(self.valid_files):
                print ("batch_id: " + str(batch_id) + " Invalid valid_file_index! " + str(self.valid_file_index) + "/" + str(len(self.valid_files)))
            f = h5py.File(self.valid_files[self.valid_file_index], 'r')
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

    def tf_read_train_samples (self, batch_id):
        images, labels = tf.py_function(self.read_train_samples, inp=[batch_id], Tout=[tf.float32, tf.float32])
        images.set_shape([self.batch_size, 128,128,128,12])
        labels.set_shape([self.batch_size, 4])
        return images, labels

    def tf_read_valid_samples (self, batch_id):
        images, labels = tf.py_function(self.read_valid_samples, inp=[batch_id], Tout=[tf.float32, tf.float32])
        images.set_shape([self.batch_size, 128,128,128,12])
        labels.set_shape([self.batch_size, 4])
        return images, labels

    def train_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_train_batches))
        dataset = dataset.map(self.tf_read_train_samples)
        dataset = dataset.repeat()
        return dataset.__iter__()

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_batches))
        dataset = dataset.map(self.tf_read_valid_samples)
        dataset = dataset.repeat()
        return dataset.__iter__()
