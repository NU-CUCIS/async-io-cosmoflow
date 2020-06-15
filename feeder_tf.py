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
from mpi4py import MPI

class cosmoflow_tf:
    def __init__ (self, yaml_file, lock, cv, batch_size = 4):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.lock = lock
        self.cv = cv
        self.batch_size = batch_size
        self.rng = np.random.default_rng()
        self.num_cached_train_batches = 0
        self.num_cached_valid_batches = 0
        self.train_file_index = 0
        self.valid_file_index = 0

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

        num_local_files = int(len(self.train_files) / self.size)
        self.num_train_batches = int(self.batches_per_file * num_local_files)
        print ("Number of training batches in the given " + str(num_local_files) +
               " files: " + str(self.num_train_batches))

        self.num_valid_batches = 0
        for file_path in self.valid_files:
            f = h5py.File(file_path, 'r')
            self.num_valid_batches += f['unitPar'].shape[0]
        self.num_valid_batches = int(self.num_valid_batches / self.batch_size)
        print ("Number of validation batches in the given " + str(len(self.valid_files)) +
               " files: " + str(self.num_valid_batches))

        self.shuffle()

    def shuffle (self):
        # Shuffle the files.
        self.shuffled_index = np.arange(len(self.train_files))
        self.rng.shuffle(self.shuffled_index)

    def read_train_samples (self, batch_id):
        # Read a new file if there are no cached batches.
        if self.num_cached_train_batches == 0:
            start = time.time()
            file_index = self.shuffled_index[self.train_file_index]
            f = h5py.File(self.train_files[file_index], 'r')
            self.train_file_index += 1
            if self.train_file_index == len(self.train_files):
                self.train_file_index = 0
            self.images = f['3Dmap'][:]
            self.labels = f['unitPar'][:]
            f.close()
            self.num_cached_train_batches = int(self.images.shape[0] / self.batch_size)

            # Some files have fewer samples than 128.
            if self.num_cached_train_batches < self.batches_per_file:
                self.batch_list = np.arange(self.batches_per_file)
                for i in range(self.num_cached_train_batches, self.batches_per_file):
                    self.batch_list[i] = (i % self.num_cached_train_batches)
                self.num_cached_train_batches = self.batches_per_file
            else:
                self.batch_list = np.arange(self.num_cached_train_batches)
            self.rng.shuffle(self.batch_list)
            end = time.time()
            print ("i/o: " + str(end - start))
            self.lock.acquire()
            self.cv.notify()
            self.lock.release()

        # Get a mini-batch from the memory buffer.
        self.num_cached_train_batches -= 1
        index = self.batch_list[self.num_cached_train_batches]
        images = self.images[index : index + self.batch_size]
        labels = self.labels[index : index + self.batch_size]
        return images, labels

    def read_valid_samples (self, batch_id):
        # Read a new file if there are no cached batches.
        if self.num_cached_valid_batches == 0:
            if self.valid_file_index == len(self.valid_files):
                print ("Invalid valid_file_index!")
            f = h5py.File(self.valid_files[self.valid_file_index], 'r')
            self.valid_file_index += 1
            self.images = f['3Dmap'][:]
            self.labels = f['unitPar'][:]
            f.close()
            self.num_cached_valid_batches = int(self.images.shape[0] / self.batch_size)

        # Get a mini-batch from the memory buffer.
        index = self.num_cached_valid_batches - 1
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
        dataset = dataset.repeat(1)
        return dataset.__iter__()
