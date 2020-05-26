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
import threading
from tensorflow.keras.utils import Sequence
from mpi4py import MPI

class cosmoflow_keras (Sequence):
    def __init__ (self, yaml_file, batch_size = 8, mode = 'train', rank = 0):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()

        self.batch_size = batch_size
        self.rank = rank
        self.mode = mode
        self.rng = np.random.default_rng()
        self.num_cached_batches = 0
        self.file_index = 0
        self.lock = threading.Lock()
        self.cv = threading.Condition(lock = self.lock)
        self.empty = 1

        self.num_files_to_keep = 1
        self.num_files_in_cache = 0
        self.head = 0
        self.tail = 0
        self.cached_data = [None] * self.num_files_to_keep
        self.cached_label = [None] * self.num_files_to_keep

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
                    if self.mode == 'train':
                        self.files = list(value['train'])
                    elif self.mode == 'valid':
                        self.files = list(value['val'])
                    else:
                        self.files = list(value['test'])

                    self.files = [str(self.prj) + "/" + 
                                  str(self.subdir) + "/" +
                                  "PeterA_2019_05_4parE-rec" +
                                  str(file_name[1]) +
                                  ".h5" for file_name in enumerate(self.files)]

            print ("Number of samples per file: " + str(self.samples_per_file))
            print ("Label size: " + str(self.label_size))
            print ("sourceDir.prj: " + str(self.prj))
            print ("sourceDir.cfs: " + str(self.cfs))
            print ("subDir: " + str(self.subdir))
            if self.mode == 'train':
                print ("training files")
            else:
                print ("validation files")

            for file_path in self.files:
                print (file_path)

        if mode == 'train':
            self.num_local_files = int(len(self.files) / self.size)
            self.offset = self.num_local_files * self.rank
            if (len(self.files) % self.size != 0):
                print ("Number of training files is not divisible by the number of processes!")
                exit()
            self.num_batches = int(self.batches_per_file * self.num_local_files)
            print ("Number of training batches in the given " + str(self.num_local_files) +
                   " files: " + str(self.num_batches))
            self.shuffle()
        else:
            self.num_batches = 0
            for file_path in self.files:
                f = h5py.File(file_path, 'r')
                self.num_batches += f['unitPar'].shape[0]
            self.num_batches = int(self.num_batches / self.batch_size)
            print ("Number of validation batches in the given " + str(len(self.files)) +
                   " files: " + str(self.num_batches))

    def shuffle (self):
        # Shuffle the files.
        self.shuffled_index = np.arange(len(self.files))
        if self.rank == 0:
            self.rng.shuffle(self.shuffled_index)
        self.comm.Bcast(self.shuffled_index, root = 0) 
        print ("R0 shuffled the files... the first file id is " + str(self.shuffled_index[0]))

    def __len__(self):
        return self.num_batches

    def __getitem__(self, input_index = 0):
        # Check if there is a file in the memory buffer.
        self.lock.acquire()
        while self.num_files_in_cache == 0:
            self.cv.wait()
        self.cv.notify()
        self.lock.release()

        # If num_cached_batches is 0 and went through the above wait(),
        # it means that a new file has been loaded by the reader.
        # So, update the num_cached_batches using the head offset.
        if self.num_cached_batches == 0:
            self.num_cached_batches = int(self.cached_data[self.head].shape[0] / self.batch_size)

            # Create the random indices for the loaded batches.
            # Note that some files have fewer samples than 128.
            if self.mode == 'train':
                if self.num_cached_batches < self.batches_per_file:
                    self.batch_list = np.arange(self.batches_per_file)
                    for i in range(self.num_cached_batches, self.batches_per_file):
                        self.batch_list[i] = (i % self.num_cached_batches)
                    self.num_cached_batches = self.batches_per_file
                else:
                    self.batch_list = np.arange(self.num_cached_batches)
                self.rng.shuffle(self.batch_list)
            else:
                self.batch_list = np.arange(self.num_cached_batches)

        ## Read a batch from the cached file.
        self.num_cached_batches -= 1
        index = self.batch_list[self.num_cached_batches]

        images = self.cached_data[self.head][index : index + self.batch_size]
        labels = self.cached_label[self.head][index : index + self.batch_size]

        ## Check if the current file has been all consumed.
        ## If yes, increase the head offset.
        ## Whenever a file is consumed, notify the i/o thread.
        if self.num_cached_batches == 0:
            self.lock.acquire()
            self.num_files_in_cache -= 1
            self.head += 1
            if self.head == self.num_files_to_keep:
                self.head = 0
            self.cv.notify()
            self.lock.release()

        return (images, labels)
