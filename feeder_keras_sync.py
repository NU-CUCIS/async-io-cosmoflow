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

        self.rank = rank
        self.batch_size = batch_size
        self.mode = mode
        self.rng = np.random.default_rng()
        self.num_cached_batches = 0
        self.lock = threading.Lock()
        self.cv = threading.Condition(lock = self.lock)
        self.num_files_in_cache = 1
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
            #self.num_batches = int(self.batches_per_file * len(self.files))
            #print ("Number of training batches in the given " + str(len(self.files)) +
            #       " files: " + str(self.num_batches))
            #self.shuffle()
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
        self.rng.shuffle(self.shuffled_index)
        self.comm.Bcast(self.shuffled_index, root = 0) 
        print ("R0 shuffled the files... the first file id is " + str(self.shuffled_index[0]))

    def __len__(self):
        return self.num_batches

    def __getitem__(self, input_index = 0):
        # Read a new file if there are no cached batches.
        #print ("input_index: " + str(input_index) + " and now " + str(self.num_cached_batches) + " batches are in queue.")
        t = time.time()
        print ("R" + str(self.rank) + " getitem at " + str(t))
        if self.num_cached_batches == 0:
            start = time.time()
            if self.mode == 'train':
                file_index = self.shuffled_index[self.file_index + self.offset]
            else:
                file_index = self.file_index

            f = h5py.File(self.files[file_index], 'r')
            self.cached_data = f['3Dmap'][:]
            self.cached_label = f['unitPar'][:]
            f.close()

            self.file_index += 1
            if self.file_index == len(self.files):
                self.file_index = 0
            self.num_cached_batches = int(self.cached_data.shape[0] / self.batch_size)

            # Some files have fewer samples than 128.
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
            end = time.time()
            print ("R" + str(self.rank) + " i/o: " + str(end - start) + " read " + self.files[file_index] + " at " + str(end))

        # Get a mini-batch from the memory buffer.
        self.num_cached_batches -= 1
        index = self.batch_list[self.num_cached_batches]
        images = self.cached_data[index : index + self.batch_size]
        labels = self.cached_label[index : index + self.batch_size]
        return (images, labels)
