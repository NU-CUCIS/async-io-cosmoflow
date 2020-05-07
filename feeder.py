'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import os
import tensorflow as tf
import yaml
import numpy as np
import h5py

class cosmoflow:
    def __init__ (self, yaml_file, mode = 'trian', batch_size = 4):
        self.mode = mode
        self.batch_size = batch_size

        # Parse the given yaml file and get the top dir and file names.
        with open (yaml_file, "r") as f:
            data = yaml.load(f, Loader = yaml.FullLoader)
            for key, value in data.items():
                if key == 'frameCnt':
                    self.samples_per_file = value

                if key == 'numPar':
                    self.label_size = value

                if key == 'sourceDir':
                    self.prj = value['prj']
                    self.cfs = value['cfs']
                    
                if key == 'subDir':
                    self.subdir = value

                if key == 'splitIdx':
                    self.train_files = list(value['train'])
                    self.test_files = list(value['test'])

                    self.train_files = [str(self.prj) + "/" + 
                                        str(self.subdir) + "/" +
                                        "PeterA_2019_05_4parE-rec" +
                                        str(file_name[1]) +
                                        ".h5" for file_name in enumerate(self.train_files)]
                    self.test_files = [str(self.prj) + "/" +
                                       str(self.subdir) + "/" +
                                       "PeterA_2019_05_4parE-rec" +
                                       str(file_name[1]) +
                                       ".h5" for file_name in enumerate(self.test_files)]
                    '''
                    self.train_files = [str(file_name[1]) for file_name in enumerate(self.train_files)]
                    self.test_files = [str(file_name[1]) for file_name in enumerate(self.test_files)]
                    '''

            print ("Number of samples per file: " + str(self.samples_per_file))
            print ("Label size: " + str(self.label_size))
            print ("sourceDir.prj: " + str(self.prj))
            print ("sourceDir.cfs: " + str(self.cfs))
            print ("subDir: " + str(self.subdir))
            print ("training files")
            for file_path in self.train_files:
                print (file_path)

            print ("test files")
            for file_path in self.test_files:
                print (file_path)

        # Create an index array for data shuffling.
        num_files = len(self.train_files)
        self.num_local_batches = int((num_files * self.samples_per_file) / self.batch_size)
        #self.batch_index = np.zeros((self.num_local_batches))

    def shuffle(self):
        rng = np.random.default_rng()

        # First, shuffle the files.
        self.file_index = np.arange(len(self.train_files))
        rng.shuffle(self.file_index)

        # Second, shuffle the batches in each file.
        # Thus, the granularity of shuffling is a local batch.
        batch_index = []
        for i in range(len(self.train_files)):
            batch_list = np.arange(int(self.samples_per_file / self.batch_size))
            rng.shuffle(batch_list)
            batch_index.append(batch_list)
        self.batch_index = np.array(batch_index)
