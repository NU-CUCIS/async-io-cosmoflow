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
    def __init__ (self, yaml_file, batch_size = 4):
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

        # Create an index array for data shuffling.
        self.num_train_batches = int((len(self.train_files) * self.samples_per_file) / self.batch_size)
        self.num_valid_batches = int((len(self.valid_files) * self.samples_per_file) / self.batch_size)

        self.shuffle()

    def shuffle (self):
        rng = np.random.default_rng()

        # First, shuffle the files.
        self.train_file_index = np.arange(len(self.train_files))
        rng.shuffle(self.train_file_index)

        # Second, shuffle the batches in each file.
        # Thus, the granularity of shuffling is a local batch.
        batch_index = []
        for i in range(len(self.train_files)):
            batch_list = np.arange(int(self.samples_per_file / self.batch_size))
            rng.shuffle(batch_list)
            batch_index.append(batch_list)
        self.batch_index = np.array(batch_index)

    def read_train_samples (self, batch_id):
        # Pick the current file index.
        my_file_index = self.train_file_index[int(batch_id.numpy() / int(self.samples_per_file / self.batch_size))]
        # Pick the batch index within the selected file.
        my_batch_index = int(batch_id.numpy() % int(self.samples_per_file / self.batch_size))
        index = self.batch_index[my_file_index][my_batch_index] * self.batch_size
        #print ("batch_id: " + str(batch_id.numpy()) + " my_file_index: " + str(my_file_index) + " my_batch_index: " + str(my_batch_index) + " index: " + str(index) + " file: " + self.train_files[my_file_index])

        '''
        TODO: Open all the files before the training.
        '''
        # Read samples [index : index+batch_size]
        f = h5py.File(self.train_files[my_file_index], 'r')
        images = f['3Dmap'][index : index + self.batch_size]
        labels = f['unitPar'][index : index + self.batch_size]
        f.close()
        return images, labels

    def read_valid_samples (self, batch_id):
        # Calculate the file index.
        my_file_index = int(batch_id.numpy() / int(self.samples_per_file / self.batch_size))
        # Calculate the batch index within the selected file.
        my_batch_index = int(batch_id.numpy() % int(self.samples_per_file / self.batch_size))
        index = my_batch_index * self.batch_size

        #print ("Reading samples [" + str(index) + ":" + str(index + self.batch_size) + "] from file " + str(self.valid_files[my_file_index]))
        '''
        TODO: Open all the files before the training.
        '''
        # Read samples [index : index+batch_size]
        f = h5py.File(self.valid_files[my_file_index], 'r')
        images = f['3Dmap'][index : index + self.batch_size]
        labels = f['unitPar'][index : index + self.batch_size]
        f.close()
        return images, labels

    def train_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_train_batches))
        dataset = dataset.map(lambda x: tf.py_function(self.read_train_samples, inp=[x], Tout=[tf.float32, tf.float32]))
        dataset = dataset.repeat()
        return dataset.__iter__()

    def valid_dataset (self):
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.num_valid_batches))
        dataset = dataset.map(lambda x: tf.py_function(self.read_valid_samples, inp=[x], Tout=[tf.float32, tf.float32]))
        dataset = dataset.repeat()
        return dataset.__iter__()
