'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import os
import tensorflow as tf
import yaml
import h5py

class cosmoflow:
    def __init__ (self, yaml_file, mode = 'trian', batch_size = 4):
        self.mode = mode
        self.batch_size = batch_size

        # Parse the given yaml file and get the top dir and file names.
        with open (yaml_file, "r") as f:
            data = yaml.load(f, Loader = yaml.FullLoader)
            for key, value in data.items():
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

            print ("sourceDir.prj: " + str(self.prj))
            print ("sourceDir.cfs: " + str(self.cfs))
            print ("subDir: " + str(self.subdir))
            print ("training files")
            for i in self.train_files:
                print (i)

            print ("test files")
            for i in self.test_files:
                print (i)
