'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import time
import h5py
import numpy as np
import multiprocessing
from mpi4py import MPI

class IOdaemon:
    def __init__ (self, dataset, buffer_size = 1):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.rng = np.random.default_rng()
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.num_train_files = len(dataset.train_files)
        self.num_valid_files = len(dataset.valid_files)
        self.file_index = 0
        self.offset = int(self.num_train_files / self.size) * self.rank
        self.data_shape = (128, 128, 128, 128, 12)
        self.label_shape = (128, 4)
        self.num_local_files = int(self.num_train_files / self.size)
        print ("R" + str(self.rank) + " will work on "  + str(self.num_local_files) + " files.")

        self.shuffle()

    def shuffle (self):
        # Shuffle the file index.
        self.shuffled_index = np.arange(self.num_train_files)
        if self.rank == 0:
            self.rng.shuffle(self.shuffled_index)
        self.comm.Bcast(self.shuffled_index, root = 0) 
        print ("R" + str(self.rank) + " shuffled the files... the first file ID is " +
               str(self.shuffled_index[0]))

    def run (self, lock, cv, finish,
             num_cached_files,
             num_cahced_samples,
             data, label, num_samples):
        num_cached_files.value = 0
        num_cached_samples.value = 0
        prev_write_index = -1
        num_buffers = len(data)
        print ("Number of buffers: " + str(num_buffers))

        while 1:
            if finish.value == 1:
                print ("R" + str(self.rank) + " Okay, I/O process will terminates...")
                break

            if num_cached_files.value < num_buffers:
                # Choose a file to read.
                file_index = self.shuffled_index[self.file_index + self.offset]

                # Choose a buffer to fill in.
                write_index = (prev_write_index + 1) % num_buffers
                prev_write_index = write_index

                # Read a file into the chosen buffer.
                start = time.time()
                f = h5py.File(self.dataset.train_files[file_index], 'r')

                num_samples[write_index].value = f['3Dmap'].shape[0]
                length = num_samples[write_index].value

                # Find the shape of the data and label.
                data_np = np.frombuffer(data[write_index], dtype = np.uint16).reshape(self.data_shape)
                np.copyto(data_np[0:length], f['3Dmap'][:])
                label_np = np.frombuffer(label[write_index], dtype = np.float32).reshape(self.label_shape)
                np.copyto(label_np[0:length], f['unitPar'][:])

                f.close()
                end = time.time()
                print ("R" + str(self.rank) + " reads " + self.dataset.train_files[file_index] + \
                       " into buffer[" + str(write_index) + "] at " + str(start) + \
                       " i/o time: " + str(end - start))

            # Update the shared status.
            lock.acquire()
            if num_cached_files.value < num_buffers:
                num_cached_files.value += 1
                self.file_index += 1
                if self.file_index == self.num_local_files:
                    self.file_index = 0
            cv.notify()
            while finish.value == 0 and num_cached_files.value == num_buffers:
                cv.wait()
            lock.release()
