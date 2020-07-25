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
    def __init__ (self, dataset, do_shuffle = 0, cache_size = 0):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.rng = np.random.default_rng()
        self.dataset = dataset
        self.shuffled_index = np.arange(self.dataset.num_train_files)
        self.cache_size = cache_size
        self.do_shuffle = do_shuffle
        self.num_train_files = len(dataset.train_files)
        self.num_valid_files = len(dataset.valid_files)
        self.file_index = 0
        self.offset = int(self.num_train_files / self.size) * self.rank
        self.data_shape = (128, 128, 128, 128, 12)
        self.label_shape = (128, 4)
        self.num_local_files = int(self.num_train_files / self.size)
        total_cache_size = self.cache_size * self.num_local_files

        if self.cache_size > 0:
            self.cache_flag = np.zeros((self.num_local_files))
            self.data_cache = np.zeros((total_cache_size, 128, 128, 128, 12), dtype='uint16')
            self.label_cache = np.zeros((total_cache_size, 4), dtype='float32')
        print ("R" + str(self.rank) + " will work on "  + str(self.num_local_files) + " files.")

    def run (self, lock, cv, finish,
             num_cached_files,
             num_cached_samples,
             data, label, num_samples):
        num_cached_files.value = 0
        num_cached_samples.value = 0
        prev_write_index = -1
        num_buffers = len(data)

        self.shuffled_index[:] = self.dataset.shared_shuffled_index[:]
        print ("R" + str(self.rank) + " updated shuffled_index, [0] is : " + str(self.shuffled_index[0]))
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

                if self.cache_size > 0:
                    # [I/O] Read the (length - cache_size) samples from the file.
                    if self.cache_flag[self.file_index] == 0:
                        # Read the entire samples from the file.
                        self.cache_flag[self.file_index] = 1
                        data_np = np.frombuffer(data[write_index], dtype = np.uint16).reshape(self.data_shape)
                        np.copyto(data_np[0:length], f['3Dmap'][0:length])
                        label_np = np.frombuffer(label[write_index], dtype = np.float32).reshape(self.label_shape)
                        np.copyto(label_np[0:length], f['unitPar'][0:length])

                        # Cache the last 'cache_size' samples.
                        cache_index = self.file_index * self.cache_size
                        np.copyto(self.data_cache[cache_index: cache_index + self.cache_size], data_np[length - self.cache_size:length])
                        np.copyto(self.label_cache[cache_index: cache_index + self.cache_size], label_np[length - self.cache_size:length])
                    else:
                        # Read only 'length - cache_size' samples.
                        data_np = np.frombuffer(data[write_index], dtype = np.uint16).reshape(self.data_shape)
                        np.copyto(data_np[0:length - self.cache_size], f['3Dmap'][0:length - self.cache_size])
                        label_np = np.frombuffer(label[write_index], dtype = np.float32).reshape(self.label_shape)
                        np.copyto(label_np[0:length - self.cache_size], f['unitPar'][0:length - self.cache_size])

                        # Copy the cached data into the shared buffer.
                        cache_index = self.file_index * self.cache_size
                        np.copyto(data_np[length - self.cache_size:length], self.data_cache[cache_index: cache_index + self.cache_size])
                        np.copyto(label_np[length - self.cache_size:length], self.label_cache[cache_index: cache_index + self.cache_size])
                else:
                    data_np = np.frombuffer(data[write_index], dtype = np.uint16).reshape(self.data_shape)
                    np.copyto(data_np[0:length], f['3Dmap'][0:length])
                    label_np = np.frombuffer(label[write_index], dtype = np.float32).reshape(self.label_shape)
                    np.copyto(label_np[0:length], f['unitPar'][0:length])

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
                    # [TODO] For now, there is no way to shuffle the files in advance when 
                    # the number of local files is smaller than 2 (double buffering).
                    #if self.do_shuffle == 1:
                    #    print ("io module waits...")
                    #    cv.wait()
                    #    self.shuffled_index[:] = self.dataset.shared_shuffled_index[:]
                    #    print ("R" + str(self.rank) + " updated shuffled_index, [0] is : " + str(self.shuffled_index[0]))
            cv.notify()
            while finish.value == 0 and num_cached_files.value == num_buffers:
                cv.wait()
            lock.release()
