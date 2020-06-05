'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import time
import numpy as np
import h5py
import multiprocessing
from mpi4py import MPI

class io_daemon:
    def __init__ (self, rank, train_dataset, valid_dataset):
        '''
        These datasets are replicates of them in the main process.
        We get these objects to reference the static values only.
        '''
        self.rank = rank
        self.comm = MPI.COMM_WORLD
        self.rng = np.random.default_rng()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.file_index = 0
        self.prev_write_index = -1
        self.data_buffer_size = 128 * 128 * 128 * 128 * 12
        self.label_buffer_size = 128 * 4
        self.io_start = np.zeros(100)
        self.io_end = np.zeros(100)
        self.index = 0

        self.shuffle()

    def shuffle (self):
        # Shuffle the files.
        self.shuffled_index = np.arange(len(self.train_dataset.files))
        if self.rank == 0:
            self.rng.shuffle(self.shuffled_index)
        self.comm.Bcast(self.shuffled_index, root = 0) 
        print ("R0 shuffled the files... the first file id is " + str(self.shuffled_index[0]))

    def run (self, num_files_in_cache, buffer_index, finish, rank, lock, cv, data0, label0, data1, label1):
        while 1:
            lock.acquire()
            while finish.value == 0 and num_files_in_cache.value == 2:
                cv.wait()

            if finish.value == 1:
                print ("R" + str(rank) + " Okay i will go die...after " + str(self.index) + " i/o operations")
                #name = "R" + str(rank) + "_io_start.txt"
                #f = open(name, "a")
                #for i in range (self.index):
                #    f.write(str(self.io_start[i]) + "\n")
                #f.close()
                #name = "R" + str(rank) + "_io_end.txt"
                #f = open(name, "a")
                #for i in range (self.index):
                #    f.write(str(self.io_end[i]) + "\n")
                #f.close()
                break
            lock.release()

            # Read a new file if any buffer is empty.
            if num_files_in_cache.value < 2:
                # Choose a file to read.
                file_index = self.shuffled_index[self.file_index + self.train_dataset.offset]

                # Choose the buffer to fill in.
                write_index = (self.prev_write_index + 1) % 2
                self.prev_write_index = write_index

                # Read a file.
                start = time.time()
                self.io_start[self.index] = start
                f = h5py.File(self.train_dataset.files[file_index], 'r')
                if write_index == 0:
                    data_np = np.frombuffer(data0, dtype = np.uint16).reshape(self.train_dataset.data_shape)
                    label_np = np.frombuffer(label0, dtype = np.float32).reshape(self.train_dataset.label_shape)
                    np.copyto(data_np, f['3Dmap'][:])
                    np.copyto(label_np, f['unitPar'][:])
                else:
                    data_np = np.frombuffer(data1, dtype = np.uint16).reshape(self.train_dataset.data_shape)
                    label_np = np.frombuffer(label1, dtype = np.float32).reshape(self.train_dataset.label_shape)
                    np.copyto(data_np, f['3Dmap'][:])
                    np.copyto(label_np, f['unitPar'][:])
                f.close()
                end = time.time()
                self.io_end[self.index] = end
                self.index += 1
                print ("R" + str(rank) + " reads " + self.train_dataset.files[file_index] + \
                       " into buffer[" + str(write_index) + "] at " + str(start) + \
                       " i/o time: " + str(end - start))

            # Update the shared status varaibles.
            lock.acquire()
            if num_files_in_cache.value < 2:
                num_files_in_cache.value += 1
                self.file_index += 1
                if self.file_index == self.train_dataset.num_local_files:
                    self.file_index = 0
            cv.notify()
            lock.release()
            time.sleep(0.1)
