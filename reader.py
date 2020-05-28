'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import yaml
import time
import numpy as np
import h5py
import multiprocessing

def io_thread (dataset):
    print ("Thread starting for " + str(dataset.mode) + " dataset.")
    finish = 0
    file_index = 0
    while 1:
        dataset.lock.acquire()
        # Read a file into buffer[tail].
        while (finish == 0) and (dataset.num_files_in_cache == dataset.num_files_to_keep):
            t = time.time()
            print ("Okay, I will sleep... at " + str(t))
            dataset.cv.wait()
        dataset.lock.release()

        # First, check if there has been a termination request.
        if finish == 1:
            print ("Okay, I will go die...")
            break

        # Check which file should be read into the memory space.
        if dataset.mode == 'train':
            index = dataset.shuffled_index[file_index + dataset.offset]
        else:
            index = file_index

        start = time.time()
        f = h5py.File(dataset.files[index], 'r')
        dataset.cached_data[dataset.tail] = f['3Dmap'][:]
        dataset.cached_label[dataset.tail] = f['unitPar'][:]
        f.close()
        end = time.time()

        # Update the tail offset.
        dataset.lock.acquire()
        dataset.tail += 1
        if dataset.tail == dataset.num_files_to_keep:
            dataset.tail = 0
        dataset.num_files_in_cache += 1
        dataset.cv.notify()
        dataset.lock.release()

        print ("R" + str(dataset.rank) + " Async reader reads files[" + str(file_index + dataset.offset) + "] " +\
               dataset.files[index] +\
               " now, head: " + str(dataset.head) +\
               ", tail: " + str(dataset.tail) + \
               " timing: " + str(end - start))

        # Remember which file was read just now.
        file_index += 1
        if file_index == dataset.num_local_files:
            file_index = 0

class Reader:
    def __init__ (self, dataset):
        self.dataset = dataset
        print ("R" + str(self.dataset.rank) + " offset: " + str(self.dataset.offset))
        self.file_index = 0
        self.finish = 0
        self.rng = np.random.default_rng()

    def run (self):
        print ("Thread starting for " + str(self.dataset.mode) + " dataset.")
        while 1:
            self.dataset.lock.acquire()
            # Read a file into buffer[tail].
            while (self.finish == 0) and (self.dataset.num_files_in_cache == self.dataset.num_files_to_keep):
                t = time.time()
                print ("Okay, I will sleep... at " + str(t))
                self.dataset.cv.wait()
            self.dataset.lock.release()

            # First, check if there has been a termination request.
            if self.finish == 1:
                print ("Okay, I will go die...")
                break

            # Check which file should be read into the memory space.
            if self.dataset.mode == 'train':
                file_index = self.dataset.shuffled_index[self.file_index + self.dataset.offset]
            else:
                file_index = self.file_index

            start = time.time()
            f = h5py.File(self.dataset.files[file_index], 'r')
            self.dataset.cached_data[self.dataset.tail] = f['3Dmap'][:]
            self.dataset.cached_label[self.dataset.tail] = f['unitPar'][:]
            f.close()
            end = time.time()

            # Update the tail offset.
            self.dataset.lock.acquire()
            self.dataset.tail += 1
            if self.dataset.tail == self.dataset.num_files_to_keep:
                self.dataset.tail = 0
            self.dataset.num_files_in_cache += 1
            self.dataset.cv.notify()
            self.dataset.lock.release()

            print ("R" + str(self.dataset.rank) + " Async reader reads files[" + str(self.file_index + self.dataset.offset) + "] " +\
                   self.dataset.files[file_index] +\
                   " now, head: " + str(self.dataset.head) +\
                   ", tail: " + str(self.dataset.tail) + \
                   " timing: " + str(end - start))

            # Remember which file was read just now.
            self.file_index += 1
            if self.file_index == self.dataset.num_local_files:
                self.file_index = 0

    def deactivate (self):
        self.dataset.lock.acquire()
        # Flag up the empty so that the thread escape the wait().
        self.dataset.empty = 1
        # Flag up the finish so that the thread is killed.
        self.finish = 1
        self.dataset.cv.notify()
        self.dataset.lock.release()
