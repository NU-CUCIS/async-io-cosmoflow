'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import yaml
import time
import numpy as np
import h5py
import threading

class Reader:
    def __init__ (self, dataset):
        self.dataset = dataset
        self.file_index = 0
        self.finish = 0
        self.rng = np.random.default_rng()
        self.shuffle()

    def shuffle (self):
        # Shuffle the files.
        self.shuffled_index = np.arange(len(self.dataset.files))
        self.rng.shuffle(self.shuffled_index)

    def activate (self):
        print ("Thread starting for " + str(self.dataset.mode) + " dataset.")
        while 1:
            time.sleep(1)
            self.dataset.lock.acquire()
            num_cached_files = self.dataset.tail - self.dataset.head
            if num_cached_files < 0: num_cached_files += self.dataset.num_files_to_keep

            # Read a file into buffer[tail].
            while (self.finish == 0) and (num_cached_files == self.dataset.num_files_to_keep):
                self.dataset.cv.wait()
            self.dataset.lock.release()

            # First, check if there has been a termination request.
            if self.finish == 1:
                print ("Okay, I will be killed in 2 seconds.")
                time.sleep(2)
                break

            # Check which file should be read into the memory space.
            if self.dataset.mode == 'train':
                file_index = self.shuffled_index[self.file_index]
            else:
                file_index = self.file_index

            f = h5py.File(self.dataset.files[file_index], 'r')
            self.dataset.cached_data[self.dataset.tail] = f['3Dmap'][:]
            self.dataset.cached_label[self.dataset.tail] = f['unitPar'][:]
            f.close()

            # Update the tail offset.
            self.dataset.lock.acquire()
            self.dataset.tail += 1
            if self.dataset.tail == self.dataset.num_files_to_keep:
                self.dataset.tail = 0
            self.dataset.lock.release()

            print ("Async reader reads files[" + str(self.file_index) + "] " + self.dataset.files[file_index] + " now, head: " + str(self.dataset.head) + ", tail: " + str(self.dataset.tail))

            # Remember which file was read just now.
            self.file_index += 1
            if self.file_index == len(self.dataset.files):
                self.file_index = 0

    def deactivate (self):
        self.dataset.lock.acquire()
        # Flag up the empty so that the thread escape the wait().
        self.dataset.empty = 1
        # Flag up the finish so that the thread is killed.
        self.finish = 1
        self.dataset.cv.notify()
        self.dataset.lock.release()
