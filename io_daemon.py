'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import time
import h5py
import multiprocessing
from mpi4py import MPI

class IOdaemon:
    def __init__ (self, dataset):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.rng = np.random.default_rng()
        self.dataset = dataset

    def run (self, lock, cv, finish):
        while 1:
            lock.acquire()
            while finish.value == 0:
                print ("Okay, I will go sleep...")
                cv.wait()

            print ("R" + str(self.rank) + " why? I woke up!")
            if finish.value == 1:
                print ("R" + str(self.rank) + " Okay, I/O process will terminates...")
                break
            lock.release()
