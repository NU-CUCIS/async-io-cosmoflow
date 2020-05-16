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
        self.finish = 0

    def activate (self):
        print ("Thread starting for " + str(self.dataset.mode) + " dataset.")
        while 1:
            time.sleep(1)
            self.dataset.lock.acquire()
            while not self.dataset.empty:
                self.dataset.cv.wait()
            print ("passed wait.")
            self.dataset.lock.release()
            if self.finish == 1:
                print ("Okay, I will be killed in 2 seconds.")
                time.sleep(2)
                break
        print ("Thread end.")

    def deactivate (self):
        print ("I will get a lock to change the finish flag here!")
        self.dataset.lock.acquire()
        # Flag up the empty so that the thread escape the wait().
        self.dataset.empty = 1
        # Flag up the finish so that the thread is killed.
        self.finish = 1
        print ("I changed the finish flag here!")
        self.dataset.cv.notify()
        self.dataset.lock.release()
