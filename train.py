'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import tensorflow as tf
from feeder import cosmoflow

if __name__ == "__main__":
    ds = cosmoflow("test.yaml")
    ds.shuffle()
