'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import tensorflow as tf
import argparse
from feeder import cosmoflow

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 8,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()

    ds = cosmoflow("test.yaml", batch_size = args.batch_size)
    ds.shuffle()
    train_ds = ds.train_dataset()
    images, labels = train_ds.next()
