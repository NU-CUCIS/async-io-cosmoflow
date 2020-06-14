'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>
Northwestern University
'''
import tensorflow as tf
import time
import argparse
from model import model
from feeder_tf import cosmoflow_tf
from train import Trainer
import horovod.tensorflow as hvd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 8,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-o", "--overlap", type = int, default = 0,
                        help = "0: do not overlap I/O with computation, 1: overlap I/O with computation")
    parser.add_argument("-c", "--checkpoint", type = int, default = 0,
                        help = "0: do not checkpoint the model, 1: checkpoint the model")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    cosmo_model = model()
    dataset = cosmoflow_tf("test.yaml", batch_size = args.batch_size)
    trainer = Trainer(cosmo_model, dataset, args.epochs, do_checkpoint = args.checkpoint)
    trainer.train()
    print ("All done!")
