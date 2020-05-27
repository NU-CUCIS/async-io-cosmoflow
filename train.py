'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
from multiprocessing import Process, Manager, Lock
from multiprocessing.managers import BaseManager

import argparse
import time
import threading
from tqdm import tqdm

import horovod.tensorflow.keras as hvd
#import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from feeder_tf import cosmoflow_tf
#from feeder_keras_async import cosmoflow_keras
from feeder_keras_sync import cosmoflow_keras
#from feeder_tf import cosmoflow_tf
from reader import Reader
from model import model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 8,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")

    args = parser.parse_args()
    return args

#def io():
#    while 1:
#        time.sleep(0.5)
#        t = time.time()
#        print ("--------woke up at " + str(t))
#
#def run():
#    args = get_parser()
#    hvd.init()
#
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, True)
#    if gpus:
#        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#
#    # Get the training dataset.
#    #dataset = cosmoflow_tf("test.yaml", batch_size = args.batch_size)
#    train_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'train', rank = hvd.rank())
#    valid_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'valid', rank = hvd.rank())
#    
#    # Asynchronous reader
#    #reader = Reader(train_dataset)
#    #x = threading.Thread(target = reader.activate)
#    #x.start()
#
#    print ("Main thread....")
#
#    # Get the model.
#    cosmo_model = model()
#
#    # Perform the training.
#    #trainer = Trainer(cosmo_model, dataset, args.epochs)
#    trainer = Trainer(cosmo_model, train_dataset, args.epochs)
#
#    start = time.time()
#    #trainer.train()
#    trainer.call_fit(train_dataset, valid_dataset)
#    end = time.time()
#    print ("----------------- end-to-end time: " + str(end - start))
#
#    #reader.deactivate()
#    #x.join()
#
#class Trainer:
#    def __init__ (self, model, dataset = None, num_epochs = 1, checkpoint_dir = "./checkpoint"):
#        self.num_epochs = num_epochs
#        self.dataset = dataset
#        self.model = model.build_model()
#        self.model.summary()
#        self.lr = PiecewiseConstantDecay(boundaries = [100],
#                                         values = [1e-4, 5e-5])
#        self.loss = MeanSquaredError()
#        self.opt = Adam(lr = 1e-4)
#        self.opt = hvd.DistributedOptimizer(self.opt)
#        self.checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0),
#                                              model = self.model,
#                                              optimizer = self.opt)
#        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
#                                                             directory = checkpoint_dir,
#                                                             max_to_keep = 3)
#        self.checkpoint.model.compile(optimizer = self.checkpoint.optimizer, loss = 'mse', experimental_run_tf_function = False)
#        self.resume()
#
#    def resume (self):
#        if self.checkpoint_manager.latest_checkpoint:
#            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
#            print ("Model restored from checkpoint at epoch " + str(self.checkpoint.epoch.numpy()))
#
#    @tf.function
#    def train_step (self, data, label):
#        with tf.GradientTape() as tape:
#            prediction = self.checkpoint.model(data, training = True)
#            loss = self.loss(label, prediction)
#        tape = hvd.DistributedGradientTape(tape)
#        gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
#        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
#        return loss
#
#    def train (self):
#        train_dataset = self.dataset.train_dataset()
#        valid_dataset = self.dataset.valid_dataset()
#
#        for epoch_id in range(self.num_epochs):
#            self.checkpoint.epoch.assign_add(1)
#            self.dataset.train_file_index = 0
#            loss_mean = Mean()
#            self.start_time = time.perf_counter()
#
#            # Train the model.
#            for i in tqdm(range(self.dataset.num_train_batches)):
#                # I/O
#                start = time.perf_counter()
#                data, label = train_dataset.next()
#                end = time.perf_counter()
#                print ("i/o + device transfer: " + str(end - start))
#
#                # Computation
#                start = time.perf_counter()
#                loss = self.train_step(data, label)
#                end = time.perf_counter()
#                print ("comp: " + str(end - start))
#                loss_mean(loss)
#
#                if epoch_id == 0 and i == 0:
#                    hvd.broadcast_variables(self.checkpoint.model.variables, root_rank=0)
#                    hvd.broadcast_variables(self.opt.variables(), root_rank=0)
#
#            timing = time.perf_counter() - self.start_time
#            train_loss = loss_mean.result()
#            loss_mean.reset_states()
#
#            #if hvd.rank() == 0:
#            #    self.checkpoint_manager.save()
#            self.dataset.shuffle()
#
#            # Evaluate the current model using the validation data.
#            #print ("Evaluating the current model using " + str(self.dataset.num_valid_batches) + " validation batches.")
#            #valid_loss = self.evaluate(valid_dataset, self.dataset.num_valid_batches)
#
#            print ("Epoch " + str(self.checkpoint.epoch.numpy()) +\
#                   " training loss = " + str(train_loss.numpy()) +\
#                   #" validation loss = " + str(valid_loss.numpy()) +\
#                   " training timing: " + str(timing) + " sec")
#
#            # Write the loss values to the output files.
#            #f = open("loss-train.txt", "a")
#            #f.write(str(train_loss.numpy()) + "\n")
#            #f.close()
#            #f = open("loss-valid.txt", "a")
#            #f.write(str(valid_loss.numpy()) + "\n")
#            #f.close()
#
#    def evaluate (self, dataset, num_valid_batches):
#        self.dataset.valid_file_index = 0
#        loss_mean = Mean()
#        for i in tqdm(range(num_valid_batches)):
#            data, label = dataset.next()
#            prediction = self.checkpoint.model(data)
#            loss = self.loss(label, prediction)
#            loss_mean(loss)
#        return loss_mean.result()
#
#    def call_fit (self, train_dataset, valid_dataset):
#        callbacks = [
#            hvd.callbacks.BroadcastGlobalVariablesCallback(0)
#        ]
#        self.checkpoint.model.fit(train_dataset,
#                                  shuffle = False,
#                                  callbacks = callbacks,
#                                  max_queue_size = 0,
#                                  epochs = self.num_epochs,
#                                  workers=1, use_multiprocessing=False,  
#                                  steps_per_epoch = train_dataset.num_batches)
#                                  #validation_data = valid_dataset,
#                                  #validation_steps = valid_dataset.num_batches)

def tf_thread (train_dataset, valid_dataset):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Build a model.
    cosmo_model = model()

    # Perform the training.
    num_epochs = 1
    checkpoint_dir = "./checkpoint"
    compiled_model = cosmo_model.build_model()
    compiled_model.summary()
    lr = PiecewiseConstantDecay(boundaries = [100],
                                values = [1e-4, 5e-5])
    loss = MeanSquaredError()
    opt = Adam(lr = 1e-4)
    opt = hvd.DistributedOptimizer(opt)
    checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0),
                                          model = compiled_model,
                                          optimizer = opt)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint = checkpoint,
                                                    directory = checkpoint_dir,
                                                    max_to_keep = 3)
    checkpoint.model.compile(optimizer = checkpoint.optimizer, loss = 'mse', experimental_run_tf_function = False)

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    ]
    t1 = time.time()
    checkpoint.model.fit(train_dataset,
                         shuffle = False,
                         callbacks = callbacks,
                         max_queue_size = 1,
                         epochs = num_epochs,
                         workers=1, use_multiprocessing=False,  
                         steps_per_epoch = train_dataset.num_batches)
                         #validation_data = valid_dataset,
                         #validation_steps = valid_dataset.num_batches)
    t2 = time.time()
    print ("fit took " + str(t2 - t1) + " and ended at " + str(t2))

def io_thread (train_dataset, valid_dataset):
    t = time.time()
    print ("i/o thread " + str(train_dataset.num_batches) + " at " + str(t))
    
if __name__ == "__main__":
    args = get_parser()
    hvd.init()

    train_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'train', rank = hvd.rank())
    valid_dataset = cosmoflow_keras("test.yaml", batch_size = args.batch_size, mode = 'valid', rank = hvd.rank())

    BaseManager.register('train_dataset', train_dataset)
    BaseManager.register('valid_dataset', valid_dataset)
    manager = BaseManager()
    manager.start()

    io_process = Process(target = io_thread, args = (train_dataset, valid_dataset))
    io_process.start()
    # NOTE: have no idea why but it hangs when tf_thread is created as Process.
    # So, let's just call it instead of making a separate process.
    tf_thread(train_dataset, valid_dataset)

    io_process.join()
    print ("All done!")
