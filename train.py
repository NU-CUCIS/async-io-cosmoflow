'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>
Northwestern University
'''
import time
import tensorflow as tf
import multiprocessing as mp
import horovod.tensorflow as hvd
from tqdm import tqdm
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

class Trainer:
    def __init__ (self, model, dataset = None, num_epochs = 1, checkpoint_dir = "./checkpoint", do_checkpoint = False):
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.model = model.build_model()
        self.model.summary()
        self.lr = PiecewiseConstantDecay(boundaries = [100],
                                         values = [1e-4, 5e-5])
        self.loss = MeanSquaredError()
        self.opt = Adam(lr = 1e-4)
        self.do_checkpoint = do_checkpoint
        self.checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0),
                                              model = self.model,
                                              optimizer = self.opt)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
                                                             directory = checkpoint_dir,
                                                             max_to_keep = 3)
        #self.checkpoint.model.compile(optimizer = self.checkpoint.optimizer, loss = 'mse', experimental_run_tf_function = False)
        self.checkpoint.model.compile(optimizer = self.checkpoint.optimizer, loss = 'mse')
        self.resume()

    def resume (self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print ("Model restored from checkpoint at epoch " + str(self.checkpoint.epoch.numpy()))

    @tf.function
    def train_step (self, data, label):
        with tf.GradientTape() as tape:
            prediction = self.checkpoint.model(data, training = True)
            loss = self.loss(label, prediction)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
        return loss

    def train (self):
        train_dataset = self.dataset.train_dataset()
        valid_dataset = self.dataset.valid_dataset()

        for epoch_id in range(self.num_epochs):
            self.checkpoint.epoch.assign_add(1)
            self.dataset.train_file_index = 0
            loss_mean = Mean()
            self.start_time = time.perf_counter()

            # Train the model.
            for i in tqdm(range(self.dataset.num_train_batches)):
                # I/O
                data, label = train_dataset.next()

                # Computation
                loss = self.train_step(data, label)
                loss_mean(loss)

                if epoch_id == 0 and i == 0:
                    hvd.broadcast_variables(self.checkpoint.model.variables, root_rank=0)
                    hvd.broadcast_variables(self.opt.variables(), root_rank=0)

            timing = time.perf_counter() - self.start_time
            train_loss = loss_mean.result()
            loss_mean.reset_states()

            if hvd.rank() == 0 and self.do_checkpoint == True:
                self.checkpoint_manager.save()
            self.dataset.shuffle()

            # Evaluate the current model using the validation data.
            #print ("Evaluating the current model using " + str(self.dataset.num_valid_batches) + " validation batches.")
            #valid_loss = self.evaluate(valid_dataset, self.dataset.num_valid_batches)

            print ("Epoch " + str(self.checkpoint.epoch.numpy()) +\
                   " training loss = " + str(train_loss.numpy()) +\
                   #" validation loss = " + str(valid_loss.numpy()) +\
                   " training timing: " + str(timing) + " sec")

            # Write the loss values to the output files.
            #f = open("loss-train.txt", "a")
            #f.write(str(train_loss.numpy()) + "\n")
            #f.close()
            #f = open("loss-valid.txt", "a")
            #f.write(str(valid_loss.numpy()) + "\n")
            #f.close()

    def evaluate (self, dataset, num_valid_batches):
        self.dataset.valid_file_index = 0
        loss_mean = Mean()
        for i in tqdm(range(num_valid_batches)):
            data, label = dataset.next()
            prediction = self.checkpoint.model(data)
            loss = self.loss(label, prediction)
            loss_mean(loss)
        return loss_mean.result()
