'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import tensorflow as tf
import time
import argparse
import horovod.tensorflow as hvd 
from tqdm import tqdm
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from feeder import cosmoflow
from model import model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 8,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")

    args = parser.parse_args()
    return args

class Trainer:
    def __init__ (self, model, dataset, num_epochs, checkpoint_dir = "./checkpoint"):
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.model = model.build_model()
        self.model.summary()
        self.lr = PiecewiseConstantDecay(boundaries = [100],
                                         values = [1e-4, 5e-5])
        self.loss = MeanSquaredError()
        self.checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0),
                                              model = self.model,
                                              #optimizer = Adam(self.lr))
                                              optimizer = Adam(lr = 1e-4))
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
                                                             directory = checkpoint_dir,
                                                             max_to_keep = 3)
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
        #tape = hvd.DistributedGradientTape(tape)
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
                start = time.perf_counter()
                data, label = train_dataset.next()
                end = time.perf_counter()
                print ("i/o + device transfer: " + str(end - start))

                # Computation
                start = time.perf_counter()
                loss = self.train_step(data, label)
                end = time.perf_counter()
                print ("comp: " + str(end - start))
                loss_mean(loss)

            timing = time.perf_counter() - self.start_time
            train_loss = loss_mean.result()
            loss_mean.reset_states()

            self.checkpoint_manager.save()
            self.dataset.shuffle()

            # Evaluate the current model using the validation data.
            print ("Evaluating the current model using " + str(self.dataset.num_valid_batches) + " validation batches.")
            valid_loss = self.evaluate(valid_dataset, self.dataset.num_valid_batches)

            print ("Epoch " + str(self.checkpoint.epoch.numpy()) +\
                   " training loss = " + str(train_loss.numpy()) +\
                   #" validation loss = " + str(valid_loss.numpy()) +\
                   " training timing: " + str(timing) + " sec")

            # Write the loss values to the output files.
            f = open("loss-train.txt", "a")
            f.write(str(train_loss.numpy()) + "\n")
            f.close()
            f = open("loss-valid.txt", "a")
            f.write(str(valid_loss.numpy()) + "\n")
            f.close()

    def evaluate (self, dataset, num_valid_batches):
        self.dataset.valid_file_index = 0
        loss_mean = Mean()
        for i in tqdm(range(num_valid_batches)):
            data, label = dataset.next()
            prediction = self.checkpoint.model(data)
            loss = self.loss(label, prediction)
            loss_mean(loss)
        return loss_mean.result()

    def call_fit (self):
        train_dataset = self.dataset.train_dataset()
        valid_dataset = self.dataset.valid_dataset()
        self.checkpoint.model.compile(optimizer = self.checkpoint.optimizer, loss = 'mse')
        self.checkpoint.model.fit(train_dataset,
                                  epochs = self.num_epochs,
                                  steps_per_epoch = self.dataset.num_train_batches)
                                  #validation_data = valid_dataset)

if __name__ == "__main__":
    args = get_parser()

    # Get the training dataset.
    dataset = cosmoflow("test.yaml", batch_size = args.batch_size)
    
    # Get the model.
    cosmo_model = model()

    # Perform the training.
    trainer = Trainer(cosmo_model, dataset, args.epochs)
    start = time.time()
    trainer.call_fit()
    end = time.time()
    print ("----------------- fit time: " + str(end - start))
    #trainer.train()
