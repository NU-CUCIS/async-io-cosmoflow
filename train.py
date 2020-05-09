'''
Sunwoo Lee
<sunwoolee1.2014@u.northwestern.edu>

Northwestern University
'''
import tensorflow as tf
import time
import argparse
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from feeder import cosmoflow
from model import model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type = int, default = 4,
                        help = "number of training samples for each mini-batch")
    parser.add_argument("-e", "--epochs", type = int, default = 1,
                        help = "number of epochs")

    args = parser.parse_args()
    return args

class Trainer:
    def __init__ (self, model, lr = 1e-4, checkpoint_dir = "./checkpoint"):
        self.lr = PiecewiseConstantDecay(boundaries = [100],
                                         values = [1e-4, 5e-5])
        self.loss = MeanAbsoluteError()
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
                                              model = model,
                                              optimizer = Adam(self.lr))
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint = self.checkpoint,
                                                             directory = checkpoint_dir,
                                                             max_to_keep = 3)
        self.resume()

    def resume (self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print ("Model restored from checkpoint at iteration " + str(self.checkpoint.step.numpy()))

    def train_step (self, data, label):
        with tf.GradientTape() as tape:
            prediction = self.checkpoint.model(data, training = True)
            loss = self.loss(label, prediction)
        gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
        return loss

    def train (self, train_dataset, valid_dataset, num_iterations, evaluation_interval):
        loss_mean = Mean()
        self.start_time = time.perf_counter()

        for data, label in train_dataset.take(num_iterations - self.checkpoint.step.numpy()):
            self.checkpoint.step.assign_add(1)
            loss = self.train_step(data, label)
            loss_mean(loss)

            if self.checkpoint.step.numpy() % evaluation_interval == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()
                timing = time.perf_counter() - self.start_time
                print ("Iteration " + str(self.checkpoint.step.numpy()) + "/" + str(num_iterations) + " loss = " + str(loss_value.numpy()) + " acc = " + "timing: " + str(timing) + " sec")
                self.checkpoint_manager.save()
                self.start_time = time.perf_counter()

if __name__ == "__main__":
    args = get_parser()

    # Get the training dataset.
    ds = cosmoflow("test.yaml", batch_size = args.batch_size)
    ds.shuffle()
    train_ds = ds.train_dataset()
    valid_ds = ds.train_dataset()

    # Get the model.
    cosmo_model = model()
    mymodel = cosmo_model.build_model()

    # Perform the training.
    trainer = Trainer(model = mymodel)
    trainer.train(train_ds, valid_ds, num_iterations = 100, evaluation_interval = 10)
