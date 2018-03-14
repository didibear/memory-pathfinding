import tensorflow as tf
import numpy as np
import time
import json
import os

from memory_pathfinding import MemoryPathfindingEnv, MemoryPathfindingEnvConfig
from macn.model import BatchMACN, MACNConfig
from dataset import get_datasets

FLAGS = tf.flags.FLAGS

# Hyperparameter
tf.flags.DEFINE_integer("epochs",           10,    "Number of epochs for training")
tf.flags.DEFINE_integer("batch_per_epoch",  10,   "Number of episodes per epochs")
tf.flags.DEFINE_float(  "learning_rate",    10e-5, "The learning rate")

tf.flags.DEFINE_string("env_spec", "./env_spec.json", MemoryPathfindingEnvConfig.__doc__)
tf.flags.DEFINE_string("model_spec", "./model_spec.json", MACNConfig.__doc__)

tf.flags.DEFINE_integer("batch_size",   64,  "Batch size (batch of episode)")

tf.flags.DEFINE_string('dataset', "./data/dataset.pkl", "Path to dataset file")
tf.flags.DEFINE_string('save', "./model/weights.ckpt", "File to save the weights")
tf.flags.DEFINE_string('load', "./model/weights.ckpt", "File to load the weights")

seq_length = 40
def main(args):
    checks()

    env_spec = spec_from_json(MemoryPathfindingEnvConfig, FLAGS.env_spec)
    env_spec.seq_length

    model_spec = spec_from_json(MACNConfig, FLAGS.model_spec)
    
    macn = BatchMACN.from_spec(model_spec,
        batch_size=FLAGS.batch_size,
        seq_length=env_spec.seq_length
    )

    # y = [batch, labels]
    y = tf.placeholder(tf.int64, shape=[None, None], name='y') # labels : actions {0,1,2,3}

    # Training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=macn.logits, name='cross_entropy')
    loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
    train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate, epsilon=1e-6, centered=True).minimize(loss)

    # Reporting
    y_ = tf.argmax(macn.prob_actions, axis=-1) # predicted action
    nb_errors = tf.reduce_sum(tf.to_float(tf.not_equal(y_, y))) # Number of wrongly selected actions

    def train_on_episode_batch(batch_images, batch_labels):
        _, _loss, _nb_err = sess.run([train_step, loss, nb_errors], feed_dict={macn.X : batch_images, y : batch_labels})
        return _loss, _nb_err
        
    def test_on_episode_batch(batch_images, batch_labels):
        return sess.run([loss, nb_errors], feed_dict={macn.X : batch_images, y : batch_labels})

    trainset, testset = get_datasets(FLAGS.dataset, test_percent=0.1)
    
    # Start training 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if loadfile_exists(FLAGS.load):
            saver.restore(sess, FLAGS.load)
            print("Weights reloaded")
        else:
            sess.run(tf.global_variables_initializer())
       
        print("Start training...")
        for epoch in range(1, FLAGS.epochs + 1):
            start_time = time.time()

            mean_loss, mean_accuracy = compute_on_dataset(sess, env_spec, trainset, train_on_episode_batch)
            
            print('Epoch: {:3d} ({:.1f} s):'.format(epoch, time.time() - start_time))
            print('\t Train Loss: {:.5f} \t Train accuracy: {:.2f}%'.format(mean_loss, 100*(mean_accuracy)))

            saver.save(sess, FLAGS.save)
        print('Training finished.')
        


        print('Testing...')
        mean_loss, mean_accuracy = compute_on_dataset(sess, env_spec, testset, test_on_episode_batch)
        print('Test Accuracy: {:.2f}%'.format(100*(mean_accuracy)))


def compute_on_dataset(sess, env_spec, dataset, compute_episode_batch):
    total_loss = 0
    total_accuracy = 0

    for batch in range(1, FLAGS.batch_per_epoch + 1):
        
        batch_images, batch_labels = dataset.next_episode_batch(FLAGS.batch_size)
        
        loss, nb_err = compute_episode_batch(batch_images, batch_labels)

        accuracy = 1 - (nb_err / (FLAGS.batch_size * env_spec.seq_length))

        total_loss += loss / FLAGS.batch_size
        total_accuracy += accuracy
    
    mean_loss = total_loss / FLAGS.batch_per_epoch
    mean_accuracy = total_accuracy / FLAGS.batch_per_epoch
    return mean_loss, mean_accuracy

        
def loadfile_exists(filepath):
    filename = os.path.basename(filepath)
    for file in os.listdir(os.path.dirname(filepath)):
        if file.startswith(filename):
            return True
    return False

def checks():
    if not os.path.exists(os.path.dirname(FLAGS.save)):
        print("Error : save file cannot be created (need folders) : " + FLAGS.save)
        exit()

def spec_from_json(SpecClass, jsonfile):
    return SpecClass(**json.load(open(jsonfile)))

if __name__ == "__main__":
    tf.app.run()