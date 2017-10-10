import tensorflow as tf
from preprocessing import Preprocessor
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

class MNISTReconstruct:

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        self.weights = dict()
        self._create_network()
        self.sess = tf.InteractiveSession()
        self._initialize_weights()
        self.saver = tf.train.Saver(self.weights)

    def post_process(self, image):
        image = np.rot90(image)
        # Normalize output image for display: Set minimum value of image to 0;
        # then scale maximum to 255
        image -= image.min()
        if image.max() == 0.0:
            return image
        image = (image / image.max()) * 255
        return image

    ###
    # NEURAL NETWORK WRAPPER FUNCTIONS
    ###
    def _weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial, name=name)
        self.weights[name] = var
        return var

    def _bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        var = tf.Variable(initial, name=name)
        self.weights[name] = var
        return var

    def _conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    ###
    # NETWORK ARCHITECTURE
    ###
    def _create_network(self):

        print("Creating network model...")

        x_image = tf.reshape(self.x, [-1,28,28,1])

        W_conv1 = self._weight_variable([5, 5, 1, 32], "W_conv1")
        b_conv1 = self._bias_variable([32], "b_conv1")
        h_conv1 = tf.nn.relu(self._conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        W_conv2 = self._weight_variable([5, 5, 32, 64], "W_conv2")
        b_conv2 = self._bias_variable([64], "b_conv2")
        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        W_fc1 = self._weight_variable([7 * 7 * 64, 784], "W_fc1")
        b_fc1 = self._bias_variable([784], "b_fc1")
        h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        #W_fc2 = weight_variable([784, 10])
        #b_fc2 = bias_variable([10])

        #y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #sig_out = tf.sigmoid(y_out)
        self.out_image = tf.reshape(h_fc1_drop, [-1, 28, 28, 1])

        self.train_error = tf.reduce_mean(tf.square(self.out_image - x_image))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.train_error)

        print("Finished creating network model.")

    def _initialize_weights(self):
        # Returns an intialization op that is executed first in the session
        self.sess.run(tf.initialize_all_variables())

    def save(self, path, step):
        # Save current values of session variables, return the save path
        save_path = self.saver.save(self.sess, path, global_step=step)
        print("Model saved as :%s"%(save_path))

    def load(self):
        # Returns CheckpointState proto from "checkpoint" file
        ckpt = tf.train.get_checkpoint_state("checkpoints/")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model Restored.")
        else:
            print("Failed to restore model!")

    ###
    # TRAINING CODE
    ###
    def train(self, steps):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        for i in range(steps):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                print("Training step %d"%(i))
                network.save("checkpoints/model.ckpt", i)
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

    ###
    # TESTING CODE
    ###
    def test(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        test_results = self.out_image.eval(feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})
        combined_images = np.zeros((0,56)) # Empty array of 'correct' dimensions for concatenation
        for i in range (10):
            test_image = np.array(test_results[i]).reshape((28,28))
            test_image = self.post_process(test_image)
            actual_image = np.array(mnist.test.images[i]).reshape((28,28)) * 255
            actual_image = np.rot90(actual_image)
            # Stack output image with actual horizontally, for comparison
            image_column = np.hstack((test_image, actual_image))
            combined_images = np.vstack((combined_images, image_column))
        Preprocessor.displayImage(combined_images)

network = MNISTReconstruct()
#network.train(2001)
network.load()
network.test()




































