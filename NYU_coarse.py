import h5py
import scipy.misc as scim
import numpy as np
import tensorflow as tf
import preprocessing
from preprocessing import Preprocessor
import glob
import re

###
# Required import during graph building!!
from preprocessing import NetworkInput
###

class NYUCoarse:

    def __init__(self, learning_rate=1e-4, dataset_name="offices"):
        self.x = tf.placeholder(tf.float32, shape=[None, 304, 228, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 74, 55])
        self.learning_rate = learning_rate
        self.weights = dict()

        self._create_network()
        print("Creating session and intitalizing weights... ")
        self.sess = tf.InteractiveSession()
        self._initialize_weights()
        self.saver = tf.train.Saver(self.weights)
        print("Done initializing session.")

        # Overwrite this when we load model with global step value
        # Increased to 0 when training, so first step is first image in batch
        self.step = -1

        # Load dataset into memory prior to training / testing
        print("Loading dataset and batching unit... ")
        self.pp = Preprocessor(dataset_name, greyscale=False)
        print("Done loading dataset.")

    def post_process(self, image):
        # Normalize output image for display: Set minimum value of image to 0;
        # then scale maximum to 255
        image -= image.min()
        if image.max() == 0.0:
            return image
        image = (image / image.max()) * 255
        return image

    def get_database_size(self):
        return self.pp.getDatasetSize()

    ###
    # NEURAL NETWORK WRAPPER FUNCTIONS
    ###

    def _weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        var = tf.Variable(initial, name=name)
        self.weights[name] = var
        return var

    def _bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        var = tf.Variable(initial, name=name)
        self.weights[name] = var
        return var

    def _conv2d(self, x, W, stride=1):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='VALID')

    ###
    # NETWORK ARCHITECTURE
    ###

    def _create_network(self):
        print("Creating network model...")

        # Input specifications
        x_image = tf.reshape(self.x, [-1,304,228,3])
        y_image = tf.reshape(self.y_, [-1, 74, 55])

        # First layer
        W_conv1 = self._weight_variable([11, 11, 3, 96], name="W_conv1")
        W_conv1 = tf.Print(W_conv1, [W_conv1], "First convolu layer weights:")
        b_conv1 = self._bias_variable([96], name="b_conv1")
        b_conv1 = tf.Print(b_conv1, [b_conv1], "First convolu layer biases:")
        h_conv1 = tf.nn.relu(self._conv2d(x_image, W_conv1, stride=4) + b_conv1)
        h_conv1 = tf.Print(h_conv1, [h_conv1], "First convolu layer:")
        h_pool1 = self._max_pool_2x2(h_conv1)
        h_pool1 = tf.Print(h_pool1, [h_pool1], "First pooling layer:")

        # Second layer
        W_conv2 = self._weight_variable([5, 5, 96, 256], name="W_conv2")
        b_conv2 = self._bias_variable([256], name="b_conv2")
        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        # Third layer
        W_conv3 = self._weight_variable([3, 3, 256, 384], name="W_conv3")
        b_conv3 = self._bias_variable([384], name="b_conv3")
        h_conv3 = tf.nn.relu(self._conv2d(h_pool2, W_conv3) + b_conv3)

        # Fourth layer
        W_conv4 = self._weight_variable([3, 3, 384, 384], name="W_conv4")
        b_conv4 = self._bias_variable([384], name="b_conv4")
        h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4) + b_conv4)

        # Fifth layer
        W_conv5 = self._weight_variable([3, 3, 384, 256], name="W_conv5")
        b_conv5 = self._bias_variable([256], name="b_conv5")
        h_conv5 = tf.nn.relu(self._conv2d(h_conv4, W_conv5) + b_conv5) #Should we use pooling here?

        # Sixth layer; fully connected
        W_fc1 = self._weight_variable([19 * 14 * 256, 4096], name="W_fc1")
        h_conv5_flat = tf.reshape(h_conv5, [-1, 19*14*256])
        b_fc1 = self._bias_variable([4096], name="b_fc1")
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

        # Dropout after layer 6
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Seventh layer; fully connected; output of Coarse network
        W_fc2 = self._weight_variable([1 * 1 * 4096, 74 * 55], name="W_fc2")
        b_fc2 = self._bias_variable([74 * 55], name="b_fc2")
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        self.out_image = tf.reshape(h_fc2, (-1, 74, 55))

        self.train_error = tf.reduce_mean(tf.log(tf.square(self.out_image - y_image)))
        self.train_error = tf.Print(self.train_error, [self.train_error], "Current training error: ")
        #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.train_error)
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.train_error) # Momentum of 0.9

        print("Done creating model.")

    def _initialize_weights(self):
        # Returns an intialization op that is executed first in the session
        self.sess.run(tf.initialize_all_variables())

    def save(self, path, step):
        # Save current values of session variables, return the save path
        save_path = self.saver.save(self.sess, path, global_step=step)
        print("Model saved as: %s"%(save_path))

    def load(self):
        # Returns CheckpointState proto from "checkpoint" file
        ckpt = tf.train.get_checkpoint_state("coarse_checkpoints/")
        print("loading: %s"%(ckpt.model_checkpoint_path))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            # Don't bother to change the initial step when we are using database chunks
            #self.step = int(re.match(r'^\D+(\d+)$', ckpt.model_checkpoint_path).group(1)) # Get Step number
            print("Model Restored.")
        else:
            print("Failed to restore model!")

    ###
    # TRAINING CODE
    ###
    def train(self, num_batches, batch_size=50, save_after_iterations=30):

        print("Training...")

        # If we loaded from a previously trained model, increment its step number by 1 and start from there.
        start_step = self.step + 1
        for i in range(start_step, start_step + num_batches):
            # Variable i: batch offset in chunk (0 ... num_batches - 1)
            # Variable i + 1: Current step (1 ... num_batches)
            current_step = i + 1
            batch = self.pp.getBatch(batch_size, i, flatten=False)

            # Print information about current training subset
            print("Training batch %d/%d; Images %d-%d/%d"%
                  (current_step, num_batches, i*batch_size + 1,
                   (current_step)*batch_size, num_batches*batch_size))

            # Train network, calculate and display errors
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
            #error_out = self.train_error.eval(feed_dict={self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
            #avg_diff = float(np.sqrt(error_out) / (74 * 55))
            #print("Training error during step %d: %f"%(current_step, error_out))
            #print("Adjusted error during step %d: %f"%(current_step, avg_diff))

            # Save network if we have gone over save_after_iterations batches, or if we hit the end
            if (current_step)%save_after_iterations == 0 or (current_step) == num_batches:
                print("Saving at step %d"%(current_step))
                self.save("coarse_checkpoints/model.ckpt", current_step)

    ###
    # TESTING CODE
    ###
    def test(self, batch_size=5, batch_index=0):
        batch = self.pp.getBatch(batch_size, batch_index, flatten=False)
        test_results = self.out_image.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
        combined_images = np.zeros((0,110)) # Empty array of 'correct' dimensions for concatenation
        for i in range (batch_size):
            test_image = np.array(test_results[i]).reshape((74,55))
            test_image = self.post_process(test_image)
            actual_image = batch[1][i]
            # Stack output image with actual horizontally, for comparison
            image_column = np.hstack((test_image, actual_image))
            combined_images = np.vstack((combined_images, image_column))
        #self.pp.displayImage(combined_images)
        combined_images = np.rot90(combined_images, 3)
        scim.imsave("test_output/test_image_" + str(batch_index).zfill(4) + ".bmp", combined_images)

# SCRIPT
# Gets all the chunks (in .pkl format) of a given prefix and trains the network
# iteratively on them.
def train_on_chunks(prefix):
    mat_files = glob.glob("datasets/" + prefix + "*.pkl")
    first  = True # Don't load old (non-existent) network when training on the first chunk!
    for filename in mat_files:

        # Get the actual name of the chunk
        regex = r"^.*/(" + prefix +".*)\.pkl$"
        m = re.match(regex, filename)
        chunk_name = m.group(1)

        # Initialize network, then train on a chunk
        network = NYUCoarse(learning_rate=1e-4, dataset_name=chunk_name)
        if(not first): # If we are not the first chunk, load the previous network
            print("Loading existing network.")
            network.load() # Load most recently saved
        else:
            first = False
        dataset_size = network.get_database_size() # Get number of images in chunk
        batch_size = 32
        num_batches = int(dataset_size / batch_size)
        print("Training on chunk: %s"%(chunk_name))
        network.train(num_batches, batch_size, save_after_iterations=50)
        network.sess.close() # Close the network session. Possibly move this elsewhere?

# SCRIPT
def train_on_randomized(dataset_name):
    network = NYUCoarse(learning_rate=1e-2, dataset_name=dataset_name)
    batch_size = 32
    num_batches = 7000 # Number of training iterations
    network.train(num_batches, batch_size, save_after_iterations=100)
    network.sess.close()

# SCRIPT
def test_network():
    network = NYUCoarse(dataset_name="nyu_depth_v2_labeled")
    network.load()
    dataset_size = network.get_database_size()
    batch_size = 5
    num_batches = 100 #int(dataset_size / batch_size)
    for i in range(num_batches):
        network.test(batch_size, i)

"""
network = NYUCoarse(learning_rate=1e-5, dataset_name="home_office_0001")
print network.get_database_size()
#network.load() # Preload network
network.train(4, batch_size=10, save_after_iterations=3)
#network.load() # Load most recently saved coarse network
#network.test()


batch = preprocessing.getBatch(1, 0, flatten=False, greyscale=False)
output_array = out_image.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print(output_array[0])
preprocessing.displayImage(np.array(output_array[0]).reshape((74,55)))
"""

#test_network()
train_on_randomized("randomized_home_offices")
#train_on_chunks("home")






