import h5py
import scipy.misc as scim
import numpy as np
import tensorflow as tf
import preprocessing
from preprocessing import NetworkInput

###
# NEURAL NETWORK WRAPPER FUNCTIONS
###

def conv2d(x, W, stride=1):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

# Custom function to return convolutional identity filter of given shape
def get_identity_filter(shape):
    if(shape[0] % 2 == 0 or shape[1] % 2 == 0):
        print("ERROR: Filter shape was not odd!")
        exit()
    else:
        xmid = (shape[0] - 1) / 2
        ymid = (shape[1] - 1) / 2
        filter = np.zeros(shape)
        for input_channel in range(shape[2]):
            for output_channel in range(shape[3]):
                if(input_channel == output_channel):
                    filter[xmid,ymid,input_channel,output_channel] = 1.0

        initial = tf.constant(filter, dtype=tf.float32)
        return tf.Variable(initial)

###
# Basic network that splits image into its component channels
###

input_size = (None, 304, 228, 3)
output_size = (None, 74, 55)
x = tf.placeholder(tf.float32, shape=input_size)
y_ = tf.placeholder(tf.float32, shape=output_size)
x_image = tf.reshape(x, [-1,304,228,3])


#filter = get_identity_filter((11,11,3,96))
#print(filter[:,:,0,0])

W1 = get_identity_filter((11,11,3,3))
h1 = conv2d(x_image, W1, stride=1)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

batch = preprocessing.getBatch(1, 0, flatten=False, greyscale=False)
output_array = h1.eval(feed_dict={x: batch[0], y_: batch[1]})
print(output_array.shape)

print(output_array[0,:,:,0])
#preprocessing.displayImage(np.array(output_array[0,:,:,0]))
#preprocessing.displayImage(np.array(output_array[0,:,:,1]))
#preprocessing.displayImage(np.array(output_array[0,:,:,2]))
color_image = np.zeros((304,228,3))
print(color_image.shape)
color_image[:,:,0] = output_array[0,:,:,0]
color_image[:,:,1] = output_array[0,:,:,1]
color_image[:,:,2] = output_array[0,:,:,2]
preprocessing.displayImage(color_image)
preprocessing.displayImage(batch[0][0])



