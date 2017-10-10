from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc as scim
import numpy as np
from preprocessing import Preprocessor
from preprocessing import NetworkInput
import h5py
import scipy.io
import glob
import re
import random

##
# File for executing custom scripts to debug / diagnose the network.
##

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch = mnist.train.next_batch(50)

arr = np.array(batch[0][49])
arr = np.reshape(arr, (28,28))
print(arr)

#print(batch[0].shape)
#scim.toimage(arr).show()

print mnist.test.images[0].shape

batch = preprocessing.getBatch(1,0, greyscale=False, flatten=False)
#preprocessing.displayImage(batch[0][0])
print(batch[0][0])

batch = preprocessing.getBatch(1,0, flatten=False, greyscale=False)

def post_process(image):
    # Normalize output image for display: Set minimum value of image to 0;
    # then scale maximum to 255
    image -= image.min()
    if image.max() == 0.0:
        return image
    image = (image / image.max()) * 255
    return image

preprocessing.displayImage(post_process(batch[0][0]))
#print(batch[1][0])


m = re.match(r"^\D+(\d+)$", "model.ckpt-120")
print(int(m.group(1)))

datasetName = "datasets/" + "offices"
mat_database = datasetName + ".mat"
#mat_contents = h5py.File(mat_database, 'r')
mat = scipy.io.loadmat(mat_database)
image = mat["collection"][0,0]['image']
depths = mat["collection"][0,0]['depths']
print(len(mat["collection"][0]))
print(image.shape)
Preprocessor.displayImage(np.rot90(image))
Preprocessor.displayImage(depths)
#print(mat["ans"])

image = np.zeros((32,32))
scim.imsave("test_output/debug" + str(15).zfill(4) + ".bmp", image)

# Get number of images in 'offices'
prefix = 'home_office'
mat_files = glob.glob("datasets/" + prefix + "*.pkl")
first  = True # Don't load old (non-existent) network when training on the first chunk!
pp = Preprocessor() # Dummy pp
sum_sizes = 0
for filename in mat_files:
    # Get the actual name of the chunk
    regex = r"^.*/(" + prefix +".*)\.pkl$"
    m = re.match(regex, filename)
    chunk_name = m.group(1)

    if(first):
        pp = Preprocessor(chunk_name)
        first = False
    else:
        pp.loadDataset(chunk_name)
    sum_sizes += pp.getDatasetSize()
    print("Size of dataset %s is %d"%(chunk_name, pp.getDatasetSize()))
print("Number of images in dataset %s is %d:"%(prefix, sum_sizes))

pp = Preprocessor()
data = pp.data
print(len(data))

bigger_set = []
bigger_set += data
bigger_set += data

print(len(bigger_set))

pp.displayImage(data[0].depthmap)

def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

data = random_permutation(data)
pp.displayImage(data[0].depthmap)

image = np.array([[3,4],[5,6],[7,8]])
mean_array = np.mean(image)
print(mean_array)
