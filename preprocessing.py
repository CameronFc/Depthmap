import h5py
import scipy.misc as scim
import numpy as np
import pickle
import scipy.io
import re
import glob
import random
from scipy import ndimage

# Data object for pickling.
class NetworkInput:

    def __init__(self, image, depthmap):
        self.image = image
        self.depthmap = depthmap

# Preprocessor class for converting raw or labelled dataset into .pkl files
# that can be fed into the network using the getBatch() method.
# Also contains some useful static methods, such as displayImage()
class Preprocessor:

    def __init__(self, dataset_name="nyu_depth_v2_labeled" ,greyscale=False):
        # Only load the dataset once into memory
        self.loadDataset(dataset_name, greyscale)

    def getDatasetSize(self):
        return len(self.data)

    # Utility function that grabs all .mat or .pkl files of given prefix
    @staticmethod
    def getChunkNamesByPrefix(prefix, ext):
        mat_files = glob.glob("datasets/" + prefix + "*" + ext)
        i = 0
        chunk_names = []
        for filename in mat_files:
            regex = r"^.*/(" + prefix +".*)" + ext +"$"
            m = re.match(regex, filename)
            chunk_name = m.group(1)
            chunk_names.append(chunk_name)
        return chunk_names

    # Downsample the input to the correct (304x228) resolution
    @staticmethod
    def downsampleNYU(image, size):
        return scim.imresize(image, size)

    # Diplays image from NYU dataset, rotated to correct viewing orientation
    @staticmethod
    def displayImage(image):
        scim.toimage(np.transpose(scim.toimage(image), 4)).show()

    # Convert image to greyscale
    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # Preprocesses the entire NYU labelled database and pickles it.
    @staticmethod
    def preprocessDataset(greyscale=True):
        datasetName = "datasets/" + "nyu_depth_v2_labeled"
        mat_database = datasetName + ".mat"
        mat_contents = h5py.File(mat_database, 'r')
        datasetSize = len(mat_contents["images"])
        if(greyscale):
            datasetName += "_greyscale"
        with open(datasetName + ".pkl", 'wb') as output:
            data = []
            for i in range(datasetSize):
                # Downsample and convert the image to greyscale
                unprocessed_image = mat_contents["images"][i]
                image = Preprocessor.downsampleNYU(unprocessed_image, (304,228))
                if(greyscale):
                    image = Preprocessor.rgb2gray(image)
                # Downsample depthmap label to network output size
                label = Preprocessor.downsampleNYU(mat_contents["depths"][i], (74,55))
                dataPoint = NetworkInput(image, label)
                data.append(dataPoint)
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    # Preprocess any of the given datasets from our matlab configuration output
    @staticmethod
    def preprocessRaw(dataset_name):
        path_no_ext = "datasets/" + dataset_name
        path_inc_ext = path_no_ext + ".mat"
        mat_contents = scipy.io.loadmat(path_inc_ext)
        datasetSize = len(mat_contents["collection"][0]) # Number of structs in cell array
        with open(path_no_ext + ".pkl", 'wb') as output:
            data = []
            for i in range(datasetSize):
                unprocessed_image = mat_contents["collection"][0,i]["image"]
                image = np.rot90(unprocessed_image) # 90 degrees counter clockwise
                image = Preprocessor.downsampleNYU(image, (304,228))
                # Downsample depthmap label to network output size
                unprocessed_label = mat_contents["collection"][0,i]["depths"]
                label = np.rot90(unprocessed_label)
                label = Preprocessor.downsampleNYU(label, (74,55))
                dataPoint = NetworkInput(image, label)
                data.append(dataPoint)
            pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

    # Loads dataset of given name into Preprocessor memory.
    def loadDataset(self, dataset_name, greyscale=False):
        path_no_ext = "datasets/" + dataset_name
        if(greyscale):
            path_no_ext += "_greyscale"
        with open(path_no_ext + ".pkl", 'rb') as input:
            data = pickle.load(input)
            self.data = data
            #return data

    # Returns a batch of given size from dataset. Increment index to get next batch.
    # Instead of checking for dataset boundaries, this will choose a random batch if index given is outside dataset
    def getBatch(self, size, index, flatten=True):
        batch = [0,0]
        if(flatten):
            batch[0] = np.zeros(shape=(size, (304 * 228))) # Input
            batch[1] = np.zeros(shape=(size, (74 * 55))) # Label
        else:
            batch[0] = np.zeros(shape=(size, 304, 228, 3)) # Input
            batch[1] = np.zeros(shape=(size, 74, 55)) # Label

        start = index * size
        end = (index + 1) * size
        maxInd = len(self.data) - 1
        if(end > maxInd):
            start = np.random.randint(0, maxInd - size)
            end = start + size
        i = 0
        while(start + i < end):
            if(flatten):
                batch[0][i] = np.ndarray.flatten(self.data[start + i].image)
                batch[1][i] = np.ndarray.flatten(self.data[start + i].depthmap)
            else:
                batch[0][i] = self.data[start + i].image
                batch[1][i] = self.data[start + i].depthmap
            i += 1
        return batch


# SCRIPT
# Programmatically calls preprocessRaw() so we don't have to specify an array of chunks,
# Only need to specify a common prefix
# Get the name of all the home_offices chunks in database directory, then preprocess them.
def processChunksByPrefix(prefix):
    chunk_names = Preprocessor.getChunkNamesByPrefix(prefix, ".mat")
    i = 0
    for chunk_name in chunk_names:
        print("Preprocessing chunk %s into .pkl"%(chunk_name))
        Preprocessor.preprocessRaw(chunk_name)
        i = i + 1
    print("Finished processing %d .mat files into .pkl format."%(i))

# Export me to a utils file
def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

# SCRIPT
# Takes all of the chunks of a given prefix, combines them into a single dataset / file,
# randomizes that file, then saves it back to .pkl. This allows us to pickup a subset of the data
# and choose random samples across the whole scene easily.
def combineRandomizeSaveDataset(prefix, targetName):

    pp = Preprocessor()
    big_dataset = []

    # Keep appending data until we have several GB in memory
    # This might cause the program to crash as it does not check if we run out of memory!
    print("Appending chunks...")
    chunk_names = Preprocessor.getChunkNamesByPrefix(prefix, ".pkl")
    for chunk_name in chunk_names:
        pp.loadDataset(chunk_name)
        big_dataset += pp.data

    print("Length of dataset: %d"%(len(big_dataset)))

    # Permute the big dataset, then save it using the target name.
    print("Permuting dataset... ")
    big_dataset = random_permutation(big_dataset)
    path_no_ext = "datasets/" + targetName
    print("Saving dataset... ")
    with open(path_no_ext + ".pkl", 'wb') as output:
        pickle.dump(big_dataset, output, pickle.HIGHEST_PROTOCOL)


#preprocessDataset(greyscale=False)
#data = loadDataset()

#subset = data[0:50]
#batch = getBatch(50, 0)
#print(batch[0][10].shape)
#im = np.ndarray.reshape(batch[1][10], (74, 55))
#displayImage(im)
#print(batch.shape)
#displayImage(data[1000].image)

"""
image = scim.toimage(mat_contents["depths"][5]);
image = scim.toimage(downsampleNYU(image))
displayImage(image)
"""
"""
datasetName = "datasets/" + "nyu_depth_v2_labeled"
mat_database = datasetName + ".mat"
mat_contents = h5py.File(mat_database, 'r')
Preprocessor.displayImage(mat_contents["images"][0])
"""

"""
pp = Preprocessor("offices")
pp.preprocessRaw("home_office_0001")
data = pp.loadDataset(dataset_name="home_office_0001")
pp.displayImage(data[0].image)
"""

#combineRandomizeSaveDataset("home", "randomized_home_offices")
#processChunksByPrefix("home")

