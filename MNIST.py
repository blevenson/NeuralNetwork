# Wrapper class for loading in the MNIST dataset
# @author Brett Levenson

class MNIST:
    
    def __init__(self, foldername):
        self.foldername = foldername
        self.files = ["t10k-images.idx3-ubyte", "t10k-labels-idx1-ubyte", "train-images-idx3-ubyte", "train-labels-idx1-ubyte"]


    # load in data from MNIST file
    # returns: nothing
    def load(self):
        pass

    # Returns: array of matrices, each matrix is an image in the training data
    def get_test_data(self):
        return [[[1, 1],[1, 1]], [[2, 2], [2, 2]], [[3, 3][3, 3]]]

    # Returns: tuple
    #      - array of matrices, each matrix is an image
    #      - array of corresponding value
    def get_training_data(self):
        return ([[[1, 1],[1, 1]], [[2, 2], [2, 2]], [[3, 3][3, 3]]], [1, 2, 3])
