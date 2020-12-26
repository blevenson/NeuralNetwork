# Neural network class
# @author Brett Levenson

import numpy as np

class NeuralNetwork:
    
    # layers: array of nodes in each layer
    #         [10, 5, 10]
    # [output x h_n] x ... x [h_n x h_1] x [h1 x input] x [input x 1]
    def __init__(self, layers):
        self.W = [] # [next_layer x prev_layer x num_layers - 1]
        
        for i in range(len(layers) - 1):
            self.W.append(np.random.rand(layers[i + 1], layers[i])) 

        self.num_layers = len(layers)
        assert(self.num_layers - 1 == len(self.W))

    def train(self, X, y):
        for x, y_expected in zip(X, y):
            print("training: ", x, y_expected)
            

    def predict(self, X):
        prev_layer = X
        
        for i in range(len(self.W)):
            output = np.matmul(self.W[i], prev_layer)
            output = self.activation(output)
            prev_layer = output

        return output    

    # sigmoid activation function
    def activation(self, val):
        return 1 / (1 + np.exp(-val))

    def derivative_activation(self, val):
        return self.activation(val) * (1 - self.activation(val))
