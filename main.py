# Runner for neural network
# @author Brett Levenson

import MNIST
import neural_network
import numpy as np

def main():
    print("Neural Network")
    
    nn = neural_network.NeuralNetwork([3, 5, 2])
    
    data = MNIST.MNIST("Dataset/")

    X = np.array([0.5, 1, 0.25]).T
    out = nn.predict(X)
    print(out)

if __name__ == "__main__":
    main()
