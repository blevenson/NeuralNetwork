# Runner for neural network
# @author Brett Levenson

import MNIST
import neural_network

def main():
    print("Neural Network")
    
    nn = neural_network.NeuralNetwork([100, 20, 10])
    
    data = MNIST.MNIST("Dataset/")

if __name__ == "__main__":
    main()
