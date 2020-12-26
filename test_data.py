# Runner for testing neural network with simple data
# @author Brett Levenson

import neural_network

def main():
    print("Running test data") 
    nn = neural_network.NeuralNetwork([2, 3, 1])
    
    # XOR data
    X = [[1, 1], [1, 0], [0, 1], [0, 0]] 
    y = [0, 1, 1, 0]

    nn.train(X, y)
    
    for i in range(len(X)):
        print(X[i], y[i], nn.predict(X[i]))

if __name__ == "__main__":
    main()
