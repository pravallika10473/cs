import numpy as np
from keras.datasets import mnist
import argparse
import random

class Network(object):
    def __init__(self, size):
        self.size = size
        self.number_of_layers = len(size)
        self.biases = [np.random.randn(y, 1) for y in size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
        
    def feed_forward(self, input):
        a = input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def cost_derivative(self, y, y_true):
        return y - y_true
    
    def back_propagation(self, X, y_true):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feed forward
        activation = X
        activations = [X]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = (np.dot(w, activation) + b)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # Backward pass
        # Output layer
        delta = self.cost_derivative(activations[-1], y_true) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.number_of_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return nabla_b, nabla_w
    
    def update_mini_batch(self, batch_X, batch_y, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for j in range(len(batch_X)):
            X = batch_X[j].reshape(-1, 1)  # Converting to column vectors
            y = batch_y[j]
            y_true = one_hot_encoding(y, 10)
            delta_nabla_b, delta_nabla_w = self.back_propagation(X, y_true)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        nabla_b = [nb / len(batch_X) for nb in nabla_b]  # Average the gradients for biases
        nabla_w = [nw / len(batch_X) for nw in nabla_w]  # Average the gradients for weights
        
        self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_b)]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def one_hot_encoding(label, num_classes=10):
    one_hot = np.zeros((num_classes, 1))
    one_hot[label] = 1
    return one_hot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set the random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
    X_train = X_train.reshape(60000, 784) 
    X_test = X_test.reshape(10000, 784)   

    X_train = X_train.astype('float32')   
    X_test = X_test.astype('float32')

    # Normalize the data to the range [0, 1]
    X_train /= 255                        
    X_test /= 255
    
    mnist_network = Network([784, 30, 10]) # Defining the network size
    
    for epoch in range(args.epochs):
        # Shuffle the training data
        combined = list(zip(X_train, y_train))
        random.shuffle(combined)
        X_train[:], y_train[:] = zip(*combined)
        
        # Train
        for i in range(0, X_train.shape[0], args.batch_size):
            batch_X = X_train[i:i + args.batch_size]
            batch_y = y_train[i:i + args.batch_size]
            
            # Update network parameters using the mini-batch
            mnist_network.update_mini_batch(batch_X, batch_y, args.learning_rate)
        
        # Test
        correct = 0
        total = len(X_test)
        for X, y in zip(X_test, y_test):
            X = X.reshape(-1, 1)  # Convert to column vectors
            y_pred = mnist_network.feed_forward(X)
            predicted_label = np.argmax(y_pred)
            if predicted_label == y:
                correct += 1

        accuracy = correct / total
        print(f"Accuracy after epoch {epoch + 1}: {accuracy * 100:.2f}%")

            
            
    

                    
