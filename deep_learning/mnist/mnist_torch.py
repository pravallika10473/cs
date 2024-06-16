import numpy as np
from keras.datasets import mnist
import argparse
import random
import torch
import torch.nn as nn


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
    
    hidden_layer=nn.linear(784,10)
    activation=hidden_layer(X_train)
    output=torch.sigmoid(activation)
    print(output.size())
    
