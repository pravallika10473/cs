import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("data.csv")

# Prepare features and target
X = data.drop(columns=["species"])
y = data["species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights randomly
weights = np.random.randn(X.shape[1])

# Learning rate
learning_rate = 0.01

# Forward pass
for epoch in range(0,5):
    for row in range(0,X_train.shape[0]):
        y_pred= np.dot(X_train.iloc[row].values,weights)
        #activation
        y_pred=math.tanh(y_pred)
        y= y_train.iloc[row]
        #loss function
        loss= (y_pred-y)**2
        # Back Propogation to minimize the loss 
        # calculate gradient loss 
        for i in range(0,weights.shape[0]):
            weights[i] -= learning_rate * 2 * (y_pred - y) * (1 - y_pred ** 2) * X_train.iloc[row].values[i]
# Prediction:
for row in range(0,X_test.shape[0]):
    y_pred = np.dot(X_test.iloc[row].values, weights)
    # Activation
    y_pred = math.tanh(y_pred)
    # Binarize the prediction
    y_pred = 1 if y_pred > 0 else 0
    y_true = y_test.iloc[row]
    print(y_pred, y_true)
    
    

