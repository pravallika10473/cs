import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import argparse

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 784 inputs to hidden layer of size 128
        self.fc2 = nn.Linear(128, 10)   # 128 inputs with size of 10 final output layer
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
    return prediction

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict the digit in an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Load the trained model weights
    model = Model()
    model.load_state_dict(torch.load('mnist_model.pth'))
    
    # Load and preprocess image
    image_tensor = preprocess_image(args.image_path)
    
    # Display the image being predicted
    plt.imshow(image_tensor.squeeze(), cmap='gray')
    plt.show()
    
    # Predict the digit
    prediction = predict_image(model, image_tensor)
    print(f'The predicted digit is: {prediction}')
