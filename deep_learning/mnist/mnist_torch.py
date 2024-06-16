import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn

def load_mnist_dataset():
    transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    )
    train_dataset= datasets.MNIST(root="torch_data/train", train=True, transform=transform, download=True)
    test_dataset=datasets.MNIST(root="torch_data/test", train=False, transform=transform, download=True)
    
if __name__=="__main__":
    torch.manual_seed(42)
    dataset= load_mnist_dataset()