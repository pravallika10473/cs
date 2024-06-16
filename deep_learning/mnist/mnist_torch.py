import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader

def load_mnist_dataset(batch_size):
    transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    )
    train_dataset= datasets.MNIST(root="torch_data/train", train=True, transform=transform, download=True)
    test_dataset=datasets.MNIST(root="torch_data/test", train=False, transform=transform, download=True)
    train_loader= DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class model(nn.Module):
    def __init__(self):
        super.__init__()
        self.fc1=nn.linear(784,128) # 784 inputs to hidden layer of size 128
        self.fc2=nn.linear(128,10) # 128 inputs with size of 10 final output layer
        
    def feed_forward(self,x):
        x=nn.flatten(x,1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
if __name__=="__main__":
    torch.manual_seed(42)
    train_loader, test_loader= load_mnist_dataset(batch_size=64)
    # for batch_idx, (data,label) in enumerate(train_loader):
    #     print("Batch id:", batch_idx+1)
    #     print("data.shape:", data.shape)
    #     print("label.shape:", label.shape)
    #     break
    
