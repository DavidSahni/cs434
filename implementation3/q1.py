import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


batch_size = 50

# train_dataset = datasets.MNIST('./data', 
#                                train=True, 
#                                download=True, 
#                                transform=transforms.ToTensor())

# validation_dataset = datasets.MNIST('./data', 
#                                     train=False, 
#                                     transform=transforms.ToTensor())

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
#                                                 batch_size=batch_size, 
#                                                 shuffle=False)

trainset = datasets.CIFAR10(root='./cifar-10-batches-py', train=True,
                                       transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

vset = datasets.CIFAR10(root='./cifar-10-valid-py', train=True,
                                       transform=transforms.ToTensor())
vloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

testset = datasets.CIFAR10(root='./cifar-10-batches-py', train=False,
                                       transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')