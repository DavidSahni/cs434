import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = torch.relu(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)

def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
      #  if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.data.item()))

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


def test():
    model.eval()
    val_loss, correct = 0, 0
    for data, target in testloader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
    return val_loss, accuracy

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
if __name__ == "__main__":
    device = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', device)


    batch_size = 200


    trainset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2)

    vset = datasets.CIFAR10(root='./data', train=False,
                                        transform=transforms.ToTensor())
    validation_loader = torch.utils.data.DataLoader(vset, batch_size=batch_size, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                                        transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers = 2)

    classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    accVecs = []
    lossVecs = []
    lrs = [.1, .01, .001, .0001]
    testAcc = (-1, -1)
    for lr in lrs:
        model = Net().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
        criterion = nn.CrossEntropyLoss()

        epochs = 10

        lossv, accv = [], []
        for epoch in range(1, epochs + 1):
            train(epoch)
            validate(lossv, accv)
        accVecs.append(accv)
        lossVecs.append(lossv)
        loss, acc = test()
        if acc > testAcc[0]:
            testAcc = (acc, lr)


    print("Test Accuracy: {}\nLearning Rate Used: {}".format(testAcc[0], testAcc[1]))

    plt.figure(figsize=(5,3))
    i = 0
    for vec in accVecs:
        lblStr = "lr={}".format(lrs[i])
        plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
        i+= 1
    plt.title('Accuracy over epochs')
    plt.legend()

    i=0
    plt.figure(figsize=(5,3))
    for vec in lossVecs:
        lblStr = "lr={}".format(lrs[i])
        plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
        i+=1
    plt.legend()

    # plt.plot(np.arange(1,epochs+1), accv)

    #plt.figure(figsize=(5,3))
    plt.title('Loss over epochs');

    plt.show()

