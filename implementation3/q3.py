import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, dropOut):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 100)
        self.fc1_drop = nn.Dropout(dropOut)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)

def train(epoch, stopVal, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    i = 0
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
        i+=1
        if i > stopVal:
            break
        
      #  if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.data.item()))

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
   # for i in range(4000, 5000):
    #        data, target = train_loader.dataset[i]
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
    
    # print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # val_loss, correct, len(validation_loader.dataset), accuracy))


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
    
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # val_loss, correct, len(validation_loader.dataset), accuracy))
    return val_loss, accuracy

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# if __name__ == "__main__":
def createNeuralNetwork(dropOut=0.2, momentum=0.5, weightDecay=0.):
    accVecs = []
    lossVecs = []
    lrs = [.1, .01, .001, .0001]
    lr = .1
    testAcc = (-1, -1)
    stopVal = 40000/batch_size #only use 4 of the 5 batches for training
    # for lr in lrs:
    global model
    global optimizer
    global criterion
    model = Net(dropOut).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=momentum, weight_decay=weightDecay)
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch, stopVal)
        validate(lossv, accv)
    # accVecs.append(accv)
    # lossVecs.append(lossv)
    loss, acc = test()
    if acc > testAcc[0]:
        testAcc = (acc, lr)
    
    # print("Test Accuracy: {}\nLearning Rate Used: {}".format(testAcc[0], testAcc[1]))
    # print("{}%".format(testAcc[0]))
    # print(str(round(testAcc[0], 2) + "%,"))

    return accv, lossv, acc

### Main
device = torch.device('cpu')
# print('Using PyTorch version:', torch.__version__, ' Device:', device)
batch_size = 200
trainset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2)
vset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
validation_loader = torch.utils.data.DataLoader(vset, batch_size=batch_size, num_workers=2)
testset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

accVecs = []
lossVecs = []
testAccList = []
epochs = 10
dropOuts = [0., 0.2, 0.4, 0.6, 0.8]
momentums = [0, 0.5, 1., 1.5, 2.]
weightDecays = [0, 0.5, 1., 1.5, 2.]


##### Drop Outs


for dropOut in dropOuts:
    accv, lossv, testAcc = createNeuralNetwork(dropOut=dropOut)
    accVecs.append(accv)
    lossVecs.append(lossv)
    testAccList.append(testAcc)

plt.figure(figsize=(5,3))
i = 0
for vec in accVecs:
    lblStr = "Drop Out = {}".format(dropOuts[i])
    plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
    i+= 1
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title('Validation Accuracy over Epochs')
plt.legend()


plt.figure(figsize=(5,3))
i = 0
for vec in lossVecs:
    lblStr = "Drop Out = {}".format(dropOuts[i])
    plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
    i+=1
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title('Training Loss over Epochs');
plt.legend()


plt.figure(figsize=(5,3))
plt.plot(dropOuts, testAccList, marker='o')
plt.xlabel("Drop Outs")
plt.ylabel("Total Accuracy")
plt.title('Total Accuracy over Drop Outs');


print("Drop Outs Complete")


##### Momentums

accVecs = []
lossVecs = []
testAccList = []

for momentum in momentums:
    accv, lossv, testAcc = createNeuralNetwork(momentum=momentum)
    accVecs.append(accv)
    lossVecs.append(lossv)
    testAccList.append(testAcc)

plt.figure(figsize=(5,3))
i = 0
for vec in accVecs:
    lblStr = "Momentum = {}".format(momentums[i])
    plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
    i+= 1
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title('Validation Accuracy over Epochs')
plt.legend()


plt.figure(figsize=(5,3))
i = 0
for vec in lossVecs:
    lblStr = "Momentum = {}".format(momentums[i])
    plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
    i+=1
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title('Training Loss over Epochs');
plt.legend()


plt.figure(figsize=(5,3))
plt.plot(momentums, testAccList, marker='o')
plt.xlabel("Momentums")
plt.ylabel("Total Accuracy")
plt.title('Total Accuracy over Momentums');


print("Momentums Complete")


##### Weight Decay


accVecs = []
lossVecs = []
testAccList = []


for weightDecay in weightDecays:
    accv, lossv, testAcc = createNeuralNetwork(weightDecay=weightDecay)
    accVecs.append(accv)
    lossVecs.append(lossv)
    testAccList.append(testAcc)

plt.figure(figsize=(5,3))
i = 0
for vec in accVecs:
    lblStr = "Weight Decay = {}".format(weightDecays[i])
    plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
    i+= 1
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title('Validation Accuracy over Epochs')
plt.legend()


plt.figure(figsize=(5,3))
i = 0
for vec in lossVecs:
    lblStr = "Drop Out = {}".format(weightDecays[i])
    plt.plot(np.arange(1,epochs+1), vec, label=lblStr, marker='o')
    i+=1
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title('Training Loss over Epochs');
plt.legend()


plt.figure(figsize=(5,3))
plt.plot(weightDecays, testAccList, marker='o')
plt.xlabel("Weight Decays")
plt.ylabel("Total Accuracy")
plt.title('Total Accuracy over Weight Decays');


print("Weight Decays Complete")


##### Show Plots


plt.show()
