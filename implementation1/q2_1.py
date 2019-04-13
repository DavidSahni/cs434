import numpy as np  # Special array stuff.
import sys          # Uses the command line to pass in values.
import utils as u   # Customs utils script
import csv
import matplotlib.pyplot as plt

def getRows(CSVPath):
    with open(CSVPath, newline='') as csvFile:
        rowReader = list(csv.reader(csvFile, delimiter=',', quotechar='|'))

        return rowReader

def caclulatePrediction(wDelta, xFeatures):
    # To do, complete this function
    #wtx = np.matmul(-wPrime, xFeatures)
    yhat = 1.0/(1.0+np.exp((-1.0*np.dot(np.transpose(wDelta),xFeatures))))
    if (yhat != 1):
        pass
        #print(yhat)
    return yhat

def plotAccuracies(r, trainAcc, testAcc):
    plt.figure(1)
    plt.subplot(111)
    plt.plot(r, trainAcc, label="Training")
    plt.ylabel("Accuracy")
    plt.xlabel("Batch Iterations")

    plt.plot(r, testAcc, label="Testing")
    plt.ylabel("Accuracy")
    plt.xlabel("Batch Iterations")
    plt.legend()
    plt.show()

def calcBatchWeights(xFeatures, yClasses, lRate, learnTo, xT, yT):
    numberOfFeats = np.size(xFeatures, axis=1)
    numberOfDataPoints = np.size(xFeatures, axis=0)
    weights = np.zeros(numberOfFeats, dtype=float)
    trainAcc = []
    testAcc = []

    for j in range(learnTo):
        trainAcc.append(u.calcRegressionAcc(weights, xFeatures, yClasses))
        testAcc.append(u.calcRegressionAcc(weights,xT,yT))
        weightsDelta = np.zeros(numberOfFeats, dtype=np.float)
        for i in range(numberOfDataPoints):
            prediction = caclulatePrediction(weights, xFeatures[i])
            preDelta = prediction - yClasses[i]
        # print(preDelta)
            newDelta = (preDelta * xFeatures[i])
            weightsDelta = weightsDelta + newDelta
        weights = weights - (lRate* weightsDelta)

    return weights, trainAcc, testAcc

### Main
if(len(sys.argv) < 4):
    sys.exit("python q2_1.py usps-4-9-train.csv usps-4-9-test.csv learningrate")

(x,y) = u.readFromFile(sys.argv[1], ",")
xT, yT = u.readFromFile(sys.argv[2],",")
xTest = xT[:,:] / 255.0
xFeatures = x[:,:] / 255.0
yClasses = y
lRate = float(sys.argv[3])
learnTo = 150
# Note that X is a list of features
# Note that Y is a list of classes respective to X
# I think the "epsilon" respresents learning rate
# Turns out the n (eta) is actually the learning rate

weights, trainAcc, testAcc  = calcBatchWeights(xFeatures, yClasses, lRate, learnTo, xTest, yT)
plotAccuracies(range(learnTo), trainAcc, testAcc)