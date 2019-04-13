
import numpy as np  # Special array stuff.
import sys          # Uses the command line to pass in values.
import utils as u   # Customs utils script
import csv
import matplotlib.pyplot as plt

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
    plt.xlabel("Lambda Value")

    plt.plot(r, testAcc, label="Testing")
    plt.ylabel("Accuracy")
    plt.xlabel("Lambda Value")
    plt.legend()
    plt.ylim(0, 1)
    plt.xscale("log")
    plt.show()


def calcRegBatch(xFeatures, yClasses, lRate, learnTo, xT, yT, lambdas):
    numberOfFeats = np.size(xFeatures, axis=1)
    numberOfDataPoints = np.size(xFeatures, axis=0)
    weights = np.zeros(numberOfFeats, dtype=float)

    trainAcc = []
    testAcc = []
    for k in lambdas:
        trainAcc.append(u.calcRegressionAcc(weights, xFeatures, yClasses))
        testAcc.append(u.calcRegressionAcc(weights,xT,yT))
        for j in range(learnTo):

            weightsDelta = np.zeros(numberOfFeats, dtype=np.float)
            for i in range(numberOfDataPoints):
                
                prediction = caclulatePrediction(weights, xFeatures[i])
                predDelta = prediction - yClasses[i]
            # printd(preDelta)
                newDelta = (predDelta * xFeatures[i]) + k * weights

                weightsDelta = weightsDelta + newDelta
            weights = weights - (lRate* weightsDelta)

    return weights, trainAcc, testAcc


if(len(sys.argv) != 4):
    sys.exit("python q2_1.py usps-4-9-train.csv usps-4-9-test.csv lambaFile")




(x,y) = u.readFromFile(sys.argv[1], ",")

xT, yT = u.readFromFile(sys.argv[2],",")

xTest= xT[:,:]/255.0

xFeatures = x[:,:] / 255.0
yClasses = y
lRate = .000001
learnTo = 80
lambdas = np.genfromtxt(sys.argv[3])

weightsL, trainAccL, testAccL  = calcRegBatch(xFeatures, yClasses, lRate, learnTo, xTest, yT, lambdas)


plotAccuracies(lambdas, trainAccL, testAccL)