import numpy as np  # Special array stuff.
import sys          # Uses the command line to pass in values.
import utils as u   # Customs utils script
import csv

def getRows(CSVPath):
    with open(CSVPath, newline='') as csvFile:
        rowReader = list(csv.reader(csvFile, delimiter=',', quotechar='|'))

        return rowReader

def calculateBatchPrediction(weights, xFeatures):
    prediction = 0. # In the notes this will be y_hat

    # To do, complete this function
    # wtx = weights * xFeatures
    # wtxSum = wtx.sum()

    prediction = .5

    return prediction

if(len(sys.argv) < 4):
    sys.exit("python q2_1.py usps-4-9-train.csv usps-4-9-test.csv learningrate")

csvList = getRows(sys.argv[1])
train = np.array(csvList, dtype=np.float)
(x,y) = u.readFromFile(train)

xFeatures = x
yClasses = y
# Note that X is a list of features
# Note that Y is a list of classes respective to X
# I think the "epsilon" respresents learning rate

numberOfDataPoints = len(xFeatures[0])
weights = np.zeros(numberOfDataPoints)
weightsDelta = np.zeros(numberOfDataPoints)
for i in range(1, numberOfDataPoints + 1):
    prediction = calculateBatchPrediction(weightsDelta, xFeatures)
    weightsDelta = weightsDelta + ((prediction - yClasses[i]) * xFeatures[i])
weights = weights - weightsDelta

print(str(weights))