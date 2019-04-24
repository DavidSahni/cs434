import sys
import csv
import math
import random

def getRows(CSVPath):
    with open(CSVPath, newline='') as csvFile:
        rowReader = list(csv.reader(csvFile, delimiter=',', quotechar='|'))

        return rowReader

def predictPoint(k, featureList, exampleCancer):
    sortDistanceList = []
    benignVotes = 0
    malignantVotes = 0

    for cancerCell in featureList:
        distance = 0.
        for i in range(1,31):
            distance += (float(cancerCell[i]) - float(exampleCancer[i]))**2
        distance = math.sqrt(distance)
        sortDistanceList.append((distance, cancerCell[0]))
        sortDistanceList.sort()

    for vote in range(k):
        if(int(sortDistanceList[vote][1]) == 1):
            malignantVotes += 1
        elif(int(sortDistanceList[vote][1]) == -1):
            benignVotes +=1

    if(benignVotes > malignantVotes):
        return -1
    elif(malignantVotes > benignVotes):
        return 1

### Main
random.seed("implementation2")  # Seeding so we run the program multiple times, if needed.
trainFile = sys.argv[1]
testFile = sys.argv[2]
k = int(sys.argv[3])
crossValidationSize = 10        # Constant, defined as "K" in (some) notes
trainingErrorCounter = 0
validationErrorCounter = 0
testingErrorCounter = 0

trainingFeaturesLists = getRows(trainFile)
testingCancerCellList = getRows(testFile)
random.shuffle(trainingFeaturesLists)
# Basically 1/10 of cells saved in one list, 9/10s saved in another.
fullTrainingListSize = len(trainingFeaturesLists)
crossValidationListSize = int(fullTrainingListSize / crossValidationSize)
crossValidationCancerCells = trainingFeaturesLists[0:crossValidationListSize]
trainingCancerCellList = trainingFeaturesLists[crossValidationListSize:fullTrainingListSize]

for cancerSample in trainingCancerCellList:
    predication = predictPoint(k, trainingCancerCellList, cancerSample)
    actual = int(cancerSample[0])
    if(predication != actual):
        trainingErrorCounter += 1

for cancerSample in crossValidationCancerCells:
    predication = predictPoint(k, trainingCancerCellList, cancerSample)
    actual = int(cancerSample[0])
    if(predication != actual):
        validationErrorCounter += 1

for cancerSample in testingCancerCellList:
    predication = predictPoint(k, trainingCancerCellList, cancerSample)
    actual = int(cancerSample[0])
    if(predication != actual):
        testingErrorCounter += 1

# Use this when running q1.py on its own.
print("(Cross validation is " + str(int(100 / crossValidationSize)) + "% of training data.)")
print("Errors in training data: " + str(trainingErrorCounter) + " / " + str(len(trainingCancerCellList)))
print("\tError rate: " + str(round(100. * trainingErrorCounter / len(trainingCancerCellList), 2)) + "%")
print("Errors in cross validation: " + str(validationErrorCounter) + " / " + str(len(crossValidationCancerCells)))
print("\tError rate: " + str(round(100. * validationErrorCounter / len(crossValidationCancerCells), 2)) + "%")
print("Errors in testing data: " + str(testingErrorCounter) + " / " + str(len(testingCancerCellList)))
print("\tError rate: " + str(round(100. * testingErrorCounter / len(testingCancerCellList), 2)) + "%")

# Use this when running q1.py en masse using knn.py

# Is it worth using a k-d tree?
# Yeah, that'd be wise.
# I found a nice library: scipy.spatial.KDTree

# Some useful formulas
# Given m = [m1, ..., mi] and n = [n1, ..., ni]
# D(m, n) = ||m - n|| = sqrt((m - n)^T * (m - n))
# S(m, n) = e^(-alpha * D(x, y))
# Not sure when to use S(m, n)... Isn't alpha = k in KNN, essentially?

# k = "number of votes" cast by nearest points.
# K = Number of equal parts with one part being used for cross validation.
#   Why do both critically important numbers use K as their name with no other explanation?
#   Why do both seem to be used interchangably within the notes and assignment prompt?
#   For the lulz