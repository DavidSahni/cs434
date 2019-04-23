import sys
import csv
import math

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
            distance += float(cancerCell[i]) - float(exampleCancer[i])
        distance = abs(distance)
        distance = math.sqrt(distance)
        sortDistanceList.append((distance, cancerCell[0]))
        sortDistanceList.sort()

    for vote in range(k):
        if(sortDistanceList[vote][1] == 1):
            malignantVotes += 1
        elif(sortDistanceList[vote][1] == -1):
            benignVotes +=1

    if(benignVotes > malignantVotes):
        return -1
    elif(malignantVotes > benignVotes):
        return 1

