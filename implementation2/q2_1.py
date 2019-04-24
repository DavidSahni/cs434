import numpy as np  # Special array stuff.
import sys          # Uses the command line to pass in values.
import csv
import matplotlib.pyplot as plt

POSRIGHT = 1
POSLEFT = -1


class node:
    threshold = None
    featureIdx = None
    iGain = 0
    direction = 0
    childA = None
    childB = None

    def __init__(self, idx=None, thold=None):
        self.threshold = thold
        self.featureIdx = idx

    def prettyPrint(self, cl=True):
        print("Test on X:", self.featureIdx)
        print("Threshold:", self.threshold)
        print("Information Gain:", round(self.iGain,5))
        if cl:
            dirStr = "negative"
            if self.direction == POSRIGHT:
                dirStr = "postive"
            print("Values above the threshold are placed into the", dirStr, "class")


    def evalArray(self, data):
        for row in data:
            val = data[self.featureIdx]
            eClass = 1 #assume its in the positive class
            if val < self.threshold: #switch to neg class if below threshold
                eClass = -1     #this assumes that pos class > threshold
            eClass = eClass * self.direction #if direction is pos class < threshold, direction = -1
        return eClass

def getClassCounts(data):
    pos = 0
    neg = 0
    for row in data:
        if row[0] > 0:
            pos += 1
        else:
            neg += 1
    return (pos, neg)

def calcEntropy(cClass1, cClass2):
    if cClass1 == 0 or cClass2 == 0:
        return 0
    total = cClass1 + cClass2

    h1 = -((cClass1/total) * np.log2(cClass1/total))
    h2 = -((cClass2/total) * np.log2(cClass2/total))
    h = h1 + h2

    return h

def calcInfGain(data, nodeT, Hs):
    s1 = []
    s2 = []
    idx = nodeT.featureIdx
    thold = nodeT.threshold
    sSize = np.shape(data)[0]
    for row in data:
        val = row[idx]
        if val < thold:
            s1.append((row[0], val))
        else:
            s2.append((row[0], val))

    s1Size = len(s1)
    s2Size = len(s2)

    (s1Pos, s1Neg) = getClassCounts(s1)
    (s2Pos, s2Neg) = getClassCounts(s2)
    
    Hs1 = calcEntropy(s1Pos, s1Neg)
    Hs2 = calcEntropy(s2Pos, s2Neg)
    if Hs1 < Hs2:
        if s1Pos > s1Neg: #values lower than thold in positive class
            nodeT.direction = POSLEFT
        else:
            nodeT.direction = POSRIGHT
    else:
        if s2Pos >  s2Neg:
            nodeT.direction = POSRIGHT
        else:
            nodeT.direction = POSLEFT

    iGain = Hs - ((s1Size/sSize) * Hs1) - ((s2Size/sSize) * Hs2)
    return iGain


def findThreshold(data, featIdx, Hs):
    fArr = []
    for row in data:
        fArr.append((row[0], row[featIdx]))
    sArr = sorted(fArr, key=lambda x: x[1])
    numFeats = len(sArr)
    bestNode = node(featIdx)   
    for i in range(numFeats):
        i2 = i-1
        test = False
        if i2 >= 0:
            test = (sArr[i][0] != sArr[i2][0])
        if test:
            newNode = node(featIdx, sArr[i][1])
            newNode.iGain = calcInfGain(data, newNode, Hs)
            if newNode.iGain > bestNode.iGain:
                bestNode = newNode

    return bestNode


def findBestTestFromData(data):
    (pos, neg) = getClassCounts(data)
    hs = calcEntropy(pos, neg)
    numFeats = np.shape(data)[1] #number of different features
    bestNode = node(0)
    for i in range(1, numFeats): #the first index is the class
        newNode = findThreshold(data, i, hs)
        if newNode.iGain > bestNode.iGain:
            bestNode = newNode

    return bestNode


def runTest(data, nodeT):
    featIdx = nodeT.featureIdx
    thold = nodeT.threshold
    belowT = np.zeros((1, np.shape(data)[1]))
    aboveT = np.zeros((1, np.shape(data)[1]))
    for row in data:
        if row[featIdx] < thold:
            belowT = np.insert(belowT, np.shape(belowT)[0], row, axis=0)
        else:
            aboveT = np.insert(aboveT, np.shape(aboveT)[0], row, axis=0)
    aboveT = aboveT[1:]
    belowT = belowT[1:]
    return aboveT, belowT

def evalTest(aboveT, belowT, nodeT, train=False):
    numwrong = 0
    totalA = np.shape(aboveT)[0]
    totalB = np.shape(belowT)[0]
    correct = 1 * nodeT.direction
    for row in aboveT:
        if row[0] != correct:
            numwrong += 1
    aboveWrong = numwrong
    correct *= -1
    for row in belowT:
        if row[0] != correct:
            numwrong += 1
    belowWrong = numwrong - aboveWrong
    errpct = numwrong/(totalA+totalB)
    print("Pct Error:", round(100*errpct, 3), "%")
    if train:
        print("Placed", totalA, "in positive class, misplaced:", aboveWrong, "(+{},-{})".format(totalA-aboveWrong, aboveWrong))
        print("Placed", totalB, "in the negative class, misplaced:", belowWrong, "(+{},-{})".format(belowWrong, totalB-belowWrong))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python q2_1.py train.csv test.csv")
        sys.exit()

    trainFile = sys.argv[1]
    testFile = sys.argv[2]

    trainD = np.genfromtxt(trainFile, dtype=np.float, delimiter=",")
    testD = np.genfromtxt(testFile, dtype=np.float, delimiter=",")

    bestNode = findBestTestFromData(trainD)

    bestNode.prettyPrint()

    aboveT, belowT = runTest(trainD, bestNode)
    print("Training ", end="")
    evalTest(aboveT, belowT, bestNode, True)

    aboveT, belowT = runTest(testD, bestNode)
    print("Testing ", end="")
    evalTest(aboveT, belowT, bestNode)