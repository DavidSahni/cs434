import q2_1 as dt
import numpy as np
import sys
import matplotlib.pyplot as plt



#recursively calculate the best decision, seperate the two groups, and call it again

def learnDecisionTree(data, D, depth):
    depth +=1
    
    bestNode = dt.findBestTestFromData(data)
    if (bestNode.iGain == 0):
        return None
    aboveT, belowT = dt.runTest(data, bestNode)
    
    if (depth < D):
        wrongA, _ = evalTest(aboveT, 1, bestNode)
        wrongB, _ = evalTest(belowT, -1, bestNode)
        if wrongA > 0: #otherwise its a leaf
            bestNode.childA = learnDecisionTree(aboveT, D, depth)
        if wrongB > 0:
            bestNode.childB = learnDecisionTree(belowT, D, depth)
    return bestNode


def evalTest(data, correct, nodeT):
    numwrong = 0
    totalA= np.shape(data)[0]

    for row in data:
        if row[0] != correct:
            numwrong += 1

    return (numwrong, totalA)


def evalDecisionTree(data, rootNode):
    aboveT, belowT = dt.runTest(data, rootNode)
    numWrong = 0
    numTotal = 0
    if rootNode.childA:
        w, t = evalDecisionTree(aboveT, rootNode.childA)
        numWrong += w
        numTotal += t
    else: #its a leaf, evaluate it
        w, t = evalTest(aboveT, rootNode.dA, rootNode)
        numWrong += w
        numTotal += t
    if rootNode.childB:
        w, t = evalDecisionTree(belowT, rootNode.childB)
        numWrong += w
        numTotal += t
    else: #its a leaf, evaluate it
        w, t = evalTest(belowT, rootNode.dB, rootNode)
        numWrong += w
        numTotal += t    
    return (numWrong, numTotal)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("python q2_1.py train.csv test.csv d")
        sys.exit()

    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    d = int(sys.argv[3])

    trainD = np.genfromtxt(trainFile, dtype=np.float, delimiter=",")
    testD = np.genfromtxt(testFile, dtype=np.float, delimiter=",")


    trainErrArr = []
    testErrArr = []

    for i in range(1, d+1):
        root = learnDecisionTree(trainD, i, 0)
        w, t = evalDecisionTree(trainD, root)
        trainErrArr.append(round((w/t)*100, 4))
        w, t = evalDecisionTree(testD, root)
        testErrArr.append(round((w/t)*100, 4))

    plt.figure(1)
    plt.subplot(111)
    plt.plot(range(1, d+1), trainErrArr, marker='o', label="Training Error")
    plt.plot(range(1, d+1), testErrArr, marker='o', label="Testing Error")
    plt.ylabel("Error %")
    plt.xlabel("Depth Limit (d)")
    plt.legend()
    plt.show()




