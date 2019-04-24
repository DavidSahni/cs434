import q2_1 as dt
import numpy as np
import sys



#recursively calculate the best decision, seperate the two groups, and call it again

def learnDecisionTree(data, D, depth):
    depth +=1
    
    bestNode = dt.findBestTestFromData(data)
    aboveT, belowT = dt.runTest(data, bestNode)
    
    if (depth < D):
        bestNode.childA = learnDecisionTree(aboveT, D, depth)
        bestNode.childB = learnDecisionTree(belowT, D, depth)
    return bestNode


def evalTest(aboveT, belowT, nodeT):
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
    return (numwrong, totalA+totalB)


def evalDecisionTree(data, rootNode):
    aboveT, belowT = dt.runTest(data, rootNode)
    numWrong = 0
    numTotal = 0
    if rootNode.childA:
        w, t = evalDecisionTree(aboveT, rootNode.childA)
        numWrong += w
        numTotal += t
    if rootNode.childB:
        evalDecisionTree(belowT, rootNode.childB)
    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python q2_1.py train.csv test.csv")
        sys.exit()

    trainFile = sys.argv[1]
    testFile = sys.argv[2]

    trainD = np.genfromtxt(trainFile, dtype=np.float, delimiter=",")
    testD = np.genfromtxt(testFile, dtype=np.float, delimiter=",")



