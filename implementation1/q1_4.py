import numpy as np
import matplotlib.pyplot as plt
import sys
import utils

if len(sys.argv) < 3:
    print("Error: include both filenames")
    quit()

trainFile = sys.argv[1]
testFile = sys.argv[2]

(xTr, yTr) = utils.readFromFile(trainFile)
(xTe, yTe) = utils.readFromFile(testFile)

DMAX = 36
trErrorArr = []
teErrorArr = []

for d in range(2, DMAX, 2):
#    generate d rows and append
    for i in range(0,d):
        colsize = np.size(xTr, axis=0)
        randFeats = np.random.rand(colsize)
        xTr = np.insert(xTr,0,randFeats,axis=1)
        colsize = np.size(xTe, axis=0)
        randFeats = np.random.rand(colsize)
        xTe = np.insert(xTe, 0, randFeats, axis=1)

    xTrPrime = utils.calcXPrime(xTr)
    xTePrime = utils.calcXPrime(xTe)

    w = utils.calcLearnedWeight(xTrPrime, yTr)
    wPrime = w.T[0][:]

    trainingASE = utils.calcASE(wPrime, xTrPrime, yTr)
    testingASE = utils.calcASE(wPrime, xTePrime, yTe)
    trErrorArr.append(trainingASE)
    teErrorArr.append(testingASE)

plt.figure(1)
plt.subplot(211)
plt.plot(range(2, DMAX,2), trErrorArr, marker='o')
plt.ylabel("Training ASE")
plt.xlabel("Number of random variables")

plt.subplot(212)
plt.plot(range(2, DMAX, 2), teErrorArr, marker='o')
plt.ylabel("Testing ASE")
plt.xlabel("Number of random variables")

plt.show()