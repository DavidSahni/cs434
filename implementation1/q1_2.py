import numpy as np
import sys

# helper functions
import utils as u

if len(sys.argv) < 3:
    sys.exit("give the housing data .txt files next time!")

#Training Data


(x,y) = u.readFromFile(sys.argv[1])


xPrime = u.calcXPrime(x)

#calculate learned weight
w = u.calcLearnedWeight(xPrime, y)
print("Learned Weights:")
print(w)
wPrime = w.T[0][:]


trainASE = u.calcASE(wPrime, xPrime, y)
print("ASE for training data:")
print(trainASE)

#Testing data


(xTest, yTest) = u.readFromFile(sys.argv[2])


xTestPrime = u.calcXPrime(xTest)
testASE = u.calcASE(wPrime, xTestPrime, yTest)
print("ASE for testing data:")
print(testASE)



#for i in range(0, colsize):


# y = y[:][np.newaxis]
# yPrime = np.transpose(y)
# print(yPrime)  idk if we need this
