import numpy as np
import sys

# helper functions
import utils as u

if len(sys.argv) < 3:
    sys.exit("give the housing data .txt files next time!")

#Training Data
train = np.genfromtxt(sys.argv[1], dtype=np.float)

(x,y) = u.readFromFile(train)

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
test = np.genfromtxt(sys.argv[2], dtype=np.float)

(xTest, yTest) = u.readFromFile(test)

xTestPrime = u.calcXPrime(xTest)
testASE = u.calcASE(wPrime, xTestPrime, yTest)
print("ASE for testing data:")
print(testASE)



#for i in range(0, colsize):


# y = y[:][np.newaxis]
# yPrime = np.transpose(y)
# print(yPrime)  idk if we need this
