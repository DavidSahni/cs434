import numpy as np
import sys

# helper functions
import utils as u

if len(sys.argv) < 3:
    sys.exit("give the housing data .txt files next time!")

print("Running tests with the removed dummy data")

#Training Data


(x,y) = u.readFromFile(sys.argv[1])


#calculate learned weight
w = u.calcLearnedWeight(x, y)
print("Learned Weights w/o dummy data:")
print(w)
wPrime = w.T[0][:]


trainASE = u.calcASE(wPrime, x, y)
print("ASE for training data w/o dummy data:")
print(trainASE)

#Testing data

(xTest, yTest) = u.readFromFile(sys.argv[2])

testASE = u.calcASE(wPrime, xTest, yTest)
print("ASE for testing data w/o dummy data:")
print(testASE)



#for i in range(0, colsize):


# y = y[:][np.newaxis]
# yPrime = np.transpose(y)
# print(yPrime)  idk if we need this
