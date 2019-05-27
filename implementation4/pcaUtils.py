import numpy as np
from numpy import linalg as LA
from scipy import linalg as LA2

### Contstants
# Assumes Python command is ran from the implementation4 folder.
dataPath = "p4-data.txt"

def getNpArrayFromFile(dataPath):
    dataVector = np.loadtxt(dataPath, delimiter=",")
    squareDataVector = dataVector.reshape(6000, 28, 28)

    return squareDataVector

def getEigenValuesAndVectors(dataVector):
    # Note LA.eig(vector) expects vector to be square shaped.
    # Not sure how to do this effectively.
    return LA.eig(dataVector)

### Main (for testing Utils)

# ### getNpArrayFromFileTest
# myNpArray = getNpArrayFromFile(dataPath)
# print("Number of squares: " + str(len(myNpArray)))
# print("Number of square cols: " + str(len(myNpArray[0])))
# print("Number of square rows: " + str(len(myNpArray[0][0])))
# print(str(myNpArray))

# ### getEigenValuesTest
# myNpArray = getNpArrayFromFile(dataPath)
# eigenValues, eigenVectors = getEigenValuesAndVectors(myNpArray)
# print("Eigenvalues data size: " + str(len(eigenValues)))
# print("Eigenvalues row size : " + str(len(eigenValues[0])))
# print("Eigenvalues: " + str(eigenValues))
# print("Eigenvalues data size: " + str(len(eigenVectors)))
# print("Eigenvalues row size : " + str(len(eigenVectors[0])))
# print("Eigenvalues data size: " + str(len(eigenVectors[0][0])))
# print("Eigenvectors: " + str(eigenVectors))


# ### Resources
# https://stackoverflow.com/questions/37976564/simple-plots-of-eigenvectors-for-sklearn-decomposition-pca