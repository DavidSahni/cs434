import numpy as np
from numpy import linalg as LA
#from scipy import linalg as LA2
from matplotlib import pyplot as plt
### Contstants
# Assumes Python command is ran from the implementation4 folder.
dataPath = "p4-data.txt"

def getNpArrayFromFile(dataPath):
    dataVector = np.loadtxt(dataPath, delimiter=",")
    #squareDataVector = dataVector.reshape(6000, 28, 28)

    return dataVector

def getEigenValuesAndVectors(dataVector):
    # Note LA.eig(vector) expects vector to be square shaped.
    # Not sure how to do this effectively.
    covMTX = np.cov(dataVector.T)
    return LA.eig(covMTX) #covariance matrix's are always real and symmetric

def getMeanVector(dataVector):
    meanVec = np.mean(dataVector, axis=0) #defaults to flattened array if axis not set...
    return meanVec

#input should be a 784 unit vector (such as one of the data points)
def graphVectorImage(imgVector, title):
    grayScaleGrid = np.reshape(imgVector, (28, 28))
    plt.imshow(grayScaleGrid, cmap="Greys")
    plt.title(title)
    #plt.show()


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
# https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/