import numpy as np
import matplotlib.pyplot as plt
import pcaUtils as pca

dataPath = "p4-data.txt"

myNpArray = pca.getNpArrayFromFile(dataPath)

eigenValues, eigenVectors = pca.getEigenValuesAndVectors(myNpArray)
eigenIndices = np.flip(eigenValues.argsort())
eigenIndices = eigenIndices[:10]
print(eigenValues[eigenIndices])
bestEigVecs = eigenVectors[:, eigenIndices].astype(np.float)
bestEigVecs *= 255
projectedData = np.dot(myNpArray, bestEigVecs)


maxValues = np.argmax(projectedData, axis=0) #returns indices of 10 values across the columns (reduced dims)

i = 1
for idx in maxValues:
    plt.figure()
    pca.graphVectorImage(myNpArray[idx], "Best fit Dimension: {}".format(i))
    plt.savefig("eig data/dim{}.png".format(i))
    i += 1
    plt.clf()

