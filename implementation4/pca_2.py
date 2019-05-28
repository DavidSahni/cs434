import numpy as np
import matplotlib.pyplot as plt
import pcaUtils as pca

dataPath = "p4-data.txt"

myNpArray = pca.getNpArrayFromFile(dataPath)
eigenValues, eigenVectors = pca.getEigenValuesAndVectors(myNpArray)
eigenIndices = eigenValues.argsort()

# i = 1
# for eigenIndex in eigenIndices[:10]:
#     plt.plot(eigenVectors[eigenIndex], marker="o", label=("Eigenvector " + str(i)))
#     i += 1

# plt.show()