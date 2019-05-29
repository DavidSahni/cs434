import numpy as np
import pcaUtils as pca

dataPath = "p4-data.txt"

myNpArray = pca.getNpArrayFromFile(dataPath)
eigenValues, eigenVectors = pca.getEigenValuesAndVectors(myNpArray)
eigenValues = np.sort(eigenValues.astype(np.float))
eigenValues = np.flip(eigenValues)

i = 1
for eigenValue in eigenValues[:10]:
    print("Eigenvalue " + str(i) + ":\n" + str(eigenValue))
    print("----------------------------------------------")
    i += 1