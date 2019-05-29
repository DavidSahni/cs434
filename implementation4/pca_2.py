import numpy as np
import matplotlib.pyplot as plt
import pcaUtils as pca

dataPath = "p4-data.txt"

myNpArray = pca.getNpArrayFromFile(dataPath)
meanVec = pca.getMeanVector(myNpArray)
pca.graphVectorImage(meanVec, "Mean Image")
plt.savefig("meanImg.png")

eigenValues, eigenVectors = pca.getEigenValuesAndVectors(myNpArray)
eigenIndices = np.flip(eigenValues.argsort())
#plt.show()
plt.clf()

i = 1
for eIdx in eigenIndices[:10]: #sorts the wrong way
    eigenVec = eigenVectors[:, eIdx].T.astype(np.float)
    eigenVec = eigenVec * 255.0
   # print(eigenVec)
    #plt.figure()
    pca.graphVectorImage(eigenVec, "Eigenvector {}".format(i))
    plt.savefig("eig data/eigVec{}.png".format(i))
    i += 1
    plt.clf()



# i = 1
# for eigenIndex in eigenIndices[:10]:
#     plt.plot(eigenVectors[eigenIndex], marker="o", label=("Eigenvector " + str(i)))
#     i += 1

# plt.show()