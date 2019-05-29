import os
import numpy as np
import matplotlib.pyplot as plt
import kmeans as km


#collect sse values
#sseK2 =
#plotSSE(sseK2, "SSE vs Iteration", "Iteration")
def plotSSE(sse):
    plt.plot([2, 4, 6, 8, 10], sse, label="SSE vs K") #find how to count indexes of sse of x-axis
    plt.ylabel("Sum of Squared Errors")
    plt.xlabel("K Value")
    plt.legend()
    plt.show()

#p2_2
#minSSEOfK = []
file = open("p4-data.txt", "r")
data = np.genfromtxt(file, dtype=np.int, delimiter=",")
smallestSSE = []
for i, kClusters in enumerate([2, 4, 6, 8, 10]):
    #sseCurrentIteration = []
    for x in range(10):
        sseVal = km.runKmeans(data, kClusters, debug=False)[-1]
        print(type(smallestSSE), len(smallestSSE))
        if len(smallestSSE) <= i:
            smallestSSE.append(sseVal)
        elif sseVal < smallestSSE[i]:
            smallestSSE[i] = sseVal

plotSSE(smallestSSE)
        
        
#plotSSE(minSSEOfK, "SSE vs K-value", "K-Value")

###don't run quite yet. May not be necessary. Need to explore assignment deliverables
