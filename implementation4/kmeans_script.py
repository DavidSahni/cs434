import os
import numpy as np
import matplotlib.pyplot as plt

def plotSSE(sse, plotLabel, xLabel):
    plt.plot(range(1, len(sse) + 1), sse, label="SSE vs Iteration") #find how to count indexes of sse of x-axis
    plt.ylabel("Sum of Squared Errors")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()

#p2_1:
cmd = "python kmeans.py 2"
#collect sse values
#sseK2 =
#plotSSE(sseK2, "SSE vs Iteration", "Iteration")


#p2_2
#minSSEOfK = []
for kClusters in [2, 4, 6, 8, 10]:
    #sseCurrentIteration = []
    for x in range(10):
        cmd = "python kmeans.py %d" % (kClusters)
        #collect sse values of this iteration
        #sseCurrentIteration.append()
    #take lowest sse within sseCurrentIteration
    #minSSEOfK.append(np.min(sseCurrentIteration))

#plotSSE(minSSEOfK, "SSE vs K-value", "K-Value")

###don't run quite yet. May not be necessary. Need to explore assignment deliverables
