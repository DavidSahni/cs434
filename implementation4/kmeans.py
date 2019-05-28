import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans #This does K-Means alg
import random

def initializeCenters(numClusters, data):
    centers = []
    centers.append(random.choice(data))
    for i in range(numClusters - 1):
        newCenter = random.choice(data)
        while (newCenter in centers):
            newCenter = random.choice(data)
        centers.append(newCenter)
    print(len(centers))
    print(centers)
    return centers

#returns index of closest cluster within centers
def assignToCluster(centers, dataPoint):
    minCenter = 0
    for c in centers:
        #changing data[x] & centers[c] to x & c resp. because python syntax is spooky. Same with c to centers.index(c)
        if np.linalg.norm(dataPoint - c) < np.linalg.norm(dataPoint - centers[minCenter]):
            minCenter = centers.index(c)
    print(minCenter)
    return minCenter

#returns new center of given cluster
def updateCenter(cluster):
    absCluster = np.absolute(cluster)
    clusterSum = np.sum(cluster)
    return (clusterSum/absCluster)

#returns sse of given clusters & centers of clusters
def findSSE(numClusters, clusters, centers):
    sseClusters = []
    for j in range(numClusters):
        sseCurrentCluster = []
        for i in clusters[j]:
            sseCurrentCluster.append(np.square(np.linalg.norm(i - centers[j])))
        sseClusters.append(np.sum(sseCurrentCluster))
    sse = np.sum(sseClusters)

    return sse

def plotSSE(sse):
    plt.plot(len(sse), sse, label="SSE vs Iteration") #find how to count indexes of sse of x-axis
    plt.ylabel("Sum of Squared Errors")
    plt.xlabel("Iteration")
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


####Starting algorithm

numClusters = int(sys.argv[1])
print("testing K-means with: [" , numClusters , "] clusters...")

#data = np.genfromtxt("p4-data.txt", dtype=np.int, delimiter=None)
data = [1, 2, 3, 4] #test to run data

#initialization
centers = initializeCenters(numClusters, data)


#Execute loop until convergence
clusters = [[]] #clusters to which the data is assigned to. ##BUG: Need to create 2-D array within initial array size of k
sseOfIteration = [] #holds the sse of each iteration
while True:
    oldClusters = clusters

    #Assignment Step
    for x in data:
        clusterIndex = assignToCluster(centers, x) #returns index of closest cluster to data point
        clusters[clusterIndex].append(x) #add data to the closest cluster center

    #Update Step
    for j in range(numClusters):
        centers[j] = updateCenter(clusters[j]) #returns new center of given cluster

    #Determine SSE
    sse = findSSE(numClusters, clusters, centers)
    sseOfIteration.append(sse)

    #Determine Convergence (can be done before Update Step)
    if oldClusters == clusters:
        break; #Clusters haven't changed. Convergence reached

#Now Graph the SSE
plotSSE(sseOfIteration)
