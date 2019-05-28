import numpy as np
import matplotlib.pyplot as plt
import sys
import random

def initializeCenters(numClusters, data):
    centersIndex = []
    arrlen = len(data)
    print(arrlen)
    while (len(centersIndex) != numClusters):
        checkCenter = random.randint(0, arrlen)
        if checkCenter not in centersIndex:
            centersIndex.append(checkCenter)

    centers = []
    for i in range(numClusters):
        print(centersIndex[i])
        centers.append(data[centersIndex[i]])
    #centers.append(random.choice(data))
    #for i in range(numClusters - 1):
    #    newCenter = random.choice(data)
    #    while newCenter in centers:
    #        newCenter = random.choice(data)
    #    centers.append(newCenter)
    print(len(centers))
    print(centers)
    return centers

#returns index of closest cluster within centers
def assignToCluster(centers, dataPoint):
    minCenter = 0
    centerIndex = 0
    for c in centers:
        #changing data[x] & centers[c] to x & c resp. because python syntax is spooky. Same with c to centers.index(c)
        if np.linalg.norm(dataPoint - c) < np.linalg.norm(dataPoint - centers[minCenter]):
            print("new min center")
            minCenter = centerIndex
        centerIndex += 1
    print("min center: ", minCenter)
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
print("testing K-means with: \'" , numClusters , "\' clusters...")

file = open("p4-data.txt", "r")
data = np.genfromtxt(file, dtype=np.int, delimiter=",")
#data = [1, 2, 3, 4] #test to run data

#initialization
centers = initializeCenters(numClusters, data)


#Execute loop until convergence
clusters = [[]] #clusters to which the data is assigned to. ##BUG: Need to create 2-D array within initial array size of k
sseOfIteration = [] #holds the sse of each iteration
iteration = 1
while True:
    oldClusters = clusters

    #Assignment Step
    for x in data:
        clusterIndex = assignToCluster(centers, x) #returns index of closest cluster to data point
        print("clusterIndex: " ,clusterIndex) #running into issue when clusterIndex is 1. Need clusters to have 2D array size of k
        clusters[clusterIndex].append(x) #add data to the closest cluster center

    #Update Step
    for j in range(numClusters):
        centers[j] = updateCenter(clusters[j]) #returns new center of given cluster

    #Determine SSE
    sse = findSSE(numClusters, clusters, centers)
    print("SSE for iteration " + iteration + ": " + sse)
    sseOfIteration.append(sse)

    #Determine Convergence (can be done before Update Step)
    if oldClusters == clusters:
        break; #Clusters haven't changed. Convergence reached

    iteration += 1
#Now Graph the SSE
plotSSE(sseOfIteration)
