import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import copy

def initializeCenters(numClusters, data):
    centersIndex = []
    arrlen = len(data)
    while (len(centersIndex) != numClusters):
        checkCenter = random.randint(0, arrlen)
        if checkCenter not in centersIndex:
            centersIndex.append(checkCenter)

    centers = []
    for i in range(numClusters):
        centers.append(data[centersIndex[i]])
    #centers.append(random.choice(data))
    #for i in range(numClusters - 1):
    #    newCenter = random.choice(data)
    #    while newCenter in centers:
    #        newCenter = random.choice(data)
    #    centers.append(newCenter)
    #print(centers)
    return centers

#returns index of closest cluster within centers
def assignToCluster(centers, dataPoint):
    minCenter = 0
    centerIndex = 0
    for c in centers:
        #changing data[x] & centers[c] to x & c resp. because python syntax is spooky. Same with c to centers.index(c)
        if np.linalg.norm(dataPoint - c) < np.linalg.norm(dataPoint - centers[minCenter]):
            minCenter = centerIndex
        centerIndex += 1
    return minCenter

#returns new center of given cluster
def updateCenter(cluster):
    clusterSum = np.sum(cluster, axis=0)
    newCenter = clusterSum / np.size(cluster, axis=0)
    return newCenter

#returns sse of given clusters & centers of clusters
def findSSE(numClusters, clusters, centers, data):
    sseClusters = []
    for j in range(numClusters):
        sseCurrentCluster = []
        clusterData = data[clusters[j]]
        for row in clusterData:
            sseCurrentCluster.append(np.square(np.linalg.norm(row - centers[j])))
        sseClusters.append(np.sum(sseCurrentCluster))
    sse = np.sum(sseClusters)

    return sse

def plotSSE(sse):
    plt.plot(range(1, len(sse)+1), sse, label="SSE vs Iteration") #find how to count indexes of sse of x-axis
    plt.ylabel("Sum of Squared Errors")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()


def runKmeans(data, numClusters, debug=True):
    centers = initializeCenters(numClusters, data)

    #Execute loop until convergence
    clusters = {} #clusters to which the data is assigned to. ##BUG: Need to create 2-D array within initial array size of k
    for i in range(numClusters):
        clusters[i] = []
    sseOfIteration = [] #holds the sse of each iteration
    iteration = 1
    while True:
        #oldClusters = copy.deepcopy(clusters)
        for i in range(numClusters):
            clusters[i] = []
        #Assignment Step
        for i,x in enumerate(data):
            clusterIndex = assignToCluster(centers, x) #returns index of closest cluster to data point
            #print("clusterIndex: " ,clusterIndex) #running into issue when clusterIndex is 1. Need clusters to have 2D array size of k
            #print(clusterIndex)
            clusters[clusterIndex].append(i) #add data to the closest cluster center
        #Update Step
        # if oldClusters == clusters:
        #     break


        breakIter = True        
        for j in range(numClusters):
            updatedCenter = updateCenter(data[clusters[j]])
            if not np.array_equal(updatedCenter, centers[j]):
                breakIter = False
            centers[j] = updatedCenter  #returns new center of given cluster
        
        if breakIter:
            break

        #Determine SSE
        sse = findSSE(numClusters, clusters, centers, data)
        if debug:
            print("iteration: {} SSE: {}".format(iteration, sse))
        sseOfIteration.append(sse)

        #Determine Convergence (can be done before Update Step)


        iteration += 1
    return sseOfIteration

if __name__ == "__main__":
    ####Starting algorithm

    numClusters = int(sys.argv[1])
    print("testing K-means with: \'" , numClusters , "\' clusters...")

    file = open("p4-data.txt", "r")
    data = np.genfromtxt(file, dtype=np.int, delimiter=",")
    #data = [1, 2, 3, 4] #test to run data

    #initialization
    sseVals = runKmeans(data, numClusters)
    #plotSSE(sseVals)
