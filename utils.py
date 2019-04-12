import numpy as np

def readFromFile( fileName ):
    data = np.genfromtxt(fileName, dtype=np.float)
    lenRow = len(data[0]) -1

    y = data[:,lenRow]
    x = data[:, 0:lenRow]

    return (x, y)

def calcASE(w, x, y):
    # yCalc = np.matmul(row.T, wPrime)
    sse = 0
    i = 0
    for row in x[:]: #xPrime now x
        yCalc = np.matmul(row.T, w) #wPrime now w
        err = y[i] - yCalc
        sse += err**2
        i += 1
    ase = sse/i
    return ase

def calcXPrime(x):
        colsize = np.size(x, axis=0)
        z = np.ones(colsize)
        xPrime = np.insert(x,0,z,axis=1)
        return xPrime

def calcLearnedWeight(x, y):
    yPrime = y[:][np.newaxis]
    yPrime = np.transpose(yPrime)


    xS = np.matmul(x.T, x)
    w1 = np.linalg.inv(xS)
    # w1 = np.linalg.inv(np.square(xPrime))
    w2 = np.matmul(x.T, yPrime)
    w = np.matmul(w1, w2)
    return w
