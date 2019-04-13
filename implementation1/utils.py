import numpy as np

def readFromFile( fileName , delim=None):

    data = np.genfromtxt(fileName, dtype=np.float, delimiter=delim)
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

def calcRegressionAcc(w, x, y):

    correct = 0
    for i in range(np.size(x,axis=0)):
        yTrue = y[i]
        wtx = np.dot(w.T, x[i])
        yhat = 1.0/(1.0+np.exp(-wtx))
        #print(yhat)
        if yhat >= .5 and yTrue == 1:
       # if wtx > 0 and yTrue is 1:
            correct += 1
        elif yhat < .5 and yTrue == 0:
       # elif wtx <= 0 and yTrue is 0:
            correct +=1
    return correct/i
