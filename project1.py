import numpy as np

data = np.genfromtxt('housing_train.txt', dtype=np.float)
lenRow = len(data[0]) -1

y = data[:,lenRow]
x = data[:, 0:lenRow]


colsize = np.size(x, axis=0)
z = np.ones(colsize)
xPrime = np.insert(x,0,z,axis=1)

yPrime = y[:][np.newaxis]
yPrime = np.transpose(yPrime)


xS = np.matmul(xPrime.T, xPrime)
w1 = np.linalg.inv(xS) 
# w1 = np.linalg.inv(np.square(xPrime))
w2 = np.matmul(xPrime.T, yPrime)
w = np.matmul(w1, w2)
print("Learned Weights:")
#print(w)
wPrime = w.T[0][:]

# yCalc = np.matmul(row.T, wPrime)
sse = 0
i = 0
for row in xPrime[:]:
    yCalc = np.matmul(row.T, wPrime)
    err = y[i] - yCalc
    sse += err**2
    i += 1
print(sse/colsize)


 

#for i in range(0, colsize):


# y = y[:][np.newaxis]
# yPrime = np.transpose(y)
# print(yPrime)  idk if we need this
    

