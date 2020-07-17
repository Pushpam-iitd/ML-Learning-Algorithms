import csv 
import math
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

filename = argv[1]
filename2 = argv[2] 
# initializing the X and Y lists 
tempX = [] 
tempY = []
muX = 0;
# reading csv files 
with open(filename, 'r') as csvfile:  
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        for col in row:
            tempX.append(float(col))
            # muX = muX+float(col) 

with open(filename2, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        for col in row:
            tempY.append(float(col))
m = len(tempX)     # the number of examples


Xt = np.array(tempX)
Xtt = Xt.reshape((m,1))
# Xtt = (Xtt-muX)/sigX
X = np.ones((m,2))
X[:,1:2] = Xtt 
# print(X)
Yt = np.array(tempY)
Y = Yt.reshape((m,1))
# X = (X-muX)/sigX

tau=float(argv[3])

def predict(x):
    a = []
    a = np.exp(-((np.square(x-Xt))/(2*tau*tau)))
    W = np.zeros((m,m))
    np.fill_diagonal(W,a)
    XT = np.transpose(X)
    f = np.linalg.inv(np.matmul(np.matmul(XT,W),X))
    g = np.matmul(np.matmul(XT,W),Y)
    Q = np.matmul(f,g)
    ans = x*Q[1][0]+Q[0][0]
    return ans

# print(predict((1.3-muX)/sigX))

XT = np.transpose(X)
f = np.linalg.inv(np.matmul(XT,X))
g = np.matmul(XT,Y)
Q = np.matmul(f,g)

# print(Q)
HQX = np.matmul(X,Q)

# plt.scatter(tempX,tempY)
plt.plot(tempX,HQX,color="darkred")



P=[]
K=[]
for i in range (0,m):
    K.append(X[i][1])
    P.append(predict(X[i][1]))
    # K.append(1.3)
    # P.append(predict(1.3))
# print(predict(2).shape)

plt.scatter(X[:,1],Y[:,0])
plt.scatter(K,P)
plt.show()





