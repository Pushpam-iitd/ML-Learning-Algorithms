import csv 
import pandas as pd
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
            tempX.append([1,float(row[0]),float(row[1])])
            # muX = muX+float(col) 
# print(tempX)
with open(filename2, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        for col in row:
            tempY.append(float(col))

m = len(tempX)     # the number of examples
lr = 0.005     # learning rate
X = np.array(tempX)
Y = np.array(tempY)
# muX = (np.sum(X,axis=0))/m    #the average of Xs
# muX = np.reshape(muX,(1,3))
# print(muX)
# muX = np.repeat(muX,[m],axis =0)
# print(muX)
y1 = X[:,1]
y2 = X[:,2]
xc = X[:,[1,2]]
Z1 =[]
Z2 =[]
O1 =[]
O2 =[]
# print(xc)
for i in range(0,m):
    if Y[i]==0:
        Z1.append(xc[i][0])
        Z2.append(xc[i][1])
    else:
        O1.append(xc[i][0])
        O2.append(xc[i][1])        
    
# print(y1)
# print(y2)
# print("X is ")
# print(X)

# sigX=np.zeros((1,3))
# print(sigX)

# sigX = np.reshape((np.sum(np.square(X-muX),axis=0))/m, (1,3))
# print(sigX)
# X= (X-muX)/sigX
# print("X is ")
# print(X)


Q =np.zeros((3))
# print(Q)
# a=np.array([1,2,3])
# b=np.array([2,3,1])

# print(-np.dot(a,b))

def gqx(Q,x):
    a = 1/(1+math.exp(-np.dot(Q,x)))
    return a

# gradient
def grad(Q):
    G=np.zeros((3))
    for j in range (0,3):
        for i in range (0,m):
            G[j] = G[j]+(Y[i]-gqx(Q,X[i]))*X[i][j]
    return G


H=np.zeros((3,3))
# Hessian
def Hessian(Q):
    for j in range (0,3):
        for k in range(0,3):
            for i in range (0,m):
                g = gqx(Q,X[i])
                H[j][k] = H[j][k]- g*(1-g)*X[i][j]*X[i][k]
    Hi = np.linalg.inv(H)
    return Hi

# Hinv = np.linalg.inv(H)
# print(H)
threshold = 0.001
def converged(q1,q2):
    b=True
    for i in range (0,3): 
        if(abs(q1[i]-q2[i]))>threshold:
            b=False
            break
    return b
    
    
while True:
    Qf =  Q - np.matmul(Hessian(Q),np.transpose(grad(Q)))
    if(converged(Qf,Q)):
        break;
    Q = Qf
    


# Qf= Q - np.matmul(Hinv,np.transpose(G))
# print(Qf)
HQX = np.matmul(X,Qf)
x1 = np.linspace(0,8,10)
# print(x1)
x2=-Qf[0]/Qf[2] - x1*Qf[1]/Qf[2]
# print(O1)

plt.plot(x1, x2)
plt.scatter(O1,O2,color = "teal")
plt.scatter(Z1,Z2,color = "orangered")
plt.title("logistic regression")
plt.legend(['separator','label-1','label-0'],loc="lower right")
# plt.savefig("linearregression.png",quality=200)
plt.show()