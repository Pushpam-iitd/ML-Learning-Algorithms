import csv 
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

filename = argv[1]
filename2 = argv[2]  
# gar = float(argv[3])
# initializing the X and Y lists 
tempX = [] 
tempY = []
muX = 0;
# reading csv files 
with open(filename, 'r') as csvfile:  
    csvreader = csv.reader(csvfile) 
    for row in csvreader:
        b=[]
        for st in row[0].split():
            b.append(int(st))
        tempX.append(b)
            # muX = muX+float(col) 
# print(tempX)
os =0
zs =0
with open(filename2, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        for col in row:
            if(col=='Alaska'):
                tempY.append(1)
                os=os+1
            else:
                tempY.append(0)
                zs=zs+1

m = len(tempX)     # the number of examples
# lr = 0.005     # learning rate
X = np.array(tempX)
Y = np.array(tempY)
# print(X)
# print(Y)
s1= X * Y[:, np.newaxis]
s0= X * (1-Y)[:,np.newaxis]

# print(s2)
mu0 = np.zeros((1,2))
mu1 = np.zeros((1,2))
sig0= np.zeros((2,2))
sig1= np.zeros((2,2))
sig= np.zeros((2,2))

# phi = 0
# let Alaska be 1.
phi = os/m
mu1 = (np.resize(np.sum(s1,axis=0),(1,2)))/os
mu0 = (np.resize(np.sum(s0,axis=0),(1,2)))/zs
# print(X[0]-mu0)
# print(np.transpose(X[0]-mu0))
# print("first is")
# print(np.matmul((np.transpose(X[0]-mu0)),(X[0]-mu0)))
for i in range (0,m):
    if Y[i]==0:
        sig0 = sig0 + np.matmul(np.transpose(X[i]-mu0),X[i]-mu0)
sig0 = sig0/zs

for i in range (0,m):
    if Y[i]==1:
        sig1 = sig1 + np.matmul(np.transpose(X[i]-mu1),X[i]-mu1)
sig1 = sig1/os

for i in range (0,m):
    if Y[i]==1:
        sig = sig + np.matmul(np.transpose(X[i]-mu1),X[i]-mu1)
    else:
        sig = sig + np.matmul(np.transpose(X[i]-mu0),X[i]-mu0)
sig = sig/m

# print(sig)
# print("sigs are ")
# print(sig0)
# print(sig1)
sigi= np.linalg.inv(sig)
# print(sigi)

# print((np.matmul(np.matmul(mu0,sigi),np.transpose(mu0))))
# print((np.matmul(np.matmul(mu1,sigi),np.transpose(mu1))))


q0 =  0.5*((np.matmul(np.matmul(mu0,sigi),np.transpose(mu0)))-(np.matmul(np.matmul(mu1,sigi),np.transpose(mu1))))-math.log((1-phi)/phi)
q = np.matmul(sigi,np.transpose(mu1))-np.matmul(sigi,np.transpose(mu0))
# print(q)
Q = []
Q.append(q0[0][0])
for a in q:
    Q.append(a[0])
# print(Q)

Xt = np.ones((m,3))
Xt[:,1:3] = X
# print(Xt)
HQX = np.matmul(Xt,np.transpose(Q))
y1 = X[:,0]
y2 = X[:,1]
xc = X[:,[0,1]]
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
        
x1 = np.linspace(60,150,20)
# print(x1)
x2=-Q[0]/Q[2] - x1*Q[1]/Q[2]

plt.plot(x1, x2, '-r', label='QX')
plt.scatter(O1,O2,color = "teal")
plt.scatter(Z1,Z2,color = "orangered")
plt.title("GDA Linear")
plt.legend(['separator','label-1','label-0'],loc="lower right")
plt.show()
# plt.scatter(y1,y2,c=Y)