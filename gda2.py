import csv 
import pandas as pd
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sys import argv

mpl.rcParams['lines.color'] = 'k'
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k'])
def axes():
    plt.axhline(0, alpha=.1)
    plt.axvline(0, alpha=.1)

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


phi = os/m

mu1 = (np.resize(np.sum(s1,axis=0),(1,2)))/os
mu0 = (np.resize(np.sum(s0,axis=0),(1,2)))/zs
# print("mus are: ")
# print(mu0)
# print(mu1)


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
z1 = np.linspace(50,200,20)
# print(x1)
z2=-Q[0]/Q[2] - z1*Q[1]/Q[2]



Xt = np.ones((m,3))
Xt[:,1:3] = X
# print(Xt)
HQX = np.matmul(Xt,np.transpose(Q))
y1 = X[:,0]
y2 = X[:,1]
x1 = np.linspace(60,200,20)
# print(x1)

sigi1 = np.linalg.inv(sig1)
sigi0 = np.linalg.inv(sig0)
# print("sigmas are")
# print(sig1)
# print(sig0)
sigd1 = np.linalg.det(sig1)
sigd0 = np.linalg.det(sig0)
A = -(sigi0-sigi1)/2


B = -((np.matmul(sigi1,np.transpose(mu1))-np.matmul(sigi0,np.transpose(mu0))))
# print(sig1)
C = (-1)*(((np.matmul(np.matmul(mu0,sigi0),np.transpose(mu0)))-(np.matmul(np.matmul(mu1,sigi1),np.transpose(mu1))))/2 - np.log((1-phi)/phi) - np.log((sigd1)/(sigd0))/2)
# print("values are")
# print(A)
# print(B)
# print(C[0][0])
# A[0][0]*x0*x0 + A[0][1]*x0*x1 + A[1][0]*x0*x1 + A[1][1]*x1*x1 + 2*(x0*B[0]+x1*B[1]) - C 

x = np.linspace(50, 200, 40)
y = np.linspace(200, 650, 40)
x, y = np.meshgrid(x, y)

a, b, c, d, e, f = A[0][0], 2*(A[0][1]) , A[1][1] , B[0][0], B[1][0], C[0][0]

# print("coeff are")
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
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
assert b**2 - 4*a*c > 0
# axes()
# print(a)
# plt.plot(z1, z2, color="darkred")
plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0], colors='k')
plt.scatter(O1,O2,color = "teal")
plt.scatter(Z1,Z2,color = "orangered")
plt.title("Gussian Discriminant Analysis")
plt.legend(['linear separator','Alaska','Canada'],loc="lower right")
# plt.scatter(y1,y2,c=4-Y)
plt.show()