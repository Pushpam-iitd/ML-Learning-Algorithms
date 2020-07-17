import csv 
import math
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation
from IPython.display import HTML, Image
# import imageio



filename = argv[1]
filename2 = argv[2] 
# initializing the X and Y lists

X = [] 
Y = []
muX = 0;
# reading csv files 
with open(filename, 'r') as csvfile:  
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        for col in row:
            X.append(float(col))
            muX = muX+float(col) 

with open(filename2, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        for col in row:
            Y.append(float(col))

m = len(X)     # the number of examples
lr = float(argv[3])     # learning rate
time_gap = float(argv[4])
muX = muX/m    #the average of Xs
sigX=0

# print(Y[2])


for x in X:
	sigX = sigX + math.pow(x-muX,2)

sigX = sigX/m
sigX = math.pow(sigX,1/2)

for i in range(0,m):
	X[i] = (X[i]-muX)/sigX

def transform(x):                #no use
	a = sigX*x + muX
	return a

def hq(x,q1,q0):
    a = q1*x + q0
    return a


def jq(q1,q0):
    J = 0
    for i in range (0, m):
        a = X[i]
        J = J + math.pow(Y[i]-hq(a,q1,q0),2)
    return J/(2*m)

def grad_q1(q1,q0):
    a=0
    for i in range (0, m):
        b = X[i]
        a = a - (Y[i] - hq(b,q1,q0))*(X[i])
    return a/m
def grad_q0(q1,q0):
    a=0
    for i in range (0, m):
        b = X[i]
        a = a - (Y[i] - hq(b,q1,q0))
    return a/m

epi = 0.001 
def doesConverge(q1,q0,q1f,q0f):
    ll = abs(jq(q1f,q0f)-jq(q1,q0))
    if ll*10000 < epi:
        # print("true difference in consecutive losses is ", ll)
        return True
    else:
        # print("difference in consecutive losses is ", ll)
        # print("\n")
        return False

q1=-40
q0=70
J_history = []
th1_history = []
th0_history = []
def gradientDescent(lr):
    global q1
    global q0
    while True:
        q1f = q1 - lr*(grad_q1(q1,q0))
        q0f = q0 - lr*(grad_q0(q1,q0)) 
        if doesConverge(q1,q0,q1f,q0f)== True:
            q1 = q1f
            q0 = q0f
            break
        else:
            q1 = q1f
            q0 = q0f
        th1_history.append(q1)
        th0_history.append(q0)
        J_history.append(jq(q1,q0))

    

gradientDescent(lr) 
# print("the parameters are ", q1,q0)



PY=[]

for i in range(0,m):
    PY.append(hq(X[i],q1,q0))

# a = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.3,1.7,2.1,2.5]

# for i in a:
#     gradientDescent(i)
#     print("final loss ",i)
#     print(jq(q1,q0))    
# print(X)

plt.scatter(X,Y)
plt.plot(X,PY)







ms = np.linspace(q0 - 40 , q0 + 40, 20)
bs = np.linspace(q1 - 40 , q1 + 40, 40)

M, B = np.meshgrid(ms, bs)

zs = np.array([jq(theta[1],theta[0]) 
               for theta in zip(np.ravel(M), np.ravel(B))])
Z = zs.reshape(M.shape)


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# PLOTTING THE SURFACE PLOT
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

figure = plt.figure(figsize = (7,7))
ax = Axes3D(figure)

#Surface plot
ax.plot_surface(M, B, Z, rstride = 1, cstride = 1, cmap = 'jet',alpha =0.6)
#ax2.plot(theta_0,theta_1,J_history_reg, marker = '*', color = 'r', alpha = .4, label = 'Gradient descent')

ax.set_xlabel('theta_0')
ax.set_ylabel('theta_1')
ax.set_zlabel('J(θ)')
# ax2.set_title('RSS gradient descent: Root at {}'.format(theta_result_reg.ravel()))
ax.view_init(45, 105)

# Create animation
line, = ax.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
point, = ax.plot([], [], [], '*', color = 'red')
display_value = ax.text(2., 2., 27.5, '')

# def init_2():
#     line.set_data([], [])
#     line.set_3d_properties([])
#     point.set_data([], [])
#     point.set_3d_properties([])
#     display_value.set_text('')

#     return line, point, display_value

def animate_2(i):
    # Animate line
    line.set_data(th0_history[:i], th1_history[:i])
    line.set_3d_properties(J_history[:i])
    
    # Animate points
    point.set_data(th0_history[:i], th1_history[:i])
    point.set_3d_properties(J_history[i])

    # Animate display value
    display_value.set_text('Loss = ' + str(J_history[i]))

    return line, point, display_value

ax.legend(loc = 1)

anim2 = animation.FuncAnimation(figure, animate_2, 
                               frames=len(th0_history), interval=1000*time_gap, 
                               repeat_delay=60, blit=True)

plt.show()


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# PLOTTING THE CONTOURS
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize = (7,7))
ax1.contour(M, B, Z, 100, cmap = 'jet')


# Create animation
line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
point, = ax1.plot([], [], '*', color = 'red', markersize = 4)
value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

# def init_1():
#     line.set_data([], [])
#     point.set_data([], [])
#     value_display.set_text('')

#     return line, point, value_display

def animate_1(i):
    # Animate line
    line.set_data(th0_history[:i], th1_history[:i])
    
    # Animate points
    point.set_data(th0_history[:i], th1_history[:i])

    # Animate value display
    value_display.set_text('Loss = ' + str(J_history[i]))

    return line, point, value_display

ax1.legend(loc = 1)

anim1 = animation.FuncAnimation(fig1, animate_1,
                               frames=len(th0_history), interval=1000*time_gap, 
                               repeat_delay=60, blit=True)
plt.show()
# HTML(anim1.to_jshtml())


