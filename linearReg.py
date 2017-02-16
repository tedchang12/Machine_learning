import numpy as np
import matplotlib.pyplot as plt	
import pythonlib as pylib
from scipy.optimize import fmin
###load data
data = pylib.parsedata("ML_ex1/ex1data1.txt")
data = np.loadtxt(data,delimiter=',')
x = (np.array(data[:,0])).reshape(len(data),-1)
x = np.concatenate((np.ones((len(x),1)),x),axis=1) # x = [1 x]
y = (data[:,1]).reshape(len(data),-1)
###plot data
pylib.plotdata(x[:,1],y,'ro')
###training
iterations = 15
alpha = 0.006
Theta = pylib.grad(x,y,alpha,iterations)
###show image
d = np.arange(1,30,1)
y = Theta[0]+d*Theta[1]
plt.plot(y)
plt.show()
