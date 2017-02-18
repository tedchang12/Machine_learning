import numpy as np
import matplotlib.pyplot as plt	
from scipy.optimize import fmin
def parsedata(filename):
	data = open(filename,'r')
	index = 0
	data = data.readlines()
	for i in range(len(data)):
		if(data[i][0]=='#'):
			index = index+1
		else:
			break
	data = data[index:len(data)-1]
	return data
def plus1Dary(data):
	x = (np.array(data[:,0])).reshape(len(data),-1)
	x = np.concatenate((np.ones((len(x),1)),x),axis=1) # x = [1 x]
	y = (data[:,1]).reshape(len(data),-1)
	return x,y
def costfunc(x,y,theta):
	size = len(x)
	losselement = np.dot(x,theta)-y
	return ((sum(losselement*losselement))/(2*size))
def grad(x,y,alpha,iterations):
	Theta = np.zeros((x.shape[1],1))
	for i in range(iterations):
		cost = costfunc(x,y,Theta)
		subsum = np.dot(x,Theta)-y
		Theta = Theta-((alpha/len(x)* np.dot(np.transpose(x),subsum)))
		print('#iteration:%d cost:%f' %(i,cost))
	return Theta
def plotdata(d,d1,s):
	x = min(d)-1
	x1 = max(d)+1
	y = min(d1)-1
	y1 = max(d1)+1
	plt.plot(d, d1, s)
	plt.axis([x, x1, y, y1])

###load data
data = parsedata("ex1data1.txt")
data = np.loadtxt(data,delimiter=',')
x = (np.array(data[:,0])).reshape(len(data),-1)
x = np.concatenate((np.ones((len(x),1)),x),axis=1) # x = [1 x]
y = (data[:,1]).reshape(len(data),-1)
###plot data
plotdata(x[:,1],y,'ro')
###training
iterations = 100
alpha = 0.006
Theta = grad(x,y,alpha,iterations)
###show image
d = np.arange(1,30,1)
y = Theta[0]+d*Theta[1]
plt.plot(y)
plt.show()
