import matplotlib.pyplot as plt
import numpy as np
##K-means function
def runkmean(x,center,iterations):
	
	for i in range(iterations):
		
		index = findClosetcenter(x,center)
		center = centroidMeans(x,center,index).transpose()
		plt.plot(center[:,0],center[:,1],'bx',ms=20.0,linewidth=10)
	kmeanplot(x,index,center)
	plt.show()
		
def kmeanplot(x,index,center):
	symbol = ['ro','go','bo','rx']
	for i in range(center.shape[0]):
		x_class = x[index[:,0]==i]
		plt.plot(x_class[:,0],x_class[:,1],symbol[i])
	return 1
def centroidMeans(x,center,index): ##compute centroid means
	mean = np.array(())
	for i in range(center.shape[0]):
		x_class = x[index[:,0]==i]
		Ck = len(x_class)
		if(Ck!=0):
			Uk = sum(x_class)/Ck
		else:
			Uk=np.zeros((1,x.shape[1]))
		mean = np.append(mean,Uk)
	return mean.reshape((-1,x.shape[1])).transpose()

def findClosetcenter(x,center): #find the closest center to each node
	index = []
	center = center.transpose()
	for i in range(len(x)):
		t_center = center.transpose() - x[i,:] #compute x-center(i)
		diag = np.diag(t_center.dot(t_center.transpose())) #compute ||x-center(i)||^2
		index.append(np.argmin(diag))
	return np.array(index).reshape((-1,1)) #return minimum index which means the closest to x
#####need to implement polynomial function
##logistic Regession
def logicCost(x,y,theta):
	hx = sigmoid(x.dot(theta))
	subsum = (-y*np.log(hx))-((1-y)*np.log(1-hx))
	return np.sum(subsum)/x.shape[0]

def logicgrad(x,y,iterations,alpha):
	ini_theta = np.zeros((x.shape[1],1),dtype=np.float128)
	for i in range(iterations):
		cost = logicCost(x,y,ini_theta)
		hx = sigmoid(np.dot(x,ini_theta))
		subsum = hx-y
		ini_theta = ini_theta-(((alpha/len(x))* np.dot(np.transpose(x),subsum)))
	return ini_theta,cost
def sigmoid(n):
	return np.array(1/(1+np.exp(-1*n)),dtype=np.float128)
def plotdata(d,d1,s):
	x = min(d)-1
	x1 = max(d)+1
	y = min(d1)-1
	y1 = max(d1)+1
	plt.plot(d, d1, s)
	plt.axis([x, x1, y, y1])

	#plt.show()
#ary = np.array([[1,2,3,4],[1,4,9,16]])
#plotdata(ary[0,:],ary[1,:],'ro')
# red dashes, blue squares and green triangles
#x = np.arange(-5,5,0.01)
#y = -(1/(1+np.exp(x)))
#plotdata(x,y,'r-')

##linear Regression
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
		print(cost)
	return Theta

#print(x[x[:,0]==1])	   get classfy