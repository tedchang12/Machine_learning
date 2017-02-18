import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
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
			Uk=np.random.rand((1,x.shape[1]))
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
#center = np.array(([1,4,7],[2,5,8],[3,6,9]))
#print(center)
classfier = 3
#x = np.array(([1,2,1],[4,5,3]))
x = sio.loadmat('ex7data2.mat')['X']
##print the initial status
center = np.random.rand(x.shape[1],classfier)+max(x[:,1])/2 #random initial center with #cols * #centers
center = center.transpose()
index = findClosetcenter(x,center)
mean = centroidMeans(x,center,index)
kmeanplot(x,index,center)
plt.plot(center[:,0],center[:,1],'bx',ms=20.0,linewidth=10)
plt.show()
##start training
runkmean(x,center,30)