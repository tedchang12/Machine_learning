import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cPickle,gzip
from collections import Counter
def runkmean(x,center,iterations,y,xtest,ytest):
	for i in range(iterations):
		index = findClosetcenter(x,center)
		center = centroidMeans(x,center,index).transpose()
		#plt.plot(center[:,0],center[:,1],'bx',ms=20.0,linewidth=10)
		index_t,resign = resignLabel(index,y)
		acc = float(sum(index_t==y))/y.shape[0]
		index_test = findClosetcenter(xtest,center)
		index_test,resign = resignLabel(index_test,ytest)
		acctest = float(sum(index_test==ytest))/ytest.shape[0]
		print('iteration:%d accurance:%f test_accurance:%f' %(i,acc,acctest))
	#kmeanplot(x,index,center)
	#plt.show()
	return index
		
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
			Uk=np.random.rand(1,center.shape[1])
		mean = np.append(mean,Uk)
		
	return mean.reshape((-1,x.shape[1])).transpose()
def resignLabel(index,y):
	index_t = []
	for i in range(10):
		mostFreq = np.array(y[index[:,0]==i])	
		counts = np.bincount(mostFreq[:, 0])
		if(counts.shape[0]>0):
			index_t.append([i,np.argmax(counts)])
		else:
			index_t.append([i,i])
	index_t = np.array(index_t).reshape(-1,2)
	for i in range(len(index)):
		index[i,0] = index_t[index[i,0],1]
	return index,index_t
def findClosetcenter(x,center): #find the closest center to each node
	index = []
	center = center.transpose()
	for i in range(len(x)):
		t_center = center.transpose() - x[i,:] #compute x-center(i)
		diag = np.diag(t_center.dot(t_center.transpose())) #compute ||x-center(i)||^2
		index.append(np.argmin(diag))
	return np.array(index).reshape((-1,1)) #return minimum index which means the closest to x
f = gzip.open('mnist.pkl.gz','rb')
train_set,valid_set,test_set = cPickle.load(f)
f.close()
xtest = np.array(test_set[0])
ytest = np.array(test_set[1]).reshape(-1,1)
x = np.array(train_set[0])
y = np.array(train_set[1]).reshape(-1,1)
center = np.random.rand(x.shape[1],10)+max(x[:,1])/2
center = center.transpose()
index = findClosetcenter(x,center)
mean = centroidMeans(x,center,index)
index = runkmean(x,center,1000,y,xtest,ytest)
index,resign = resignLabel(index,y)
acc = float(sum(index==y))/y.shape[0]
print(acc)
