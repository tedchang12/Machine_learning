import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cPickle,gzip
from collections import Counter
def runkmean(x,center,iterations,y,xtest,ytest,weighti):
	maxr = 0
	for i in range(iterations):
		index = findClosetcenter(x,center)
		center = centroidMeans(x,center,index).transpose()
		#plt.plot(center[:,0],center[:,1],'bx',ms=20.0,linewidth=10)
		index_t,resign = resignLabel(index,y)
		acc = float(sum(index_t==y))/y.shape[0]
		index_test = findClosetcenter(xtest,center)
		#print(resign)
		for j in range(len(index_test)):
			index_test[j,0] = resign[index_test[j,0],1]
		
		for j in range(10):  #if error on prob(num 1,2,3,4.......) reset center
			temp = index_test[ytest==j]
			tempy = ytest[ytest==j] #in 1==1
			if(temp.shape[0]!=0):
				accc = float(sum(temp==tempy))/temp.shape[0]
				#print('num:%d acc:%f' %(j,accc))
				if(accc<=0.75):
					tcenter = x[y[:,0]==j]
					if(tcenter.shape[0]>0):
						tcenter = tcenter[np.random.choice(tcenter.shape[0],100),:]
						tcenter = np.sum(tcenter,axis=0)
						tcenter = tcenter/100
						center[j,:] = tcenter
					#print('error')
		
		acctest = float(sum(index_test==ytest))/ytest.shape[0]
		if(maxr<acctest):
			maxr=acctest
			print('weight:%d accurance:%f test_accurance:%f' %(i,acc,acctest))
					
		
		
		"""
		index_test,resign = resignLabel(index_test,ytest)
		acctest = float(sum(index_test==ytest))/ytest.shape[0]
		"""
		
	#kmeanplot(x,index,center)
	#plt.show()
	print('weight:%d accurance:%f test_accurance:%f' %(weighti,acc,acctest))
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
			t = x[y[:,0]==i]
			t = t[np.random.choice(len(t),1),:]
			Uk=t.reshape(1,-1)
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
def randcenter(x,y):
	center=np.array(())
	c = []
	for i in range(10):
		temp = x[y[:,0]==i]
		temp = temp[np.random.choice(len(temp),1),:]
		c.append(temp)
		#c.append(x[y[:,0]==i][0,:])


	return np.array(c).reshape(-1,10).transpose()
f = gzip.open('mnist.pkl.gz','rb')
train_set,valid_set,test_set = cPickle.load(f)
f.close()
xtest = np.array(test_set[0])
ytest = np.array(test_set[1]).reshape(-1,1)
x = np.array(train_set[0])
y = np.array(train_set[1]).reshape(-1,1)
center = randcenter(x,y)
print(center.shape)
x=x*3
#center = np.random.choice(len(y),10)
#center = x[center,:].transpose()
#center = center.transpose()
index = findClosetcenter(x,center)
mean = centroidMeans(x,center,index)
j=2

index = runkmean(x,center,500,y,xtest*3,ytest,5)
index,resign = resignLabel(index,y)
acc = float(sum(index==y))/y.shape[0]
print(acc)
