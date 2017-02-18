import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cPickle,gzip
#load file
f = gzip.open('mnist.pkl.gz','rb')
train_set,valid_set,test_set = cPickle.load(f)
f.close()
flattened_images = train_set[0]
#print(train_set.shape) 50000*784
def validSoftmax(xtest,ytest,theta_t,iteration,x,y):
	pred = np.argmax(xtest.dot(theta_t),axis=1).reshape(-1,1)
	predx = pred==ytest
	result = float(sum(predx))/pred.shape[0]
	predt = np.argmax(x.dot(theta_t),axis=1).reshape(-1,1)
	predtx = predt==y
	resultTrain = float(sum(predtx))/predt.shape[0]
	print("Iteration:%d Precision:%f \nPrecisionTrain:%f" %(iteration,result,resultTrain))
def plot_num(data):
	pixel = data.reshape((28,28))
	plt.imshow(pixel,cmap='gray')
	plt.show()
def soft_maxGrad(x,y,theta,lamda,iterations,probM,xtest,ytest):
	for i in range(iterations):
		cost,subsum,alpha,xmultheta = softmaxCost(x,y,theta,lamda)
		prob = -(np.exp(xmultheta-alpha)/subsum)
		prob = -(x.transpose().dot(prob+probM)/x.shape[0])+lamda*theta
		theta = theta-prob
		if(i%10==0):
			print('Number of iteration:%d Cost:%f'%(i,cost))
			validSoftmax(xtest,ytest,theta,i,x,y)
	return theta
def softmaxCost(x,y,theta,lamda):
	cost = 0
	sumXsquare = (sum(sum(x**2)))*lamda/2 ##regularization
	xmultheta = x.dot(theta)
	alpha = np.max(xmultheta)
	subsum = sum(np.nan_to_num(np.exp(xmultheta-alpha)).transpose()).reshape(-1,1)
	#print(subsum.shape)
	"""
	for i in range(theta.shape[1]):
		classx = (x[y[:,0]==i])
		classx = np.exp(classx.dot(theta[:,i])-alpha).reshape(-1,1)
		classsum = (subsum[y[:,0]==i])
		cost = cost +sum(classx/classsum)
	cost = -cost/x.shape[0]+sumXsquare
	"""
	return cost,subsum,alpha,xmultheta
def genProb(x,t,y):
	prob = np.zeros((x.shape[0],t))
	for i in range(len(x)):
		prob[i,y[i]]=1
	return prob
######
def main():
	f = gzip.open('mnist.pkl.gz','rb')
	train_set,valid_set,test_set = cPickle.load(f)
	f.close()
	xtest = np.array(test_set[0])
	ytest = np.array(test_set[1]).reshape(-1,1)
	x = np.array(train_set[0])
	y = np.array(train_set[1]).reshape(-1,1)
	theta = np.zeros((x.shape[1],(max(y)+1))) ##random initial theta with feature * #class
	probM = genProb(x,max(y)+1,y)  ##[0,0,1,0,0,0,0,0,0,0] if y==2
	##start training, after 70 times iteration the accurance on test set is about 92%
	theta = soft_maxGrad(x,y,theta,0.003,10,probM,xtest,ytest) #learning rate = 0.003, iteration = 72 times
	###Demo function, random pick up 10 sample
	DemoRand = np.random.choice(len(ytest),10)
	for i in range(10):
		pred = np.argmax(xtest[DemoRand[i],:].dot(theta),axis=0).reshape(-1,1)
		print(pred)
		plot_num(xtest[DemoRand[i],:])
main()


##print(a)
"""
	for i in range(10):
		pylib.plot_num(x[i])
		print(y[i])


	for i in range(len(probM)):
		print([probM[i,:],y[i]])
	"""
	#pylib.plot_num(train_set[0])   ####plot function