import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cPickle,gzip
import pythonlib as pylib
#load file
f = gzip.open('ML_ex3/mnist.pkl.gz','rb')
train_set,valid_set,test_set = cPickle.load(f)
f.close()
flattened_images = train_set[0]
#print(train_set.shape) 50000*784
######
def main():
	f = gzip.open('ML_ex3/mnist.pkl.gz','rb')
	train_set,valid_set,test_set = cPickle.load(f)
	f.close()
	xtest = np.array(test_set[0])
	ytest = np.array(test_set[1]).reshape(-1,1)
	x = np.array(train_set[0])
	y = np.array(train_set[1]).reshape(-1,1)
	theta = np.zeros((x.shape[1],(max(y)+1))) ##random initial theta with feature * #class
	probM = pylib.genProb(x,max(y)+1,y)  ##[0,0,1,0,0,0,0,0,0,0] if y==2
	theta = pylib.soft_maxGrad(x,y,theta,0.003,72,probM,xtest,ytest)##start training, after 70 times iteration the accurance on test set is about 92%
	###Demo function, random pick up 10 sample
	DemoRand = np.random.choice(len(ytest),10)
	for i in range(10):
		pred = np.argmax(xtest[DemoRand[i],:].dot(theta),axis=0).reshape(-1,1)
		print(pred)
		pylib.plot_num(xtest[DemoRand[i],:])
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