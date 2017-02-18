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
	theta = pylib.soft_maxGrad(x,y,theta,0.0001,4000,probM,xtest,ytest)##start training
	testx = (x[y[:,0]==1])
	x = (np.argmax(testx.dot(theta),axis=1))
	for i in range(len(x)):
		print(x[i])
	np.savetxt('test.out', theta, delimiter=',')
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