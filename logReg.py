import matplotlib.pyplot as plt
import numpy as np
import pythonlib as pylib

data = pylib.parsedata('ML_ex2/ex2data1.txt')
data = np.loadtxt(data,delimiter=',')
x = data[:,0:2]
x = np.concatenate((np.ones((len(x),1)),x),axis=1) 
print(x.shape)
y = data[:,2]
y = y.reshape((-1,1))
#plot with true label and false label
xtrue = data[data[:,2]==1]
xfalse = data[data[:,2]==0]
plt.plot(xtrue[:,0],xtrue[:,1],'ro')
plt.plot(xfalse[:,0],xfalse[:,1],'bx')
#compute final theta and cost
ini_theta = np.zeros((x.shape[1],1))
cost = pylib.logicCost(x,y,ini_theta)
theta,cost = pylib.logicgrad(x,y,300000,0.003) #(x,y,iteration,alpha)
#calculate miss presiction
result = pylib.sigmoid(x.dot(theta))
mispred = np.logical_xor(result>0.5 , y==1)
error_rate = (float(np.sum(mispred))/x.shape[0])
#output result
print('Loss: %f\nmiss_prediction: %f%% boundary has been shown as follow' % (cost,error_rate*100))
plot_x = np.array(([min(x[:,2])-2,  max(x[:,2])+2]));
#Calculate the decision boundary line and plot
plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0]);
plt.plot(plot_x,plot_y)
plt.show()
