import matplotlib.pyplot as plt
import numpy as np
import pythonlib as pylib
import scipy.io as sio
#center = np.array(([1,4,7],[2,5,8],[3,6,9]))
#print(center)
classfier = 3
#x = np.array(([1,2,1],[4,5,3]))
x = sio.loadmat('ML_ex3/ex7data2.mat')['X']
##print the initial status
center = np.random.rand(x.shape[1],classfier)+max(x[:,1])/2 #random initial center with #cols * #centers
center = center.transpose()
index = pylib.findClosetcenter(x,center)
mean = pylib.centroidMeans(x,center,index)
pylib.kmeanplot(x,index,center)
plt.plot(center[:,0],center[:,1],'bx',ms=20.0,linewidth=10)
plt.show()
##start training
pylib.runkmean(x,center,30)