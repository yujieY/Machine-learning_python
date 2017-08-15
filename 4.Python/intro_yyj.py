# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from scipy.optimize import leastsq
from scipy import stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from scipy.interpolate import BarycentricInterpolator
import scipy as sp
import math


def residual(t, x, y):
    return y - (t[0] * x ** 2 + t[1] * x + t[2])

def residual2(t, x, y):
    print t[0], t[1]
    return y - (t[0]*np.sin(t[1]*x) + t[2])    

if __name__ == '__main__':
	# a = np.arange(0,60,10).reshape((-1,1)) + np.arange(6)
	# print a

	# d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]],dtype=np.float)
	# print d
	# f = d.astype(np.int)
	# print f

	# np.set_printoptions(linewidth=100,suppress=True)
	# a = np.arange(1,10,0.5)
	# print a

	# b = np.linspace(1, 10,10,endpoint=False)
	# print b

	# c = np.logspace(1, 4,4,base = 2)
	# print c

	# d = np.logspace(0, 10,11,base = 2)
	# print d

	# e = 'Abcdezzzz'
	# f = np.fromstring(e,dtype=np.int8)
	# print f

	# a = np.arange(10 )
	# print a 
	# print a[3:6]
	# print a[3:]
	# print a[:]
	# print a[::2]
	# print a[::-1]
	# b = a[2:5]
	# b[0] = 200
	# print a

	# a = np.logspace(0,9,10,base= 2)
	# print a
	# b = np.arange(0,10,2 )
	# print b
	# c = a[b]
	# print c
	# c[0] = 100
	# print a
	# print c

	# a = np.random.rand(10)
	# print a
	# print a>0.5
	# b = a[a>0.5]
	# print b
	# a[a>0.5] = 0
	# print a
	# print b

	# a = np.arange(0,60,10 )
	# print 'a=',a
	# b = a.reshape(-1,1)
	# print b
	# c = np.arange(6)
	# d = b + c
	# print d
	# print d[[0,1,2],[2,3,4]]
	# print d[4,[2,3,4]]
	# print d[3:,[2,3,4]]

	# for j in np.logspace(0,7,8):
	# 	x = np.linspace(0,10,j)
	# 	start = time.clock()
	# 	y = np.sin(x)
	# 	t1 = time.clock() - start

	# 	x = x.tolist()
	# 	start = time.clock()
	# 	for i,t in enumerate(x):
	# 		x[i] = math.sin(t)
	# 	t2 = time.clock() - start
	# 	print j,':',t1,t2,t1/t2

	# a = np.array([1,2,3,4,4,4,5,6,7,7,8])
	# b = np.unique(a)
	# print b
	# c = np.array(((1, 2), (3, 4), (5, 6), (1, 3), (3, 4), (7, 6)))
	# b = np.unique(c)
	# print b

	# # r,i = np.split(c, (1,), axis=1)
	# # x = r + i * 1j
	# x = c[:,0] + c[:,1] * 1j
	# print x
	# print np.unique(x)
	# print np.unique(x,return_index=True)
	# a = np.unique(x,return_index=True)[1]
	# print c[a]

	# a = np.array(list(set(tuple(t) for t in c)))
	# print a

	# a = np.arange(1,7 ).reshape((2,3))
	# b = np.arange(11,17 ).reshape((2,3))
	# c = np.arange(21,27 ).reshape((2,3))
	# d = np.arange(31,37 ).reshape((2,3))

	# print 'a=\n',a
	# print 'b=\n',b
	# print 'c=\n',c
	# print 'd=\n',d
	# # s = np.stack((a,b,c,d),axis=0)
	# # print 'axis = 0',s.shape,'\n',s
	# # s = np.stack((a,b,c,d),axis=1)
	# # print 'axis = 1',s.shape,'\n',s
	# s = np.stack((a,b,c,d),axis=2)
	# print 'axis = 2',s.shape,'\n',s

	# a = np.arange(1,10 ).reshape((3,3))
	# print a
	# b = a + 10
	# print b
	# print np.dot(a, b)

	# a = np.arange(1,10 )
	# print a
	# b = np.arange(20,24 )
	# print b
	# print np.concatenate((a,b))

# #高斯分布函数图像
# 	mpl.rcParams['font.sans-serif'] = [u'SimHei']
# 	mpl.rcParams['axes.unicode_minus'] = False
# 	mu = 0
# 	sigma = 1
# 	x = np.linspace(mu - 3 * sigma, mu + 3 * sigma,51)
# 	y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
# 	print x.shape
# 	print 'x:\n',x
# 	print y.shape
# 	print 'y:\n',y
# 	plt.figure(facecolor='w')
# 	plt.plot(x,y,'ro-',linewidth=2)
# 	plt.xlabel('X',fontsize=15)
# 	plt.ylabel('Y',fontsize=15)
# 	plt.title(u'高斯分布函数',fontsize=18)
# 	plt.grid(True)
# 	plt.show()

#三维图像
	# x,y = np.mgrid[-3:3:7j,-3:3:7j]
	# print x
	# print y
	# u = np.linspace(-3, 3,101)
	# x,y = np.meshgrid(u,u)
	# # print x
	# # print y
	# z =  x*y*np.exp(-(x**2+y**2)/2) / math.sqrt(2*math.pi)
	# fig = plt.figure()
	# ax = fig.add_subplot(1,1,1,projection='3d')
	# ax.plot_surface(x,y,z,rstride=2,cstride=2,cmap=cm.gist_heat,linewidth=0.1)
	# plt.show()

#线性回归1
	# x = np.linspace(-2, 2,50)
	# A,B,C = 2,3,-1
	# y = (A*x**2+B*x+C) + np.random.rand(len(x))*0.75

	# t = leastsq(residual,[0,0,0],args=(x,y))
	# theta = t[0]
	# print '真实值：',A,B,C
	# print '预测值：',theta
	# y_hat = theta[0]*x**2+theta[1]*x+theta[2]
	# plt.plot(x,y,'r-',linewidth=2,label=u'actual')
	# plt.plot(x,y_hat,'g-',linewidth=2,label=u'predict')
	# plt.legend(loc='upper left')
	# plt.grid()
	# plt.show()

#线性回归2
	x = np.linspace(0, 5,100)
	a,w,phi = 5,1.5,-2
	y = a * np.sin(w*x) + phi + np.random.rand(len(x))*0.5

	t = leastsq(residual2, [3,5,1],args = (x, y))
	theta = t[0]
	print '真实值：',a,w,phi
	print '预测值：',theta
	y_hat = theta[0]*np.sin(theta[1]*x) + theta[2]
	plt.plot(x,y,'r-',linewidth=2,label='actual')
	plt.plot(x,y_hat,'g-',linewidth=2,label='predict')
	plt.legend(loc='lower left')
	plt.grid()
	plt.show()