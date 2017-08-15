# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
	path = 'iris.data'
	data = pd.read_csv(path,header=None)
	data[4] = pd.Categorical(data[4]).codes
	x,y = np.split(data.values, (4,),axis=1)
	# print 'x:\n',x
	# print 'y:\n',y
	x = x[:,:2]
	# print 'x:\n',x
	lr = Pipeline([('sc',StandardScaler()),
					('poly',PolynomialFeatures(degree=3)),
					('clf',LogisticRegression())])
	lr.fit(x,y.ravel())
	y_hat = lr.predict(x)
	#predict_proba(分类器各自的概率)
	y_hat_prob = lr.predict_proba(x)
	# print 'y_hat:\n',y_hat
	# print 'y_hat_prob:\n',y_hat_prob
	print lr.score(x,y)
	print '准确度：%.2f' %(100*np.mean(y_hat == y.ravel()))

	#画图
	N,M = 500,500
	x1_min,x1_max = x[:,0].min(),x[:,0].max()
	x2_min,x2_max = x[:,1].min(),x[:,1].max()
	t1 = np.linspace(x1_min, x1_max,N)
	t2 = np.linspace(x2_min, x2_max,M)
	x1,x2 = np.meshgrid(t1,t2)
	x_test = np.stack((x1.flat,x2.flat),axis=1)

	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False
	cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g','r','b'])
	y_hat = lr.predict(x_test)
	y_hat = y_hat.reshape(x1.shape)

	plt.figure(facecolor='w')
	plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
	plt.scatter(x[:,0], x[:,1],c=y,edgecolors='k',s=50,cmap=cm_dark)
	plt.xlabel(u'花萼长度',fontsize=14)
	plt.ylabel(u'花萼宽度',fontsize=14)
	plt.xlim(x1_min,x1_max)
	plt.ylim(x2_min,x2_max)
	plt.grid()

	patchs = [patches.Patch(color='#77E0A0',label='Iris-setosa'),
			  patches.Patch(color='#FF8080',label='Iris-versicolor'),
			  patches.Patch(color='#A0A0FF',label='Iris-virginica')]

    # patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
    #           mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
    #           mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]

	plt.legend(handles=patchs,loc='upper right')
	plt.title(u'鸢尾花Logistic回归分类效果 - 标准化', fontsize=17)
	plt.show()


