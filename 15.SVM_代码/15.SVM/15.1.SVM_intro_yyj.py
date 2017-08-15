# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
	iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
	path = '..\\..\\9.Regression\\9.Regression\\iris.data'	
	data = pd.read_csv(path,header=None)
	x,y = data[[0,1]],pd.Categorical(data[4]).codes
	x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.7)
	#分类器
	#C-目标函数的惩罚系数，默认为1；kernel-‘rbf’，‘linear’，‘ploy’，‘sigmoid’，默认为rbf；
	#decision_function_shape-‘ovo’-一对一；‘ovr’-多对多，默认为None
	clf = svm.SVC(C=0.1,kernel='linear',decision_function_shape='ovr')
	#kernel='rbf'------高斯核函数
	# clf = svm.SVC(C=0.8,kernel='rbf',gamma=20,decision_function_shape='ovr')
	clf.fit(x_train,y_train)

	#准确率
	print clf.score(x_train,y_train)
	print '训练集准确度：',accuracy_score(y_train, clf.predict(x_train))
	print clf.score(x_test,y_test)
	print '测试集准确度:',accuracy_score(y_test, clf.predict(x_test))
	
	#decision_function
	#输出训练集前五行数据
	print x_train[:5]
	#decision_function-返回的是样本距离超平面的距离
	print 'decision_function()\n':,clf.decision_function(x_train)
	print 'predict:\n',clf.predict(x_train)

	#画图
	x1_min,x2_min = x.min()
	x1_max,x2_max = x.max()
	t1 = np.linspace(x1_min, x1_max,500)
	t2 = np.linspace(x2_min, x2_max,500)
	x1,x2 = np.meshgrid(t1,t2)
	x_show = np.stack((x1.flat,x2.flat),axis=1)
	y_hat = clf.predict(x_show)
	y_hat = y_hat.reshape(x1.shape)

	mpl.rcParams['font.sans-serif'] = [u'SimHei']
	mpl.rcParams['axes.unicode_minus'] = False

	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap('g', 'r', 'b')
	plt.figure(facecolor='w')
	plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
	plt.scatter(x[0], x[1],c=y,camp=cm_dark,edgecolors='k',s=50) #样本
	plt.scatter(x_test[0], x_test[1],edgecolors='k',zorder=10,s=120,facecolor='none')
	plt.xlabel(iris_feature[0],fontsize=14)
	plt.ylabel(iris_feature[1],fontsize=14)
	plt.xlim(x1_min,x1_max)
	plt.ylim(x2_min,x2_max)
	plt.grid(True)
	plt.tight_layout(2)
	plt.show()