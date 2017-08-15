# -*- coding:utf-8 -*-

#numpy(数值数组和矩阵类型的基本运算)
import numpy as np
import numbers
#Scipy(高等数学、信号处理、优化、统计)
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
#LogisticRegressionCV(使用交叉验证的方式来选择正则化系数)，LogisticRegression(需指定正则化系数C，默认为1)
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
#svm(支持向量机，可用于做分类（svc）和预测（svr）)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#超参数自动搜索模块GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#label_binarize(数据二值化，将数据转化为0或1)
from sklearn.preprocessing import label_binarize
from numpy import interp 
# metrics（评估预测误差）
from sklearn import metrics
#itertools（创建有效迭代器），cycle（创建一个迭代器，对iterable中的元素反复执行循环操作，内部会生成iterable中的元素的一个副本，此副本用于返回循环中的重复项。）
from itertools import cycle


if __name__=='__main__':
	print np.random.seed(0)
	pd.set_option('display.width',300)
	np.set_printoptions(suppress=True)
	data = pd.read_csv('iris.data',header=None)
	iris_type = data[4].unique()
	# print iris_type
	#enumerate将其组成一个索引序列，利用它可以同时获得索引和值
	# for i,iris_types in enumerate(iris_type):
	# 	data.set_value(data[4] == iris_types,4,i)
	data[4] = pd.Categorical(data[4]).codes
	# print data
	x = data.iloc[:,:2]
	# print x
	n,feature = x.shape
	y = data.iloc[:,-1]
	c_number = data[4].unique().size
	# print c_number
	x,x_test,y,y_test = train_test_split(x,y,train_size=0.6,random_state=0)
	y_one_hot = label_binarize(y_test, classes=np.arange(c_number ))
	# print y_one_hot
	alpha = np.logspace(-2, 2,20)
	# print alpha
	models = [
		['KNN',KNeighborsClassifier(n_neighbors=7)],
		['LogisticRegression',LogisticRegressionCV(Cs=alpha,cv=3)],
		['SVM(Linear)',GridSearchCV(SVC(kernel='linear', decision_function_shape='ovr'),param_grid={'C':alpha})],
		['SVM(RBF)',GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'),param_grid={'C':alpha,'gamma':alpha})]]
	colors = cycle('gmcr')
	mpl.rcParams['font.sans-serif'] = u'SimHei'
	mpl.rcParams['axes.unicode_minus'] = False
	plt.figure(figsize=(7,6),facecolor='w')
	for (name,model),color in zip(models,colors):
		model.fit(x,y)
		print 'model:',name 
		#hasattr(object, name)--判断对象object是否包含命名为name的特性
		if hasattr(model, 'C_'):
			#每个折叠的最佳分数
			print 'C_:',model.C_
		if hasattr(model, 'best_params_'):
			print 'best_params_:',model.best_params_
		if hasattr(model, 'predict_proba'):
			print 'predict_proba:'
			#每个组分所占比例
			y_score = model.predict_proba(x_test)
			print 'y_socre',y_score
		else:
			y_score = model.decision_function(x_test)
			
		fpr,tpr,thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
		auc = metrics.auc(fpr, tpr)
		# print auc
		plt.plot(fpr,tpr,c=color,lw=2,alpha=0.7,label=u'%s,AUC=%.3f' %(name,auc))

		plt.plot((0,1),(0,1),c='#808080',lw=2,ls='--',alpha=0.7)
	plt.xlim((-0.01, 1.02))
	plt.ylim((-0.01, 1.02))
	plt.xticks(np.arange(0, 1.1, 0.1))
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.xlabel('False Positive Rate', fontsize=13)
	plt.ylabel('True Positive Rate', fontsize=13)
	plt.grid(b=True, ls=':')
	plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
	# plt.legend(loc='lower right', fancybox=True, framealpha=0.8, edgecolor='#303030', fontsize=12)
	plt.title(u'鸢尾花数据不同分类器的ROC和AUC', fontsize=17)
	plt.show()


