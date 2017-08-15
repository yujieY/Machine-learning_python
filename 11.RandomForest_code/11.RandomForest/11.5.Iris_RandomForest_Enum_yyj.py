# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ =='__main__':
	mpl.rcParams['font.sans-serif'] = [u'SimHei']
	mpl.rcParams['axes.unicode_minus'] = False

	iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
	iris_features = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
	path = '..\\..\\9.Regression\\9.Regression\\iris.data'
	data = pd.read_csv(path,header=None)
	x_prime = data[range(4)]
	y = pd.Categorical(data[4]).codes
	x_prime_train,x_prime_test,y_train,y_test = train_test_split(x_prime,y,train_size=0.7,random_state=0)

	feature_paris = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
	plt.figure(figsize=(11,8),facecolor='#FFFFFF')
	for i,pair in enumerate(feature_paris):
		x_train = x_prime_train[pair]
		x_test = x_prime_test[pair]

		#n_estimators(决策树个数)，oob_score（决策树训练中没有被bootstrap选中的数据,约37%，即验证数据）
		model = RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=5,oob_score=True)
		model.fit(x_train,y_train)

		N,M = 500,500
		x1_min,x2_min = x_train.min()
		x1_max,x2_max = x_train.max()
		t1 = np.linspace(x1_min,x1_max, N)
		t2 = np.linspace(x2_min,x2_max, M)
		x1,x2 = np.meshgrid(t1,t2)
		x_show = np.stack((x1.flat,x2.flat),axis=1)

		#训练集上的预测结果
		y_train_hat = model.predict(x_train)
		acc_train = accuracy_score(y_train, y_train_hat)
		y_test_hat = model.predict(x_test)
		acc_test = accuracy_score(y_test, y_test_hat)
		print '结果如下：\n'
		print '特征：',iris_features[pair[0]],'+',iris_features[pair[1]]
		print 'OOB_SCORE:',model.oob_score_
		print '训练集准确率：%.4f' % (100 * acc_train)
		print '测试集准确率：%.4f' % (100 * acc_test)

		cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
		cm_dark = mpl.colors.ListedColormap(['g','r','b'])
		y_hat = model.predict(x_show)
		y_hat = y_hat.reshape(x1.shape)
		plt.subplot(2,3,i+1)
		plt.contour(x1,x2,y_hat,colors='k',levels=[0,1],antialiased=True, linestyles='--', linewidths=1.5)
		plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
		plt.scatter(x_train[pair[0]], x_train[pair[1]],c=y_train,s=20,edgecolors='k',cmap=cm_dark,label=u'训练集')
		plt.scatter(x_test[pair[0]], x_test[pair[1]],c=y_test,s=100,marker='*',edgecolors='k',cmap=cm_dark,label=u'测试集')		
		plt.xlabel(iris_feature[pair[0]],fontsize=15)
		plt.ylabel(iris_feature[pair[1]],fontsize=15)
		plt.xlim(x1_min,x1_max)
		plt.ylim(x2_min,x2_max)
		plt.grid(True)
	plt.suptitle(u'随机森林对鸢尾花数据两特征组合的分类结果', fontsize=18)
	plt.tight_layout(2)
	plt.show()