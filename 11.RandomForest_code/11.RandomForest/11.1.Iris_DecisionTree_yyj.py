# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pydotplus

if __name__=='__main__':
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False

	iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
	iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
	iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

	path = '..\\..\\9.Regression\\9.Regression\\iris.data'  # 数据文件路径
	data =pd.read_csv(path,header=None)
	x = data[range(4)]
	y = LabelEncoder().fit_transform(data[4])

	x = x.iloc[:,:2]
	#在随机划分训练集和测试集时候，划分的结果并不是那么随机，也即，确定下来random_state是某个值后，重复调用这个函数，划分结果是确定的。
	x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=1)
	model = DecisionTreeClassifier(criterion='entropy')	
	model.fit(x, y)
	y_test_hat = model.predict(x_test)
	print 'accuracy_score:',accuracy_score(y_test, y_test_hat)

	#保存
	#1.输出
	# with open('iris.dot','w') as f:
	# 	tree.export_graphviz(model,out_file=f)
	# #输出为pdf格式
	# dot_data = tree.export_graphviz(model,out_file=None,feature_names=iris_feature_E,class_names=iris_class,
	# 								filled=True,rounded=True,special_characters=True)
	# graph = pydotplus.graph_from_dot_data(dot_data)
	# graph.write_pdf('iris.pdf')
	# f = open('iris.png','wb')
	# f.write(graph.create_png())
	# f.close()

	#画图
	N,M = 50,50	
	x1_min,x2_min = x.min()
	x1_max,x2_max = x.max()
	t1 = np.linspace(x1_min,x1_max,N)
	t2 = np.linspace(x2_min,x2_max,M)
	x1,x2 = np.meshgrid(t1,t2)
	# print x1.shape
	x_show = np.stack((x1.flat,x2.flat),axis=1)
	# print x_show.shape

	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
	y_show_hat = model.predict(x_show)
	# print y_show_hat.shape
	# print y_show_hat
	y_show_hat = y_show_hat.reshape(x1.shape)
	# print y_show_hat
	plt.figure(facecolor='w')
	plt.pcolormesh(x1,x2,y_show_hat,cmap=cm_light)
	plt.scatter(x_test[0], x_test[1],c=y_test.ravel(),edgecolors='k',s=150,zorder=10,cmap=cm_dark,marker='*')#测试数据
	plt.scatter(x[0], x[1],c=y.ravel(),edgecolors='k',s=40,cmap=cm_dark)#全部数据	
	plt.xlabel(iris_feature[0],fontsize=15)
	plt.ylabel(iris_feature[1],fontsize=15)
	plt.xlim(x1_min,x1_max)
	plt.ylim(x2_min,x2_max)
	plt.grid(True)
	plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
	plt.show()

	#训练集上的预测结果
	y_test = y_test.reshape(-1)
	# print y_test.shape
	#result结果为true或false
	result = (y_test_hat == y_test)
	acc = np.mean(result)
	# print '准确度：%.2f' % (100 * acc)

	#过拟合：错误率
	depth = np.arange(1,15)
	err_list = []
	for d in depth:
		clf = DecisionTreeClassifier(criterion='entropy',max_depth=d)
		clf.fit(x_train, y_train)
		y_test_hat = clf.predict(x_test)
		result = (y_test_hat == y_test)
		err = 1 - np.mean(result)
		err_list.append(err)

	plt.figure(facecolor='w')
	plt.plot(depth,err_list,'ro-',lw=2)
	plt.xlabel(u'决策树深度',fontsize=15)
	plt.ylabel(u'错误率',fontsize=15)
	plt.title(u'决策树深度与过拟合',fontsize=17)
	plt.grid(b=True,ls=':')
	plt.show()


