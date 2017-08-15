# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

if __name__ == '__main__':
	data = pd.read_csv('bipartition.txt',sep='\t',header=None)
	x,y = data[range(2)],data[2]

	#分类器
	clf_params = (('linear',0.1),('linear',0.5),('linear',1),('linear',2),
					('rbf',1,0.1),('rbf',1,1),('rbf',1,10),('rbf',1,100),
					('rbf',5,0.1),('rbf',5,1),('rbf',5,10),('rbf',5,100))
	x1_min,x2_min = x.min()
	x1_max,x2_max = x.max()
	t1 = np.linspace(x1_min, x1_max,500)
	t2 = np.linspace(x2_min, x2_max,500)
	x1,x2 = np.meshgrid(t1,t2)
	x_show = np.stack((x1.flat,x2.flat),axis=1)

	cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
	cm_dark = mpl.colors.ListedColormap(['g','r'])
	mpl.rcParams['font.sans-serif'] = [u'SimHei']
	mpl.rcParams['axes.unicode_minus'] = False
	plt.figure(figsize=(14,10),facecolor='w')
	for i ,param in enumerate(clf_params):
		clf = svm.SVC(C=param[1],kernel=param[0])
		if param[0] == 'rbf':
			clf.gamma = param[2]
			title = u'高斯核，C=%.1f,$\gamma$=%.1f' % (param[1],param[2])
		else:
			title = u'线性核，C=%.1f' % param[1]
		
		clf.fit(x,y)
		y_hat = clf.predict(x)
		print '准确率:\n',accuracy_score(y,y_hat)			

		#画图
		print '支撑向量的数目：',clf.n_support_
		#参数α？
		print '支撑向量的系数：',clf.dual_coef_
		print '支撑向量：',clf.support_
		plt.subplot(3,4,i+1)
		y_grid_hat = clf.predict(x_show)
		y_grid_hat = y_grid_hat.reshape(x1.shape)
		plt.pcolormesh(x1,x2,y_grid_hat,cmap=cm_light,alpha=0.8)
		plt.scatter(x[0], x[1],c=y,cmap=cm_dark,edgecolors='k',s=40)
		#loc-选取指定列
		plt.scatter(x.loc[clf.n_support_,0],x.loc[clf.n_support_,1],edgecolors='k',marker='o',s=100 )
		print 'decision_function(x):',clf.decision_function(x)
		print 'predict(x)',clf.predict(x)
		z = clf.decision_function(x_show)
		z = z.reshape(x1.shape)
		plt.contour(x1,x2,z,colors=list('kbrbk'),linestyles=['--','--','-','--','--'],
					linewidths=[1,0.5,1.5,0.5,1],levels=[-1,-0.5,0,0.5,1])
		plt.xlim(x1_min,x1_max)
		plt.ylim(x2_min,x2_max)
		plt.title(title,fontsize=14)
	plt.suptitle(u'SVM不同参数的分类',fontsize=20)
	plt.tight_layout(2)
	plt.subplots_adjust(top=0.92)
	plt.show()