# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
	data = pd.read_csv('Advertising.csv')
	x = data[['TV','Radio','Newspaper']]
	y = data['Sales']

	x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.8)
	model = Ridge()
	alpha = np.logspace(-3,2 ,10)
	np.set_printoptions(suppress=True)
	# print 'alpha:',alpha
	ridge_model = GridSearchCV(model, param_grid={'alpha':alpha},cv=5)
	ridge_model.fit(x_train,y_train)
	# print '超参：\n',ridge_model.best_params_

	order = y_test.argsort(axis=0)
	y_test = y_test.values[order]
	x_test = x_test.values[order,:]
	y_hat = ridge_model.predict(x_test)
	# print ridge_model.score(x_test,y_test)
	# print ridge_model.score(x_train,y_train)
	mse = np.average((y_hat - np.array(y_test)) ** 2)
	rmse = np.sqrt(mse)
	print 'mse:\n',mse
	print 'rmse:\n',rmse

	t = np.arange(len(x_test) )
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False
	plt.figure(facecolor='w')
	plt.plot(t,y_test,'y-',linewidth=2,label=u'真实数据')
	plt.legend(loc='upper right')
	plt.title(u'线性回归预测销量',fontsize=20)
	plt.grid()
	plt.show()