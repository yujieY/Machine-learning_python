# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__=='__main__':
	path = 'Advertising.csv'
	data = pd.read_csv(path)
	x = data[['TV','Radio','Newspaper']]
	y = data['Sales']

	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False

	#绘制1
	# plt.figure(facecolor='w')
	# plt.plot(data['TV'],y,'ro',label='TV')
	# plt.plot(data['Radio'],y,'g^',label='Radio')
	# plt.plot(data['Newspaper'],y,'m*',label='Newspaper')
	# plt.legend(loc='lower right')
	# plt.xlabel(u'广告花费',fontsize=16)
	# plt.ylabel(u'销售额',fontsize=16)
	# plt.title(u'广告花费与销售额对比数据',fontsize=20)
	# plt.grid()
	# plt.show()

	#绘制2
	# plt.figure(facecolor='w',figsize=(9,10))
	# plt.subplot(311)
	# plt.plot(data['TV'],y,'ro')
	# plt.title('TV')
	# plt.grid()
	# plt.subplot(312)
	# plt.plot(data['Radio'],y,'b^')
	# plt.title('Radio')
	# plt.grid()
	# plt.subplot(313)
	# plt.plot(data['Newspaper'],y,'m*')
	# plt.title('Newspaper')
	# plt.grid()
	# plt.tight_layout()
	# plt.show()

	x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=1)
	model = LinearRegression()
	model.fit(x_train, y_train)
	# print model
	# print model.intercept_,model.coef_
	order = y_test.argsort(axis=0)
	y_test = y_test.values[order]
	x_test = x_test.values[order,:]
	y_hat = model.predict(x_test)
	#均方误差
	mse = np.average((y_hat - np.array(y_test))**2)
	rmse = np.sqrt(mse)
	# print 'mse:',mse
	# print 'rmse',rmse
	# print 'r2(训练)=',model.score(x_train,y_train)
	# print 'r2(测试)=',model.score(x_test,y_test)

	plt.figure(facecolor='w')
	t = np.arange(len(x_test) )
	# print t
	plt.plot(t,y_test,'r-',linewidth=2,label=u'真实数据')
	plt.plot(t,y_hat,'bo',linewidth=2,label=u'测试数据')
	plt.legend(loc='upper right')
	#b显示网格，默认为true
	plt.grid(b=False)
	plt.show()

