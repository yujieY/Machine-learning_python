# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np

def g_h(y_hat,y):
	p = 1.0 / (1.0 + np.exp(-y_hat))
	g = p - y.get_label()
	h = p * (1.0 - p)
	return g,h

def error_rate(y_hat,y):
	return 'error',float(sum(y.get_label() != (y_hat > 0.5)))/len(y_hat)


if __name__ == '__main__':
	#读取蘑菇数据（有无毒）
	data_train = xgb.DMatrix('agaricus_train.txt')
	data_test = xgb.DMatrix('agaricus_test.txt')
	# print data_train
	# print type(data_test)

	#设置参数
	#max_depth(最大深度)，eta（0<eta<=1.是否加入超参或学习率，1表示原目标函数）
	param = {'max_depth':3,'eta':1,'silent':1,'objective':'binary:logistic'}
	watchlist = [(data_test,'eval'),(data_train,'train')]
	#进行轮数
	n_round=7
	bst = xgb.train(param, data_train,num_boost_round=n_round,evals=watchlist,obj=g_h,feval=error_rate)
	
	#计算错误率
	y_hat = bst.predict(data_test)
	y = data_test.get_label()	
	# print y_hat
	# print y
	error = sum(y != (y_hat > 0.5))
	# error_rate = error / len(y_hat)-----------error需转换为浮点型，否则error_rate结果为0
	error_rate = float(error) / len(y_hat)
	print '样本总数：\t',len(y_hat)
	print '错误数目：\t%4d' % error
	print '错误率：\t%.5f%%' % error_rate