# -*- coding:utf-8 -*-

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
#StandardScaler-可以保存训练集中的参数（均值、方差）
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
	data = pd.read_csv('wine.data',header=None)
	x,y = data.iloc[:,1:],data[0]
	x = MinMaxScaler().fit_transform(x)
	x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.5)
	
	lr = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),cv=3)
	lr.fit(x_train, y_train.ravel())
	print '参数alpha：\t%.3f' % lr.alpha_
	y_train_test = lr.predict(x_train)
	y_hat = lr.predict(x_test)
	print 'RidgeClassifier训练集准确率:\t',accuracy_score(y_train, y_train_test)
	print 'RidgeClassifier测试集准确率:\t',accuracy_score(y_test, y_hat)	

	rf = RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=5,oob_score=True)
	rf.fit(x_train,y_train.ravel())		
	print 'oob_score:\t%.3f' % rf.oob_score_
	y_train_hat = rf.predict(x_train)
	y_hat = rf.predict(x_test)
	print 'RandomForestClassifier训练集准确率：',accuracy_score(y_train, y_train_hat)
	print 'RandomForestClassifier测试集准确率:',accuracy_score(y_test, y_hat)

	gb = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=2)	
	gb.fit(x_train, y_train.ravel())	
	y_train_hat = gb.predict(x_train)
	y_hat = gb.predict(x_test)
	print 'GradientBoostingClassifier训练集准确率：',accuracy_score(y_train, y_train_hat)
	print 'GradientBoostingClassifier测试集准确率:',accuracy_score(y_test, y_hat)

	y_train[y_train == 3] =0
	y_test[y_test == 3] = 0
	data_train = xgb.DMatrix(x_train,label=y_train)
	data_test = xgb.DMatrix(x_test,label=y_test)
	watchList = [(data_test,'eval',(data_train,'train'))]
	params = {'max_depth':1,'eta':0.9,'silent':1,'objective':'multi:softmax','num_class':3}
	bst = xgb.train(params, data_train,num_boost_round=5,evals=watchList)
	y_train_hat = bst.predict(data_train)
	y_hat = bst.predict(data_test)
	print 'XGBoost训练集准确率：',accuracy_score(y_train, y_train_hat)
	print 'XGBoost测试集准确率：',accuracy_score(y_test, y_hat)



