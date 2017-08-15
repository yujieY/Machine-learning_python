# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
	path = '..\\9.Regression\\9.Regression\\iris.data'
	# data = np.loadtxt(path,dtype=float,delimiter=',',converters={4:iris_type})
	data = pd.read_csv(path,header=None)
	x,y = data[range(4)],data[4]
	y = pd.Categorical(y).codes
	x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.7)
	
	data_train = xgb.DMatrix(x_train,label=y_train)
	data_test = xgb.DMatrix(x_test,label=y_test)
	watchList = [(data_test,'eval'),(data_train,'train')] 	
	param = {'max_depth':4,'eta':0.5,'silent':1,'objective':'multi:softmax','num_class':3}
	
	bst = xgb.train(param, data_train,num_boost_round=6,evals=watchList)
	y_hat = bst.predict(data_test)
	result = sum(y_test == y_hat)
	print '正确率：\t',float(result) / len(y_hat)

	models = [('LogisticRegression',LogisticRegressionCV(Cs=10,cv=3)),('RandomForest',RandomForestClassifier(n_estimators=30,criterion='gini'))]
	for name,model in models:
		model.fit(x_train,y_train)
		print name,'训练集正确率：',accuracy_score(y_train, model.predict(x_train))
		print name,'测试集正确率：',accuracy_score(y_test, model.predict(x_test))			



