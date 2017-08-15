# -*- coding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pandas as pd
import csv

def load_data(file_name,is_train):
	data = pd.read_csv(file_name)
	pd.set_option('display.width',400)
	# print 'data.describe():\n',data.describe()

	#性别
	data['Sex'] = pd.Categorical(data['Sex']).codes

	#补齐船票价格缺失值
	if len(data.Fare[data.Fare == 0]) > 0:
		fare = np.zeros(3)
		for f in range(0,3):
			#median------某列的中位数；dropna---非空值
			fare[f] = data[data['Pclass'] == f+1]['Fare'].dropna().median()
		# print 'fare:\t',fare
		for f in range(0,3):
			data.loc[(data.Fare == 0) & (data.Pclass == f+1),'Fare'] = fare[f]

	#年龄：使用均值代替缺失值
	# mean_age = data['Age'].dropna().mean()
	# data.loc[(data.Age.isnull()),'Age'] = mean_age
	if is_train:
		#年龄：使用随机森林预测年龄缺失值
		# print '随机森林预测缺失年龄：--start--'
		data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
		age_exist = data_for_age.loc[(data.Age.notnull())]
		# print '-----------------------------------',data_for_age[:,0]
		age_null = data_for_age.loc[(data.Age.isnull())]		
		# print 'age_exist:\n',age_exist
		x = age_exist.values[:,1:]
		y = age_exist.values[:,0]
		rf = RandomForestRegressor(n_estimators=20)
		rf.fit(x,y)
		age_hat = rf.predict(age_null.values[:,1:])
		# print 'age_hat:\n',age_hat
		data.loc[(data.Age.isnull()),'Age'] = age_hat
		# print '随机森林预测缺失年龄：--over--'
	else:
		# print '随机森林预测缺失年龄2：--start--'
		data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
		age_exist = data_for_age.loc[(data.Age.notnull())]
		age_null = data_for_age.loc[(data.Age.isnull())]
		x = age_exist.values[:,1:]
		y = age_exist.values[:,0]
		rf = RandomForestRegressor(n_estimators=1000)
		rf.fit(x,y)
		age_hat = rf.predict(age_null.values[:,1,:])
		# print 'age_hat:\n',age_hat
		data.loc[(data.Age.isnull()),'Age'] = age_hat
		# print '随机森林预测缺失年龄2：--over--'
		#cut--------分组(bins-分组数目)
		data['Age'] = pd.cut(data['Age'],bins=6,label=np.arange(6))

	#起始城市
	data.loc[(data.Embarked.isnull()),'Embarked'] = 'S'  #保留缺失出发城市
	embarked_data = pd.get_dummies(data.Embarked)
	# print 'embarked_data:\n',embarked_data	
	# embarked_data = embarked_data.rename(columns={'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'U': 'UnknownCity'})
	embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))	
	#concat文本链接，axis=1表示columns链接
	data = pd.concat([data,embarked_data],axis=1)	
	data.to_csv('new_titanic.csv')

	x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
	y = data['Survived']

	x = np.array(x)
	y = np.array(y)

	x = np.tile(x, (5, 1))
	y = np.tile(y, (5, ))
	# print x.shape
	if is_train:
		return x,y
	return x,data['PassengerId']

if __name__ == '__main__':
	x,y = load_data('Titanic.train.csv', True)
	x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.7)
	#penalty------L2正则化项
	lr = LogisticRegression(penalty='l2')	
	lr.fit(x, y)
	y_hat = lr.predict(x_test)
	lr_acc = accuracy_score(y_test, y_hat)

	rf = RandomForestClassifier(n_estimators=100)
	rf.fit(x_train,y_train)
	y_hat = rf.predict(x_test)
	rf_acc = accuracy_score(y_test, y_hat)

	data_train = xgb.DMatrix(x_train,label=y_train)
	data_test = xgb.DMatrix(x_test,label=y_test)
	watchList = [(data_test,'eval'),(data_train,'train')]
	param = {'max_depth':6,'eta':0.8,'silent':1,'objective':'binary:logistic'}
	bst = xgb.train(param, data_train,num_boost_round=100,evals=watchList)	
	y_hat = bst.predict(data_test)

	y_hat[y_hat > 0.5] = 1
	y_hat[~(y_hat > 0.5)] = 0
	xgb_acc = accuracy_score(y_test, y_hat)

	print 'Logistic回归：%.3f%%' % (100 * lr_acc)
	print '随机森林：%.3f%%' % (100 * rf_acc)
	print 'XGBoost：%.3f%%' % (100 * xgb_acc)	