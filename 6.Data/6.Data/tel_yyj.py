# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
#将类别特征标记为 0 到classes - 1 的数
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
#将数据在缩放在固定区间的类，默认缩放到区间 [0, 1]
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
	pd.set_option('display.width',300)

	data = pd.read_csv('tel.csv', skipinitialspace=True, thousands=',')
	# print '原始数据：\n', data.head(10)
	le = LabelEncoder()
	for col in data.columns:
		data[col] = le.fit_transform(data[col])

	#年龄分组
	bins = [-1, 6, 12, 18, 24, 35, 50, 70]
	data['age'] = pd.cut(data['age'], bins=bins,labels=np.arange(len(bins)-1))
	# print data

	columns_log = ['income', 'tollten', 'longmon', 'tollmon', 'equipmon', 'cardmon',
					'wiremon', 'longten', 'tollten', 'equipten', 'cardten', 'wireten',]	
	mms = MinMaxScaler()
	for col in columns_log:
		data[col] = np.log(data[col] - data[col].min() + 1)
		data[col] = mms.fit_transform(data[col].values.reshape(-1,1))
	# print '==============================='
	# print data

	columns_one_hot = ['region', 'age', 'address', 'ed', 'reside', 'custcat']
	for col in columns_one_hot:
		data = data.join(pd.get_dummies(data[col],prefix=col))
	# print data
	data.drop(columns_one_hot,axis=1,inplace=True)
	# print '===========drop==========='
	# print data
	columns = list(data.columns)
	columns.remove('churn')
	x = data[columns]
	y = data['churn']
	print '分组与One-Hot编码后：\n',data.head(10)

	x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75)

	#n_estimators(森林中树的个数)，criterion(测量划分质量。 支持的标准是基尼系杂质的“gini”和信息增益的“熵”)
	#min_samples_split(拆分节点所需最小样本数)，
	clf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=12,min_samples_split=5,
		oob_score=True,class_weight={0: 1, 1: 1/y_train.mean()})
	#获取特征信息和目标值信息
	clf.fit(x_train,y_train)

	#特征选择
	important_features = pd.DataFrame(data={'features':x.columns,'importance':clf.feature_importances_})
	important_features.sort_values(by='importance',axis=0,ascending=False,inplace=True)
	important_features['cum_importance'] = important_features['importance'].cumsum()
	# print '特征重要度：\n',important_features
	select_features = important_features.loc[important_features['cum_importance'] < 0.95,'features']
	# print select_features

	#重新组织数据
	x_train = x_train[select_features]
	x_test = x_test[select_features]

	#模型训练
	clf.fit(x_train,y_train)
	print 'OOB socre:\n',clf.oob_score_

