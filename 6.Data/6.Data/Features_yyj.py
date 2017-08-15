# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
#feature_selection(特征选择)，SelectKBest（选择排名排在前N个的变量）
#SelectPercentile（选择排名在前n%的变量），chi2（可以解决分类问题）
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2
from sklearn.linear_model import LogisticRegressionCV
#sklearn.metrics模块包括分数函数，性能度量和成对度量和距离计算。
from sklearn import metrics
from sklearn.model_selection import train_test_split
#实现参数在数据集上的重复使用
from sklearn.pipeline import Pipeline
#生成多项式和交互特征
from sklearn.preprocessing import PolynomialFeatures
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
#椭圆弧
import matplotlib.patches as mpatches

def extend(a,b):
	return 1.05*a - 0.05*b,1.05*b - 0.05*a

if __name__ =='__main__':
	stype = 'pca'
	pd.set_option('display.width',200)
	data = pd.read_csv('iris.csv',header = None)
	columns = np.array(['花萼长度','花萼宽度','花瓣长度','花瓣宽度','类型'])
	data.rename(columns = dict(zip(np.arange(5),columns)),inplace=True)
	data['类型'] = pd.Categorical(data['类型']).codes
	# print data.head()
	x = data[columns[:-1]]
	y = data[columns[-1]]

	if stype == 'pca':
		#n_components(保留主成分个数，即特征个数),whiten(白化，对变量的标准差标准化，)
		pca = PCA(n_components = 2,whiten = True,random_state = 0)
		#实现对特征信息的提取和转化
		x = pca.fit_transform(x)
		# print x
		# print '各方向方差：\n',pca.explained_variance_
		# print '方差所占比例：\n',pca.explained_variance_ratio_
		x1_lable,x2_lable = u'组分1',u'组分2'
		# title = u'鸢尾花数据PCA降维'

		cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
		cm_dark = mpl.colors.ListedColormap(['g','r','b'])
		mpl.rcParams['font.sans-serif'] = 'SimHei'
		mpl.rcParams['axes.unicode_minus'] =False
		# plt.figure(facecolor='w')
		# #用于描绘散点图
		# plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_light)
		# plt.grid(b=True, ls=':')
		# plt.xlabel(x1_lable, fontsize=14)
		# plt.ylabel(x2_lable, fontsize=14)
		# plt.title(title, fontsize=18)
		# # plt.savefig('1.png')
		# plt.show()

		x,x_test,y,y_test = train_test_split(x,y,train_size=0.7)
		#PolynomialFeatures
		#生成由特征的所有多项式组合组成的新特征矩阵，其特征的度数小于或等于指定度。 
		#例如，如果一个输入样本是二维的并且形式为[a，b]，
		#则2级多项式特征是[1，a，b，a ^ 2，ab，b ^ 2]。

		#LogisticRegressionCV
		#Cs表示正则化强度的倒数，较小的值表示更强的正则化。
		#cv表示k折交叉验证
		#fit_intercept表示是否添加常量至决策函数中

		model = Pipeline([('poly',PolynomialFeatures(degree=2,include_bias=True)),
			('lr',LogisticRegressionCV(Cs=np.logspace(-3, 4,8),cv=5,fit_intercept=False))])
		model.fit(x,y)
		# print '最优参数：\n',model.get_params('lr')['lr'].C_
		# y_hat = model.predict(x)
		# print '训练精确度：',metrics.accuracy_score(y, y_hat)
		# y_test_hat = model.predict(x_test)
		# print '测试精确度：',metrics.accuracy_score(y_test, y_test_hat)

		N,M = 500 , 500
		x1_min,x1_max = extend(x[:,0].min(),x[:,0].max())
		x2_min,x2_max = extend(x[:,1].min(),x[:,1].max())
		t1 = np.linspace(x1_min, x1_max,500)
		t2 = np.linspace(x2_min, x2_max,500)
		# print t1
		#生成网格采样点
		x1,x2 = np.meshgrid(t1,t2)
		# print x1
		#测试点
		x_show  = np.stack((x1.flat,x2.flat),axis=1)
		y_hat = model.predict(x_show)
		# print y_hat.shape
		# print x1.shape
		#使之与输入的形状相同
		y_hat = y_hat.reshape(x1.shape)
		plt.figure(facecolor='w')
		#绘制四边形网格，cmap（网格颜色）
		plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
		plt.scatter(x[:,0], x[:,1],s=30,c=y,edgecolors='k',cmap=cm_dark)
		plt.xlabel(x1_lable,fontsize=14)
		plt.ylabel(x2_lable,fontsize=14)
		plt.xlim(x1_min,x1_max)
		plt.ylim(x2_min,x2_max)
		#打开或关闭轴网络
		plt.grid(b=True,ls=':')

		patchs = [mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
				  mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
				  mpatches.Patch(color='#A0A0FF', label='Iris-virginica')]
		plt.legend(handles=patchs, fancybox=True, framealpha=0.8, loc='lower right')
		plt.title(u'鸢尾花Logistic回归分类效果', fontsize=17)
		plt.show()


