# -*- coding:utf-8 -*-

import numpy as np
from os import listdir
import operator

#将图像转换为测试向量
def img2vector(filename):
	returnVect = np.zeros((1,1024))

	fr = open(filename)

	for i in range(32):
		lineStr = fr.readline()

		#将每行前32个字符转成int存入向量
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])

	return returnVect

#KNN算法实现
def knn_classify(inx,dataSet,labels,k):
	#获取样本数据量
	dataSetSize = dataSet.shape[0]

	#最小二乘法
	#计算测试数据与每个样本数据对应数据项的差值
	diff = np.tile(inx,(dataSetSize,1)) - dataSet
	sqDiff = diff ** 2
	sqDistance = (sqDiff.sum(axis=1)) ** 0.5

	#argsort按照距离从小到大排列,并返回其对应的索引
	sortedDis = sqDistance.argsort()
	classCount = {}

	#依次取出最近的K个样本
	for i in range(k):
		#记录样本所属类别
		voteLabel = labels[sortedDis[i]]
		#get如果指定的键值不存在，则返回默认值0
		classCount[voteLabel] = classCount.get(voteLabel,0) + 1

	#对类别出现的频次进行排序，key--用列元素的某个属性或函数作为关键字；operator.itemgetter--用于获取对象对应维的数据；reverse--降序
	sortedClassCount = sorted(classCount.iteritems,key=operator.itemgetter(1),reverse=True)

	#返回出现频次最高的类别
	return sortedClassCount[0][0]

#算法测试
def KNN_run():
	#样本数据的类标签列表
	train_Lables = []

	#样本数据文件列表
	trainFileList = listdir('digits/trainingDigits')
	m = len(trainFileList)

	#初始化样本数据矩阵（m*1024）
	trainMatrix = np.zeros((m,1024))

	#依次读取所有样本数据至样本矩阵
	for i in range(m):
		fileName = trainFileList[i]
		nameStr = fileName.split('.')[0]
		classNum = int(nameStr.split('_')[0])
		train_Lables.append(classNum)

		#将样本数据存入矩阵
		train_data[i,:] = img2vector('digits/trainingDigits/%s' % fileName)
	
	testFileList = listdir('digits/testDigits')	
	
	#初始化错误率
	errorRate = 0.0
	m_test = len(testFileList)

	#循环测试每个测试数据文件	
	for i in m_test:
		fileName = testFileList[i]
		fileStr = fileName.split('.')[0]
		classNum = int(fileStr.split('_')[0])

		#提取测试数据向量
		vectorTestData = img2vector('digits/testDigits/%s' % fileName)
		#对数据文件进行分类
		classifyResult = knn_classify(vectorTestData, train_data, train_Lables, 3)

		#输出Knn算法分类结果和真实的分类
		print 'the classifier came back with: %d, the real answer is: %d' % (classifyResult !=,classNum)
		#输出Knn算法的分类准确率
		print 'knn error Rate:\t%.3f%%', % (sum(classifyResult != classNum)/m_test * 100)		

KNN_run()