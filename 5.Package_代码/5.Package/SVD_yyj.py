# -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image                   #Python Imaging Library,64位：pillow = PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint


def restore1(sigma,u,v,K):
	m = len(u)
	n = len(v[0])
	a = np.zeros((m,n))
	for k in range(K):
		uk = u[:,k].reshape(m,1)
		vk = v[k].reshape(1,n)
		a += sigma[k] * np.dot(uk,vk)
	a[a>255] = 255
	a[a<0] = 0
	# a = a.clip(0,255)
	# return a.astype(np.int)
	# return np.rint(a)
	# return np.rint(a).astype('uint8')
	return np.uint8(a)

def restore2(sigma,u,v,K):
	m = len(u)
	n = len(v[0])
	a = np.zeros((m,n))
	for k in range(K+1):
		for i in range(m):
			a[i] +=sigma[k] * u[i][k] * v[k]
	a[a<0] = 0
	a[a<255] = 255
	return np.uint8(a)

if __name__ == '__main__':
	A = Image.open("lena.png",'r')
	print A
	output_path = r'.\SVD_output'
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	a = np.array(A)
	print a.shape
	# K = 50
	K=12
	u_r,sigma_r,v_r = np.linalg.svd(a[:,:,0])
	u_g,sigma_g,v_g = np.linalg.svd(a[:,:,1])
	u_b,sigma_b,v_b = np.linalg.svd(a[:,:,2])
	plt.figure(figsize=(11,9),facecolor='w')
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False
	for k in range(1,K+1):
		print k
		R = restore1(sigma_r, u_r, v_r, k)
		G = restore1(sigma_g, u_g, v_g, k)
		B = restore1(sigma_b, u_b, v_b, k)
		# R = restore2(sigma_r, u_r, v_r, k)
		# G = restore2(sigma_g, u_g, v_g, k)
		# B = restore2(sigma_b, u_b, v_b, k)
		I = np.stack((R,G,B),axis=2)
		#注意img如果是uint16的矩阵而不转为uint8的话，Image.fromarray这句会报错
		Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
		if k <= 12:
			plt.subplot(3,4,k)
			plt.imshow(I)
			plt.axis('off')
			plt.title(u'奇异值个数：%d' % k)
	plt.suptitle(u'SVD与图像分解',fontsize=20)
	plt.tight_layout(0.3, rect=(0, 0, 1, 0.92))
	plt.show()