# -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import nmf as nmf

if __name__ == '__main__':
	A = Image.open("lena.png",'r')
	print A
	output_path = r'.\NMF_output'
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	a = np.array(A)
	print a.shape
	plt.figure(figsize=(11,9),facecolor='w')
	mpl.rcParams['font.sans-serif'] = [u'simHei']
	mpl.rcParams['axes.unicode_minus'] = False

	print a[:,:,0]

	# for k in np.arange(1, 5):
	# 	print 'k:',k
	# 	w_r,h_r = nmf.KLUpdateRule(a[:,:,0],k,10)
	# 	w_g,h_g = nmf.KLUpdateRule(a[:,:,1],k,10)
	# 	w_b,h_b = nmf.KLUpdateRule(a[:,:,2],k,10)
	# 	R = np.dot(w_r, h_r)
	# 	G = np.dot(w_g, h_g)
	# 	B = np.dot(w_b, h_b)
	# 	I = np.stack((R,G,B),axis=2)
	# 	Image.fromarray(I).save('%s\\nmf_%d.png' % (output_path,k))
	# 	plt.subplot(2,2,k)
	# 	plt.imshow(I)
	# 	plt.axis('off')
	# 	plt.title(u'K:%d' % k)
	# plt.suptitle(u'非负矩阵与图像分解',fontsize=20)
	# plt.tight_layout(0.3, rect=(0, 0, 1, 0.92))
	# plt.show()


	w_r,h_r = nmf.EuclideanDistanceUpdateRule(a[:,:,0],4,20)
	# w_g,h_g = nmf.KLUpdateRule(a[:,:,1],2,50)
	# w_b,h_b = nmf.KLUpdateRule(a[:,:,2],2,50)
	R = np.dot(w_r, h_r)
	print R
	# G = np.dot(w_g, h_g)
	# B = np.dot(w_b, h_b)
	# I = np.stack((R,G,B),axis=2)
	# Image.fromarray(I).save('%s\\nmf_%d.png' % (output_path,1))