# -*- coding:utf-8 -*-

import numpy as np

def EuclideanDistanceUpdateRule(V, k, iters):
	m, n = V.shape
	W = np.random.random((m,k))
	H = np.random.random((k,n))

	for _ in xrange(iters):
		print _
		for a in xrange(k):
			for mu in xrange(n):
				H[a,mu] = H[a,mu] * np.dot(W.T, V)[a, mu]/np.dot(np.dot(W.T, W), H)[a, mu]
			for i in xrange(m):
				W[i,a] = W[i,a] * np.dot(V, H.T)[i,a]/np.dot(np.dot(W, H), H.T)[i, a]

	return np.rint(W).astype('uint8'), np.rint(H).astype('uint8')

def KLUpdateRule(V,k,iters):
	m, n = V.shape
	W = np.random.random((m,k))
	H = np.random.random((k,n))

	for it in xrange(iters):
		print it
		for k in  xrange(k):
			for i in xrange(m):
				W[i,k] = W[i,k] * sum(H[k,:] * V[i,:]/np.dot(W, H)[i,:])/sum(H[k,:])
#				W[i,k] = W[i,k] * sum(np.dot(H,V.T)[k,i]/np.dot(W, H)[i,:])/sum(H[k,:])

			for j in xrange(n):
				H[k,j] = H[k,j] * sum(W[:,k] * V[:,j]/np.dot(W, H[:,j]))/sum(W[:,k])
#				H[k,j] = H[k,j] * sum(np.dot(V.T,W)[j,k]/np.dot(W, H[:,j]))/sum(W[:,k])
	# print type(W)		
	# return np.rint(W).astype('uint8'), np.rint(H).astype('uint8')
	return W,H



# if __name__ == "__main__":

# #    v = np.array([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.]])
#     v = np.loadtxt(open("D:/nmf.csv","rb"),delimiter=",")

# #    w,h = EuclideanDistanceUpdateRule(v,2)
#     w,h = KLUpdateRule(v,2)
#     y = np.dot(w,h)
#     print("y:")
#     print(y)
#     print("v:")
#     print(v)
