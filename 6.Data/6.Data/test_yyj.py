# -*- coding:utf-8 -*-

import numpy as np

x = np.arange(1,5,1 )
y = np.arange(5,10,1 )
n,m = np.meshgrid(x,y)
print 'n:\n',n
print 'm:\n',m# 
# show = np.stack((n.flat,m.flat),axis=0)
# print show
show = np.stack((n.flat,m.flat),axis=1)
print show