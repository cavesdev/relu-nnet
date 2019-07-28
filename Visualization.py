#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 23:06:21 2019

@author: yangsong
"""

import matplotlib.pyplot as plt
x=np.arange(-10,10,0.1)
relu=np.copy(x);relu[relu<=0]=0;
leaky_relu=np.copy(x);leaky_relu[leaky_relu<=0]=leaky_relu[leaky_relu<=0]*0.01
sigmoid=1/(1+np.exp(-x))
tanh= (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)

ax.plot(x, relu, color='skyblue',label='relu')
ax.plot(x,leaky_relu, color='blue',label='leaky relu')
ax.plot(x,sigmoid,color='green',label='sigmoid')
ax.plot(x,tanh,color='red',label='tanh')

plt.legend(loc=2)
plt.draw()
