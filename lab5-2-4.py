# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:24:34 2020

@author: 75965
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

nx, ny = 50, 40
m1 = np.array([0,3])
m2 = np.array([3,2.5])
C1 = C2 = np.array([[2,1], [1,2]], np.float32)
P1 = P2 = 0.5

A = np.linalg.cholesky(C1)
U1 = np.random.randn(200,2)
X1 = U1 @ A.T + m1
U2 = np.random.randn(200,2)
X2 = U2 @ A.T + m2

#new code:
C = np.array([[2,1], [1,2]], np.float32)

Ci = np.linalg.inv(2*C)
uF = Ci @ (m1-m2)
print(uF.shape)
yp1 = X1 @ uF
yp2 = X2 @ uF
matplotlib.rcParams.update({'font.size': 16})
plt.hist(yp1, bins=40,alpha=0.5)
plt.hist(yp2, bins=40,alpha=0.5)
plt.savefig('histogramprojections.png')