# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:26:41 2020

@author: 75965
"""

# Define a range over which to slide a threshold
#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
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

C = np.array([[2,1], [1,2]], np.float32)

Ci = np.linalg.inv(2*C)
uF = Ci @ (m1-m2)
print(uF.shape)
yp1 = X1 @ uF
yp2 = X2 @ uF

#new code:
pmin = np.min( np.array( (np.min(yp1), np.min(yp2) )))
pmax = np.max( np.array( (np.max(yp1), np.max(yp2) )))
print(pmin, pmax)
# Set up an array of thresholds
#
nRocPoints = 50;
thRange = np.linspace(pmin, pmax, nRocPoints)
ROC = np.zeros( (nRocPoints, 2) )
# Compute True Positives and False positives at each threshold
#
for i in range(len(thRange)):
    thresh = thRange[i]
    TP = len(yp2[yp2 > thresh]) * 100 / len(yp2)
    FP = len(yp1[yp1 > thresh]) * 100 / len(yp1)
    ROC[i,:] = [TP, FP]
# Plot ROC curve
#
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(ROC[:,0], ROC[:,1], c='m')
ax.set_xlabel('False Positive')
ax.set_ylabel('True Positive')
ax.set_title('Receiver Operating Characteristics')
ax.grid(True)
plt.savefig('rocCurve.png')