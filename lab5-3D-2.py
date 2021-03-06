# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:23:47 2020

@author: 75965
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
def gauss2D(x, m, C):
    Ci = np.linalg.inv(C)  
    dC = np.linalg.det(C1)  
    num = np.exp(-0.5 * np.dot((x-m).T, np.dot(Ci, (x-m))))
    den = 2 * np.pi * dC
    return num/den

def posteriorPlot(nx, ny, m1, C1, m2, C2, P1, P2):
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            num = P1*gauss2D(xvec, m1, C1)
            den = P1*gauss2D(xvec, m1, C1) + P2*gauss2D(xvec, m2, C2)
            Z[i,j] = num / den
    return X, Y, Z

def twoDGaussianPlot (nx, ny, m, C):
    x = np.linspace(-5, 7, nx)   
    y = np.linspace(-3, 8, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')  
    Z = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xvec = np.array([X[i,j], Y[i,j]])
            Z[i,j] = gauss2D(xvec, m, C)
    return X, Y, Z

#new code:
nx, ny = 50, 40
m1 = np.array([0,3])
m2 = np.array([3,2.5])
C1 = np.array([[2,0], [0,2]], np.float32)
C2 = np.array([[1.5,0], [0,1.5]], np.float32)
P1 = P2 = 0.5

A = np.linalg.cholesky(C1)
U1 = np.random.randn(200,2)
X1 = U1 @ A.T + m1
B = np.linalg.cholesky(C2)
U2 = np.random.randn(200,2)
X2 = U2 @ B.T + m2

fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
ax[0].scatter(X1[:,0], X1[:,1], c="r", s=4)
ax[0].scatter(X2[:,0], X2[:,1], c="b", s=4)
ax[0].set_xlim(-4, 8)
ax[0].set_ylim(-4, 8)

X_p, Y_p, Z_p = posteriorPlot(nx, ny, m1, C1, m2, C2, P1, P2)
X_1, Y_1, Z_1 = twoDGaussianPlot(nx, ny, m1, C1)
X_2, Y_2, Z_2 = twoDGaussianPlot(nx, ny, m2, C2)

ax[0].contour(X_1, Y_1, Z_1, 5)
ax[0].contour(X_2, Y_2, Z_2, 5)
ax[1].contour(X_p, Y_p, Z_p, 5)


plt.show()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D surface
ax.plot_surface(X_p, Y_p, Z_p)