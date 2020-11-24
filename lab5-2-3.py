# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:10:28 2020

@author: 75965
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def fisher_lda(m1,m2,c1,c2):
    c=c1+c2
    c_inv=np.linalg.inv(c)
    return c_inv @(m1-m2)
def gauss_2D_value(x,m,c):
    c_inv=np.linalg.inv(c)
    c_det=np.linalg.det(c)
    num=np.exp(-0.5*np.dot((x-m).T ,np.dot(c_inv,(x-m))))
    den=2*np.pi*np.sqrt(c_det)
    return num/den

def gauss_2D_posterior_value(x,m1,c1,p1,m2,c2,p2):
    num=p1*gauss_2D_value(x,m1,c1)
    den=p1*gauss_2D_value(x,m1,c1)+p2*gauss_2D_value(x,m2,c2)
    return num/den

def gauss_2D_grid_values(n,m,c,xlim,ylim):
    x=np.linspace(xlim[0], xlim[1],n)
    y=np.linspace(ylim[0], ylim[1],n)
    X,Y=np.meshgrid(x, y,indexing='ij')
    Z=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            xvec=np.array(X[i,j],Y[i,j])
            Z[i,j]=gauss_2D_value(xvec, m, c)
    return X,Y,Z

def project(w,ps):
    return ps@w

def gauss_2D_sample(m,c,n=200):
    return np.random.multivariate_normal(m,c,n)

def plot_sample(m,c,axis,ps=None):
    if ps is None:
        ps = gauss_2D_sample(m, c)
    axis.scatter([p[0] for p in ps],[p[1] for p in ps],alpha=0.5)

def plot_contour(m,c,xlim,ylim,axis,n=50,level=5,cmap=None):
    grid_x,grid_y,grid_z=gauss_2D_grid_values(n, m, c, xlim, ylim)
    axis.contour(grid_x,grid_y,grid_z,level,cmap=cmap)

m1 = np.array([0,3])
m2 = np.array([3,2.5])
C1 = C2 = np.array([[2,1], [1,2]], np.float32)

ps1 =gauss_2D_sample(m1,C1)
ps2 =gauss_2D_sample(m2,C2)

wf=fisher_lda(m1,m2,C1,C2)
print(wf)

proj1=project(wf,ps1)
proj2=project(wf,ps2)

xlim=[-5,8]
ylim=[-4,9]

fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(8,12),gridspec_kw={'height_ratios':[2,1]})

plot_sample(m1,C1,axes[0],ps=ps1)
plot_contour(m1,C1,xlim,ylim,axes[0])
plot_sample(m2,C2,axes[0],ps=ps2)
plot_contour(m2,C2,xlim,ylim,axes[0],cmap='autumn')

axes[0].set_xlim(*xlim)
axes[0].set_ylim(*ylim)

axes[1].hist(proj1,25,alpha=0.5)
axes[1].hist(proj2,25,alpha=0.5)