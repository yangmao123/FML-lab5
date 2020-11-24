# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:28:46 2020

@author: 75965
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#new code:
def calculate_roc(yp1,yp2,n=50):
    pmin = np.min( np.array( (np.min(yp1), np.min(yp2) )))
    pmax = np.max( np.array( (np.max(yp1), np.max(yp2) )))
    thRange = np.linspace(pmin, pmax, n)
    ROC = np.zeros((n, 2))
    for i, thresh in enumerate(thRange):
        TP = len(yp2[yp2 > thresh])  / len(yp2)
        FP = len(yp1[yp1 > thresh])  / len(yp1)
        ROC[i,:] = [TP, FP]
    return ROC, thRange

def calculate_auc(roc):
    return np.trapz(roc[:,1][::-1],roc[:,0][::-1])


def accuracy_at_threshold(thresh):
    tp = len(proj1[proj1 >= thresh])
    tn = len(proj2[proj2 <= thresh])
    acc = (tp +tn)/(len(proj1) + len(proj2))
    return acc


def find_best_thresh(roc, thRange):
    dist_to_best = []
    for p in roc:
        d = np.linalg.norm(p - np.asarray([0,1]))
        dist_to_best.append(d)
    best_thresh = thRange[np.argmin(dist_to_best)]
    best_acc = accuracy_at_threshold(best_thresh)
    return best_thresh, best_acc

def plot_roc(roc, ax, label=None, color=None):
    ax.plot(roc[:,0], roc[:,1], label=label, color=color)
    #axis.set_xlabel('False Positive Rate')
    #fig, ax = plt.subplots(figsize=(6,6))
    #ax.plot(roc[:,0], roc[:,1], c='m')
    ax.set_xlabel('False Positive')
    ax.set_ylabel('True Positive')
    ax.set_title('Receiver Operating Characteristics')
    ax.grid(True)
    #plt.savefig('rocCurve.png')
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

roc, thRange = calculate_roc(proj1,proj2)
auc = calculate_auc(roc)
best_thresh, best_acc = find_best_thresh(roc, thRange)

print('the area of roc',thRange)
print('AUC:{:.3f}'.format(auc))
print('Best Threshold:{:.3f}'.format(best_thresh))
print('Best Accuracy:{:.3f}'.format(best_acc))
      
fig, ax = plt.subplots(figsize=(6,6))
plot_roc(roc,ax)