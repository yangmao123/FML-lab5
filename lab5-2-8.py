# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 05:33:40 2020

@author: 75965
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
def fisher_lda(m1,m2,c1,c2):
    c=c1+c2
    c_inv=np.linalg.inv(c)
    return c_inv @(m1-m2)

def project(w,ps):
    return ps@w

def gauss_2D_sample(m,c,n=200):
    return np.random.multivariate_normal(m,c,n)

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

def calculate_auc(roc):
    return np.trapz(roc[:,1][::-1],roc[:,0][::-1])

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
nx, ny = 50, 40
m1 = np.array([0,3])
m2 = np.array([3,2.5])
C1 = C2 = np.array([[2,1], [1,2]], np.float32)
P1 = P2 = 0.5

wf=fisher_lda(m1,m2,C1,C2)

ps1 =gauss_2D_sample(m1,C1)
ps2 =gauss_2D_sample(m2,C2)

proj1=project(wf,ps1)
proj2=project(wf,ps2)

roc, thRange = calculate_roc(proj1,proj2)
auc = calculate_auc(roc)
#new code:
w_rand = np.random.uniform(size=(2,))-0.5
w_mean = m1 - m2
print('w Fisher:',wf)
print('w Rand:',w_rand)
print('w Fisher:',w_mean)

proj_rand1 = project(w_rand,ps1)
proj_rand2 = project(w_rand,ps2)
roc_rand,_=calculate_roc(proj_rand1,proj_rand2)
auc_rand =calculate_auc(roc_rand)

proj_mean1 =project(w_mean,ps1)
proj_mean2 =project(w_mean,ps2)
roc_mean,_ =calculate_roc(proj_mean1,proj_mean2)
auc_mean=calculate_auc(roc_mean)

print('')
print('AUC Fisher:{:.3f}'.format(auc))
print('AUC Random:{:.3f}'.format(auc_rand))
print('AUC Fisher:{:.3f}'.format(auc_mean))

fig, axes=plt.subplots(nrows=3,ncols=1,figsize=(8,16),gridspec_kw={'height_ratios':[2,1,1]})
plot_roc(roc,axes[0],color='m',label='Fisher')
plot_roc(roc_rand,axes[0],label='Random')
plot_roc(roc_mean,axes[0],label='Mean Intersection')
axes[0].legend(loc='best')

axes[1].hist(proj_rand1,25,alpha=0.5)
axes[1].hist(proj_rand2,25,alpha=0.5)
axes[1].grid(True)
axes[1].set_title('Project for Random Vector')

axes[2].hist(proj_mean1,25,alpha=0.5)
axes[2].hist(proj_mean2,25,alpha=0.5)
axes[2].grid(True)
axes[2].set_title('Project for Mean Intersection Vector')
