'''
Created on 11 ago 2017

@author: davide
'''

'''
    Computes and explains the log-loss function
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def log_loss(y_hat, y):
    '''
        Computes the log-loss function
        params: y_hat: estimate of true value
                y: true value
    '''
    value = -(y*np.log(y_hat) + (1 - y)*np.log(1 - y_hat))
    return value

def log_loss_3dplot():
    N = 100 # number of samples
    delta = 0.01
    y = np.linspace(0+delta, 1-delta, N) 
    y_hat = np.linspace(0+delta, 1-delta, N)
    y, y_hat = np.meshgrid(y, y_hat)
    Z = log_loss(y,  y_hat)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(y, y_hat, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
  
    fig = plt.figure()    
    ax = fig.gca(projection='3d')
    ax.plot_surface(y, y_hat, Z, rstride = 3, cstride = 3, linewidth = 1, antialiased=True, cmap = cm.viridis)
    cset = ax.contourf(y, y_hat, Z, zdir = 'z', offset = -0.15, cmap = cm.viridis)
    
    ax.view_init(27, -21)
    plt.show()

log_loss_3dplot()

y = 1
y_hat = np.linspace(0, 1, int(1E3))
l = log_loss(y_hat, y)
_ = plt.plot(y_hat, l)
_ = plt.xlabel('y_hat')
_ = plt.ylabel('log loss')
_ = plt.suptitle("Log loss as a function of y_hat when the truth is 1")
plt.margins(0.02)
plt.show()

y = 0
y_hat = np.linspace(0, 1, int(1E3))
l = log_loss(y_hat, y)
_ = plt.plot(y_hat, l)
_ = plt.xlabel('y_hat')
_ = plt.ylabel('log loss')
_ = plt.suptitle("Log loss as a function of y_hat when the truth is 0")
plt.margins(0.02)
plt.show()
