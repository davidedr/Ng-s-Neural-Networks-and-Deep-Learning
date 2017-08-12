'''
Created on 12 ago 2017

@author: davide
'''

'''
    Computes the derivative wrt y_hat of the log loss function
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def der_log_loss(y_hat, y):
    value = -y/y_hat + (1 - y)/(1 - y_hat)
    return value

y = 1
y_hat = np.linspace(0, 1, int(1E3))
l = der_log_loss(y_hat, y)
_ = plt.plot(y_hat, l)
_ = plt.xlabel('y_hat')
_ = plt.ylabel('log loss')
_ = plt.suptitle("Derivative of Log loss as a function of y_hat when the truth is 1")
plt.margins(0.02)
plt.show()

y = 0
y_hat = np.linspace(0, 1, int(1E3))
l = der_log_loss(y_hat, y)
_ = plt.plot(y_hat, l)
_ = plt.xlabel('y_hat')
_ = plt.ylabel('log loss')
_ = plt.suptitle("Derivative Log loss as a function of y_hat when the truth is 0")
plt.margins(0.02)
plt.show()
