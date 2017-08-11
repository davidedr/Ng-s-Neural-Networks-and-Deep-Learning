'''
Created on 11 ago 2017

@author: davide
'''

'''
    Computes and explain 
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(z):
    '''
        Computes the sigmoid function for the given value z
    '''
    value = 1/(1 + np.exp(-z))
    return value

print(sigmoid(1E5))
print(1E-5)
print(sigmoid(1E-5))
print(sigmoid(0.00001))

z = np.linspace(-10, 10, int(1E3))
sigma = sigmoid(z)
print(sigma)

sns.set()
_ = plt.plot(z, sigma)
_ = plt.xlabel('z')
_ = plt.ylabel('sigma(z)')
plt.margins(0.02)
plt.show()
