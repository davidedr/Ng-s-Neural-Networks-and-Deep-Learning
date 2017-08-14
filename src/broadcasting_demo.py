'''
Created on 12 ago 2017

@author: davide
'''

'''
'''

import numpy as np

A = np.matrix([
    [56., 0., 4.4, 68.],
    [1.2, 104., 52., 8.],
    [1.8, 135., 99., 0.9]
    ])

print(A)
print(A.shape)
print()

S1 = np.sum(A, axis = 0)

print(S1)
print(S1.shape)
print()

S2 = A.sum(axis = 0)
print(S2)
print(S2.shape)
print()

P = A/S2*100
print(P)
print(P.shape)
print()
