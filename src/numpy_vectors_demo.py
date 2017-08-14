'''
Created on 12 ago 2017

@author: davide
'''

'''
'''
import numpy as np

'''
    Use this
    a is a fully fledged and commonly understood vector
'''
N = 5
a = np.random.rand(N, 1) # Explicitly state the size
print(a)
print(a.shape)
print()

a_T = np.transpose(a)
print(a_T)
print(a_T.shape)
print()

'''
    Do NOT use this!
    a is a so called "rank one array" vector. Does NOT behave like a mathematical vector 
'''
a = np.random.rand(N)
print(a)
print(a.shape)
print()

a_T = np.transpose(a.T)
print(a_T)
print(a_T.shape)
print()

'''
    Anyway, you can convert a rank one array to a mathematical vector
    using reshape
'''
a_m = a.reshape(5, 1)
print(a_m)
print(a_m.shape)
print()
