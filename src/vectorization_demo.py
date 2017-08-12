'''
Created on 12 ago 2017

@author: davide
'''

'''
    Compares performance of vectorized and non-vectorized versions
    of dot product computation
'''

import numpy as np
import time

N = int(1E6)
a = np.random.rand(N)
b = np.random.rand(N)

'''
    Vectorized
'''
tic = time.time()
c1 = np.dot(a, b)
toc = time.time()

print("Vectorized dot product of two " + str(N) + "-sized vectors: " + str(1000*(toc - tic)) + " ms")

'''
    Non vectorized
'''
tic = time.time()
c2 = 0
for i in range(len(a)):
    c2 += a[i]*b[i]
toc = time.time()

print("Non vectorized dot product of two " + str(N) + "-sized vectors: " + str(1000*(toc - tic)) + " ms")

'''
    Comparison of summing elements of a vector
'''
print()
tic = time.time()
c3 = np.sum(a)
toc = time.time()

print("Vectorized sum of elements of " + str(N) + "-sized vector: " + str(1000*(toc - tic)) + " ms")

tic = time.time()
c4 = 0
for i in range(len(a)):
    c4 += a[i]
toc = time.time()

print("Non vectorized sum of elements of " + str(N) + "-sized vector: " + str(1000*(toc - tic)) + " ms")
