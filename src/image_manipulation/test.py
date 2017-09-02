'''
Created on 02 set 2017

@author: davide
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(100.0)
x.shape = (10, 10)

interp = 'bilinear'
#interp = 'nearest'
lim = -2, 11, -2, 6
plt.subplot(211, axisbg='g')
plt.title('blue should be up')
plt.imshow(x, origin='upper', interpolation=interp, cmap='jet')
#plt.axis(lim)

plt.subplot(212, axisbg='y')
plt.title('blue should be down')
plt.imshow(x, origin='lower', interpolation=interp, cmap='jet')
#plt.axis(lim)
plt.show()