'''
Created on 02 set 2017

@author: davide
'''

'''
    Demonstrate some statistical manipulation on the celebs images dataset
    Loosily based on Kadenze's Parag K. Mital course
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from libs import utils

files = utils.get_celeb_files()
n = 50
img = plt.imread(files[n])
print('image no. ' + str(n) + ', ' + files[n] + ', has shape: ' + str(img.shape) + '.')
print(img)
plt.imshow(img)
plt.show()

plt.figure()
plt.subplot(131)
plt.imshow(img[:, :, 0], cmap = 'gray')
plt.subplot(132)
plt.imshow(img[:, :, 1], cmap = 'gray')
plt.subplot(133)
plt.imshow(img[:, :, 2], cmap = 'gray')
plt.show()