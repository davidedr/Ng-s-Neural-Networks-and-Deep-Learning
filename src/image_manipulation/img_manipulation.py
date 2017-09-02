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

imgs = utils.get_celeb_imgs()
plt.figure()
plt.imshow(imgs[0])
plt.show()

# Turn into batch array
data = np.array(imgs)

img_mean = np.mean(data, axis = 0)
img_stddev = np.std(data, axis = 0)

plt.figure()
plt.suptitle("Dataset's mean and std dev images" , fontsize = 16)
plt.subplot(121)
plt.title('Mean image')
plt.imshow(img_mean.astype(np.uint8))
plt.subplot(122)
plt.title('Std dev image')
plt.imshow(img_stddev.astype(np.uint8))
plt.show()

