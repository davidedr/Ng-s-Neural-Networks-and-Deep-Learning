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
print("Read and plot image n. " + str(n) + ", " + files[n] + "...")
img = plt.imread(files[n])
print('image no. ' + str(n) + ', ' + files[n] + ', has shape: ' + str(img.shape) + '.')
print(img)
plt.imshow(img)
plt.title('image no. ' + str(n) + ', ' + files[n])
plt.show()

print("Plot image n. " + str(n) + ", " + files[n] + " as separate RGB channels...")
plt.figure()
plt.subplot(141)
plt.imshow(img)
plt.subplot(142)
plt.imshow(img[:, :, 0], cmap = 'Reds')
plt.subplot(143)
plt.imshow(img[:, :, 1], cmap = 'Greens')
plt.subplot(144)
plt.imshow(img[:, :, 2], cmap = 'Blues')
plt.show()

print("Read all images....")
imgs = utils.get_celeb_imgs()
n = list(utils.sample_without_replacement(len(files), 1))[0]
plt.figure()
plt.imshow(imgs[n])
plt.title("Example image: " + str(n) + ", " + files[n])
plt.show()

# Turn into batch array
data = np.array(imgs)

print("Compute and plot mean and standard deviation images...")
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

# Histogram of R, G, B for the n-th image
print("Compute and plot histograms and estimated densities...")
n = list(utils.sample_without_replacement(len(files), 1))[0]
plt.figure()
plt.subplot(141)
plt.imshow(imgs[n])
plt.subplot(142)
plt.hist(imgs[n][:, :, 0])
plt.subplot(143)
plt.hist(imgs[n][:, :, 1])
plt.subplot(144)
plt.hist(imgs[n][:, :, 2])
plt.show()

plt.figure()
kwargs = dict(histtype = 'stepfilled', alpha = 0.3, bins = 255)
plt.hist(imgs[n][:, :, 0].ravel(), **kwargs)
plt.hist(imgs[n][:, :, 1].ravel(), **kwargs)
plt.hist(imgs[n][:, :, 2].ravel(), **kwargs)
plt.title("RGB histogram of image " + str(n) + ", " + files[n] + ".\n(matplotlib)")
plt.show()

import seaborn as sns
sns.set(color_codes=True)
sns.distplot(imgs[n][:, :, 0].ravel())
plt.title("Red histogram of image " + str(n) + ", " + files[n] + ".\n(seaborn)")
plt.show()

colours = ('r', 'g', 'b')
for i, colour in enumerate(colours):
    sns.distplot(imgs[n][:, :, i].ravel(), color = colour)
plt.title("RGB histogram of image " + str(n) + ", " + files[n] + ".\n(seaborn)")
plt.show()

plt.hist(np.array(imgs).ravel(), 255)
plt.title('Histogram of all images: np.array(imgs).ravel()\n(matplotlib)')
plt.show()

colours = ('r', 'g', 'b')
for i, colour in enumerate(colours):
    sns.distplot(img_mean[:, :, i].ravel(), color = colour)
plt.title("RGB histogram of MEAN image\n(seaborn)")
plt.show()

# Images normalization
plt.figure()
plt.subplot(321)
plt.imshow(img[n])
plt.subplot(322)
sns.distplot(img[n].ravel())
plt.title('Image'+ str(n) + ", " + files[n])

plt.subplot(323)
plt.imshow(img_mean.astype(np.uint8))
plt.subplot(324)
sns.distplot(img_mean.astype(np.uint8).ravel())
plt.title('Mean image')

img_nomean = img[n] - img_mean
plt.subplot(325)
plt.imshow(img_nomean.astype(np.uint8))
plt.subplot(326)
sns.distplot(img_nomean.astype(np.uint8).ravel())
plt.title('Image  - mean image')
plt.show()