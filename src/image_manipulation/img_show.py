'''
Created on 29 ago 2017

@author: davide
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

landing_foldername = 'img_align_celeba'
img_index = 1
f = '000%03d.jpg' % img_index
image_filename = os.path.join(landing_foldername, f)
print("image_filename: '" + image_filename + "'")

# How to plot an image?
print('Image plotting...')
img = mpimg.imread(image_filename)
plt.imshow(img)
plt.show()

# How to get the shape of an image?
print('Getting image shape...')
print("img.shape: " + str(img.shape) + ".")

# How to make multiple plots?
# How to plot the different channels of an  image?
print('Plotting image RGB channels...')
plt.figure(1)
plt.subplot(131)
plt.imshow(img[:, :, 0])

plt.subplot(132)
plt.imshow(img[:, :, 1])

plt.subplot(133)
plt.imshow(img[:, :, 2])

plt.show()

# How to get image elements type and range?
print('Getting image elements type and range...')
print("min: " + str(np.min(img)) + ", max: " + str(np.max(img)) + ".")
print("img.dtype: " + str(img.dtype) + ".")

img.astype(np.float32)
print("img.dtype: " + str(img.dtype) + ".")

# How to choose an image at random?
print('Choosing an image at random...')
files = [file_i for file_i in os.listdir(landing_foldername)
        if '.jpg' in file_i or '.png' in file_i or '.jpeg' in file_i]

print(np.random.randint(0, len(files)))
print(np.random.randint(0, len(files)))
print(np.random.randint(0, len(files)))

# ok now stop kidding:
filename = files[np.random.randint(0, len(files))]
img = plt.imread(os.path.join(landing_foldername, filename))
plt.imshow(img)
plt.show()

def plot_image(filename, foldername = 'img_align_celeba'):
    img = plt.imread(os.path.join(foldername, filename))
    plt.imshow(img)
    plt.show()
    
print("Using plot_imqage:")
plot_image(files[np.random.randint(0, len(files))])

# How to crop an image to square dimensions?
def imcrop_tosquare(img):
    print(img.shape)
    if img.shape[0] > img.shape[1]:
        extra = img.shape[0] - img.shape[1]
        if extra % 2 == 0:
            crop = img[extra // 2: -extra // 2, :]
        else:
            crop = img[max(0, extra // 2 + 1): min(-1, -(extra // 2)), :]
    elif img.shape[0] < img.shape[1]:
        extra = img.shape[1] - img.shape[0]
        if extra % 2 == 0:
            crop = img[:, extra // 2: -extra // 2]
        else:
            crop = img[:, max(0, extra // 2 + 1): min(-1, -(extra // 2))]
    else:
        crop = img
        
    print('img.shape: ' + str(img.shape) + ', crop.shape: ' + str(crop.shape) + '.')
    return  crop

print('Cropping an image to square size...')
filename = files[np.random.randint(0, len(files))]
img = plt.imread(os.path.join(landing_foldername, filename))
crop = imcrop_tosquare(img)

plt.figure(2)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(crop)
plt.show()

# How to crop an image getting a definite % from the centre of the image?
def imcrop(img, amt):
    if amt <= 0 or amt >= 1:
        return img
    row_i = int(img.shape[0]*amt)//2
    col_i = int(img.shape[1]*amt)//2
    crop = img[row_i:-row_i, col_i:-col_i]
    return crop

print('Cropping an image to a definite %...')
filename = files[np.random.randint(0, len(files))]
img = plt.imread(os.path.join(landing_foldername, filename))
crop = imcrop(img, 0.25)

plt.figure(2)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(crop)
plt.show()

# How to resize an image?
print('Image resizing...')
from scipy.misc import imresize
filename = files[np.random.randint(0, len(files))]
img = plt.imread(os.path.join(landing_foldername, filename))
img_square = imcrop_tosquare(img)
img_rsz = imresize(img_square, (64, 64))
plt.figure(3)
plt.subplot(141)
plt.imshow(img)
plt.subplot(142)
plt.imshow(img_square)
plt.subplot(143)
plt.imshow(img_rsz)
plt.subplot(144)
plt.imshow(img_rsz, interpolation = 'nearest')
plt.show()

plt.figure()
plt.imshow(img_rsz, interpolation = 'nearest')
plt.show()
