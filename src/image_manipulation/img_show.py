'''
Created on 29 ago 2017

@author: davide
'''
from sklearn.datasets.lfw import _load_imgs

'''
    Demonstrates several procedures related to image manipulation
        Image reading form disk
        Plotting
        Plotting some RGB channels only
        Getting image properties
        Cropping to square
        Cropping to eliminate background
        Resize
        Normalize
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# How read an image and how to plot it?
print('Image loading and plotting...')
landing_foldername = 'img_align_celeba'
img_index = 1
f = '000%03d.jpg' % img_index
image_filename = os.path.join(landing_foldername, f)
print("\timage_filename: '" + image_filename + "'")

img = mpimg.imread(image_filename)
plt.imshow(img)
plt.show()

# How to get the shape of an image?
print('Getting image shape...')
print("\timg.shape: " + str(img.shape) + ".")

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
print("\tmin: " + str(np.min(img)) + ", max: " + str(np.max(img)) + ".")
print("\timg.dtype: " + str(img.dtype) + ".")

img.astype(np.float32)
print("\timg.dtype: " + str(img.dtype) + ".")

# How to choose an image at random?
print('Choosing an image at random...')
files = [file_i for file_i in os.listdir(landing_foldername)
        if '.jpg' in file_i or '.png' in file_i or '.jpeg' in file_i]

print("\t" + str(np.random.randint(0, len(files))))
print("\t" + str(np.random.randint(0, len(files))))
print("\t" + str(np.random.randint(0, len(files))))

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
def imcrop_tosquare(img, DEBUG = 0):
    print("\timg.shape: " + str(img.shape))
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
        
    
    if DEBUG > 0:
        print('\timg.shape: ' + str(img.shape) + ', crop.shape: ' + str(crop.shape) + '.')
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

# How to compute the mean color of each pixel?
print('Compute mean color of each pixel and display it in gray scale...')
img_mean = np.mean(img_rsz, axis = 2)
print("\t" + str(img_mean.shape))
plt.imshow(img_mean, cmap='gray')
plt.show()

# How to normalize a set of images?
print("Normalize a set of images: crop to square to remove the longer edge, crop to remove some background, resize to 64 x 64...")
to_be_shown_filename = files[np.random.randint(0, len(files))] 
imgs = []
for filename in files:
    img = plt.imread(os.path.join(landing_foldername, filename))
    
    img_square = imcrop_tosquare(img)
    img_crop = imcrop(img_square, 0.2)
    img_resize = imresize(img_crop, (64, 64))
    
    imgs.append(img_resize)
    if filename == to_be_shown_filename:
        print("\tShow an example: '" + filename + "'...")
        plt.figure()
        plt.subplot(141)
        plt.imshow(img)
        plt.subplot(142)
        plt.imshow(img_square)
        plt.subplot(143)
        plt.imshow(img_crop)
        plt.subplot(144)
        plt.imshow(img_resize)
        plt.show()

# How to deal with the batch dimension N?
# Convert an array of images to a single array N x H x W x C
# N = number of images in the batch
# H = height or number of rows in each image
# W = width or number of cols in each image
# C = number of channels (colors)in the image (RGB: 3, Grayscale: 1)
#
# batch size = N x H x W x C
print("Converting to a batch...")
print("\tsingle image shape: " + str(imgs[0].shape) + ".")
print("\tnumber of images: " + str(len(imgs)) + ".")
data = np.array(imgs)
print("\tbatch size: " + str(data.shape) + ".")
print()
print("\talternative way...")
data = np.concatenate([img_i[np.newaxis] for img_i in imgs], axis=0)
print("\tbatch size: " + str(data.shape) + ".")
print('Done')   