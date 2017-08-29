'''
Created on 29 ago 2017

@author: davide
'''

import os

import urllib.request

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

landing_foldername = 'img_align_celeba'
if not os.path.isdir(landing_foldername): 
    os.mkdir(landing_foldername)
for img_index in range(1, 11):
    f = '000%03d.jpg' % img_index
    url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f
    print("url: '" + url + "'")
    
    image_filename = os.path.join(landing_foldername, f)
    if os.path.isfile(image_filename):
        os.remove(image_filename)
    urllib.request.urlretrieve(url, image_filename)
    
print('Download complete.')

files = os.listdir(landing_foldername)
print(files)

files = [file_i for file_i in os.listdir(landing_foldername) if '.jpg' in file_i]
print(files)

files = [file_i for file_i in os.listdir(landing_foldername)
        if '.jpg' in file_i or '.png' in file_i or '.jpeg' in file_i]
print(files)

import matplotlib.pyplot as plt
import numpy as np

image_filename = os.path.join(landing_foldername, files[0])

import matplotlib.image as mpimg
plt.imshow(mpimg.imread(image_filename))

img = plt.imread(image_filename)
plt.imshow(img)

print("img.shape: " + str(img.shape) + ".")

plt.figure()
plt.imgshow(img[:, :, 0])
plt.figure()

plt.imshow(img[:, :, 0])
plt.figure()

print('Done.')