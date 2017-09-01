'''
Created on 29 ago 2017

@author: davide
'''

'''
    Demonstrates how to download images from a server, how to get them from disk,
    how to get them shown
'''

import os
import urllib.request
import ssl

OVERWRITE = 0

print("Downloading from AWS...")
ssl._create_default_https_context = ssl._create_unverified_context
landing_foldername = 'img_align_celeba'
if not os.path.isdir(landing_foldername): 
    os.mkdir(landing_foldername)
for img_index in range(1, 11):
    f = '000%03d.jpg' % img_index
    url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f
    print("\turl: '" + url + "'")
    
    image_filename = os.path.join(landing_foldername, f)
    if os.path.isfile(image_filename):
        if OVERWRITE > 0:
            os.remove(image_filename)
        else:
            continue
    urllib.request.urlretrieve(url, image_filename)
    
print('\tDownload complete.')

print("Listing images from the disk...")
files = os.listdir(landing_foldername)
print("\t" + str(files))

files = [file_i for file_i in os.listdir(landing_foldername) if '.jpg' in file_i]
print("\t" + str(files))

files = [file_i for file_i in os.listdir(landing_foldername)
        if '.jpg' in file_i or '.png' in file_i or '.jpeg' in file_i]
print("\t" + str(files))


print("Read images from disk...")
image_filename = os.path.join(landing_foldername, files[0])
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

print("Show image...")
img = plt.imread(image_filename)
plt.imshow(img)
plt.show()

print("\timg.shape: " + str(img.shape) + ".")

print("Show RED channel only...")
plt.figure()
plt.imshow(img[:, :, 0])
plt.show()

print('Done.')