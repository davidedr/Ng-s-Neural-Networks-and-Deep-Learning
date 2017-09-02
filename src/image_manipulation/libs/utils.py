'''
Created on 02 set 2017

@author: davide
'''

import urllib.request
import os

def get_celeb_files(N = 100, foldername = 'img_align_celeba', overwrite = False, DEBUG = 0):
    '''
        Download first N images form the celebs images dataset
        store in foldername folder
        
        Returns
        -------
        Array of filenames
    '''
    
    if not os.path.exists(foldername):
        os.mkdir(foldername)
        
    for i in range(0, N):
        img_index = i+1
        filename = '000%03d.jpg' % img_index
        filename_pathcomplete = os.path.join(foldername, filename)
        
        if (os.path.exists(filename_pathcomplete) and not overwrite):
            continue
        
        url = 'https://s3.amazonaws.com/cadl/celeb-align/' + filename
        if DEBUG > 0:
            print(filename_pathcomplete + ", " + url)
        urllib.request.urlretrieve(url, filename_pathcomplete)
        
    files = [os.path.join(foldername, filename_i) for filename_i in os.listdir(foldername) if '.jpg' in filename_i]
    return files
        
if __name__ == "__main__":
    get_celeb_files(10, DEBUG = 1)
    
import random
def sample_without_replacement(N, r):
    '''
        Generate r randomly chosen, sorted integers in [1, N]
    '''
    rand = random.random
    pop = N
    for samp in range(r, 0, -1):
        cum_prob = 1.0
        x = rand()
        while x < cum_prob:
            cum_prob -= cum_prob*samp/pop
            pop -= 1
        yield (N - pop - 1) + 1

if __name__ == "__main__":
    print(list(sample_without_replacement(100, 10)))
    '''
        TODO: check numbers distribution in given list. Should be uniform
    '''
    
import matplotlib.pyplot as plt

def get_celeb_imgs(N = 100, foldername = 'img_align_celeba', overwrite = False, DEBUG = 0, r = 10, random = False):
    '''
        
    '''
    files = get_celeb_files(N, foldername, overwrite, DEBUG)
    imgs = [plt.imread(filename_pathcomplete) for filename_pathcomplete in files]
    return imgs

if __name__ == "__main__":
    imgs = get_celeb_imgs()
    print(len(imgs))
    print(imgs[0].shape)