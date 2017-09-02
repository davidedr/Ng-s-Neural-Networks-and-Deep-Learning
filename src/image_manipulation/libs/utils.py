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

'''
    Test
'''
if __name__ == "__main__":
    get_celeb_files(10, DEBUG = 1)