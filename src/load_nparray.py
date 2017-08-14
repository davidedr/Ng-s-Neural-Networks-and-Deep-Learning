'''
Created on 14 ago 2017

@author: davide
'''

'''
    Read data from files ftp-uploaded from Jupiter notebook 
'''
import numpy as np

def read_from_npy_file(filename):
    print('Reading from file...')
    data = np.load(filename)
   
    return data

train_set_x_orig = read_from_npy_file('./../data/train_set_x_orig.npy')
print(train_set_x_orig.shape)

train_set_y = read_from_npy_file('./../data/train_set_y.npy')
print(train_set_y.shape)

test_set_x_orig = read_from_npy_file('./../data/test_set_x_orig.npy')
print(test_set_x_orig.shape)

test_set_y = read_from_npy_file('./../data/test_set_y.npy')
print(test_set_y.shape)

