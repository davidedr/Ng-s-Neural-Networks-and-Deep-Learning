'''
Created on 29 ago 2017

@author: davide
'''

import os
import ftplib

DEBUG = 0
def upload(filename, dest_folder = '/www/public/'):
    session = ftplib.FTP('ftp.pollisrl.it','username','password')
    file = open(filename,'rb')                  # file to send
    dest_filename = dest_folder + filename
    response = session.storbinary('STOR ' + dest_filename, file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    if DEBUG > 0: print('response: ' + response)

def save_upload(data, filename):
    if DEBUG > 0: print('Save & upload: ' + filename + '...')
    if DEBUG > 0: print('Shape: ' + str(data.shape))
    np.save(filename, data)
    if DEBUG > 0: print(os.path.getsize(filename))
    upload(filename)

save_upload(train_set_x_orig, 'train_set_x_orig.npy')
save_upload(train_set_y, 'train_set_y.npy')
save_upload(test_set_x_orig, 'test_set_x_orig.npy')
save_upload(test_set_y, 'test_set_y.npy')

if DEBUG >0: print('Done.')