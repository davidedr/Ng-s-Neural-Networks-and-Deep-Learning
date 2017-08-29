'''
Created on 29 ago 2017

@author: davide
'''

import os
import zipfile

def zipdir(path, ziph, zipfilename):
    for root, dirs, files in os.walk(path):
        for file in files:
            #print(root + '/' + file)
            if zipfilename == file:
                print("skip")
                continue 
            ziph.write(os.path.join(root, file))

if __name__ == '__main__':
    cwd = os.getcwd()
    print(cwd)
    print(__file__)
    
    full_path = os.path.realpath(__file__)
    
    print(os.path.dirname(full_path))
    print("This file directory and name")
    path, filename = os.path.split(full_path)
    print(path + ' --> ' + filename + "\n")    

    print(os.getcwd().split('\\')[-1])
    print(os.getcwd().split(os.sep)[-1])
        
    rootdirname = 'C:/Nuova cartella/Neural Networks and Deep Learning/code'
    #zipfilename = 'C:/Python.zip'
    zipfilename = 'Python.zip'
    if os.path.isfile(zipfilename):
        os.remove(zipfilename)
    zipf = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED)
    zipdir(rootdirname, zipf, zipfilename)
    zipf.close()
    