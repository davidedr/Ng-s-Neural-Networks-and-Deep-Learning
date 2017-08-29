'''
Created on 29 ago 2017

@author: davide
'''

import os
import zipfile

def zipdir(path, ziph, zipfilename, DEBUG = 0):
    for root, dirs, files in os.walk(path):
        for file in files:
            if DEBUG > 0:
                print(root + '/' + file)
                
            if zipfilename == file:
                if DEBUG > 0:
                    print("skip")
                continue 
                
            ziph.write(os.path.join(root, file))

def packit():
    rootdirname = '.'
    zipfilename = os.getcwd().split(os.sep)[-1]+'.zip'
    if os.path.isfile(zipfilename):
        os.remove(zipfilename)
    zipf = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_DEFLATED)
    zipdir(rootdirname, zipf, zipfilename)
    zipf.close()
    print("Paking done.")
    
packit()