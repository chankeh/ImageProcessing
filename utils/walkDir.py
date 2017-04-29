import os

rootDir = './'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Folder: %s' % dirName)
    for fname in fileList:
        print('\t%s' % fname)