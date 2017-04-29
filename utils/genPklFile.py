import cPickle
import os
import json
import pylab
import numpy
from PIL import Image

i = 0;

# r'data\Expression 1 文件一共含有165张，每张大小40*40.
# olivettifaces 则保存的就是这165张图片的信息。
# olivettifaces_label 中包含的是165张图片的标签信息

olivettifaces=numpy.empty((165,1600))
olivettifaces_label=numpy.empty(165)

#  下面这函数是列出文件夹中所有的文件，到filename中
for filename in os.listdir(r'data\Expression 1\Anger (AN)'):
    print filename
    if filename != 'Thumbs.db':
        basedir = 'data\Expression 1\Anger (AN)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)
        # 标签要从0开始，不然在cnn训练时会有错误
        olivettifaces_label[i]=0
        i = i + 1



for filename in os.listdir(r'data\Expression 1\Disgust (DI)'):
    print filename
    if(filename!='Thumbs.db'):
        basedir = 'data\Expression 1\Disgust (DI)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=1
        i = i + 1


for filename in os.listdir(r'data\Expression 1\Fear (FE)'):
    print filename
    if(filename!='Thumbs.db'):
        basedir = 'data\Expression 1\Fear (FE)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=2
        i = i + 1



for filename in os.listdir(r'data\Expression 1\Happiness (HA)'):
    print filename
    if(filename!='Thumbs.db'):
        basedir = 'data\Expression 1\Happiness (HA)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=3
        i = i + 1



for filename in os.listdir(r'data\Expression 1\Sadness (SA)'):
    print filename
    if(filename!='Thumbs.db'):
        basedir = 'data\Expression 1\Sadness (SA)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=4
        i = i + 1



for filename in os.listdir(r'data\Expression 1\Surprise (SU)'):
    print filename
    if(filename!='Thumbs.db'):
        basedir = 'data\Expression 1\Surprise (SU)/'
        imgage = Image.open(basedir + filename)

        img_ndarray = numpy.asarray(imgage, dtype='float64')/256
        olivettifaces[i]=numpy.ndarray.flatten(img_ndarray)

        olivettifaces_label[i]=5
        i = i + 1

olivettifaces_label=olivettifaces_label.astype(numpy.int)
# 下面是生成pkl格式的文件，保存数据。
write_file=open('olivettifaces.pkl','wb')
cPickle.dump(olivettifaces,write_file,-1)
cPickle.dump(olivettifaces_label,write_file,-1)
write_file.close()

# 从pkl文件中读取数据显示图像和标签。
read_file=open('olivettifaces.pkl','rb')
faces=cPickle.load(read_file)
label=cPickle.load(read_file)
read_file.close()
img0=faces[100].reshape(40,40)
pylab.imshow(img0)
pylab.gray()
pylab.show()
print label[0:165]