# -*- coding: utf-8 -*-
import cPickle
import numpy
import utils.fileUtil as file
from PIL import Image
from utils.labelFile2Map import *

def process_images(label_file, one_hot=False, num_classes=10):
    if file.getFileName(label_file) == 'train.txt':
        images = numpy.empty((60000, 784))
        labels = numpy.empty(60000)
    if file.getFileName(label_file) == 'test.txt':
        images = numpy.empty((10000, 784))
        labels = numpy.empty(10000)
    lines = readLines(label_file)
    label_record = map(lines)
    file_name_length = len(file.getFileName(label_file))
    image_dir = label_file[:-1*file_name_length]
    print len(label_record)
    index = 0
    for name in label_record:
        # print label_record[name]
        image = Image.open(image_dir + str(label_record[name]) + '/' + name)
        print "processing %d: " % index + image_dir + str(label_record[name]) + '/' + name

        img_ndarray = numpy.asarray(image, dtype='float32')
        images[index] = numpy.ndarray.flatten(img_ndarray)
        labels[index] = numpy.int(label_record[name])

        index = index + 1
    print index
    num_images = index
    rows = 28
    cols = 28
    # print train_images.reshape(num_images, rows, cols, 1)numpy.fromarrays(train_labels,)
    # print numpy.array(train_labels, dtype=numpy.uint8)
    if one_hot:
      return images.reshape(num_images, rows, cols, 1), dense_to_one_hot(numpy.array(labels, dtype=numpy.uint8), num_classes)
    return images.reshape(num_images, rows, cols, 1), numpy.array(labels, dtype=numpy.uint8)

if __name__ == "__main__":
    image_root = "./MNIST_data/"
    train_label = "./MNIST_data/mnist_train/train.txt"
    test_label = "./MNIST_data/mnist_test/test.txt"
    process_images(image_root, train_label)