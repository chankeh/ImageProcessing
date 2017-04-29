# -*- coding: utf-8 -*-
import cPickle
import numpy
from PIL import Image
from utils.labelFile2Map import *

i = 0;

train_images = numpy.empty((60000, 784))
train_labels = numpy.empty(60000)

image_root = "./MNIST_data/"
train_label = "./MNIST_data/mnist_train/train.txt"
test_label = "./MNIST_data/mnist_train/test.txt"

def process_train():
    lines = readLines(train_label)
    label_record = map(lines)
    train_dir = image_root + "mnist_train/"
    print len(label_record)
    index = 0
    for name in label_record:
        # print label_record[name]
        image = Image.open(train_dir + str(label_record[name]) + '/' + name)
        print "processing %d: " % index + train_dir + str(label_record[name]) + '/' + name

        img_ndarray = numpy.asarray(image, dtype='float64') / 256
        train_images[index] = numpy.ndarray.flatten(img_ndarray)
        train_labels[index] = numpy.int(label_record[name])

        # write_file = open('../PKLDataset/olivettifaces.pkl', 'wb')
        # cPickle.dump(train_images, write_file, -1)
        # cPickle.dump(train_labels, write_file, -1)
        # write_file.close()
        index = index + 1
    return train_images, train_labels

if __name__ == "__main__":
    process_train()