import utils.fileUtil as file
label_file = "./MNIST_data/mnist_train/train.txt"
lenth = file.getFileName(label_file)
print len(lenth)
print label_file[:-1*len(lenth)]