# ImageProcessing
Image processing by mrlittlepig

### 数据读取
将MNIST_data/mnist_data.zip解压到跟目录

利用[utils/genFileList.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/utils/genFileList.py)对数据进行处理生成label文件

利用[utils/labelFile2Map.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/utils/labelFile2Map.py)将label文件用字典形式保存在内存里面

在[datasets/imnist.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/datasets/imnist.py)中process_images函数用来读取图片文件夹，返回训练所用到的图片数据和标签数据

### 训练数据
在[classification.py](https://github.com/mrlittlepig/ImageProcessing/blob/master/classification.py)中定义了训练网络，运行开始训练

[中文博客](https://mrlittlepig.github.io/2017/04/30/tensorflow-for-image-processing/)
