import glob
import os
import skimage.io as skio
import Image

IMG_DIR = './tmp/files.txt'
DST_FILE = '/tmp/files2.txt'
back_end = '*.bmp'

def read_img(image_dir):
    name_list = glob.glob(image_dir + os.sep + back_end)
    for i in range(len(name_list)):
        image = skio.imread(name_list[i])
        sp = name_list[i].split('.bmp')
        skio.imsave(sp[0] + '.jpg', image)
        print("processing " + str(i) + ": " + sp[0] + '.jpg')


imgslist = glob.glob('./tmp/*.jpg')

def small_img():
    for imgs in imgslist:
        imgspath, ext = os.path.splitext(imgs)
        img = Image.open(imgs)
        (x,y) = img.size
        small_img =img.resize((720,1024),Image.ANTIALIAS)
        small_img.save(imgs)
    print("done")


if __name__ == '__main__':
    # read_img(IMG_DIR)
    small_img(IMG_DIR)