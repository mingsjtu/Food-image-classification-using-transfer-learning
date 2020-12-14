from random import normalvariate
import numpy as np
import os
from PIL import Image
from random import randint

IMAGE_PATH = "/home2/qianchen/Multilabel-Classification/course_data/MTFood-1000/train/"


# 等比例地把图片较短的一边缩放到区间[256,480]
def rescale(image):
    w = image.size[0]
    h = image.size[1]
    random_size = 224
    if w < h:
        return image.resize((random_size, round(h / w * random_size)))
    else:
        return image.resize((round(w / h * random_size), random_size))


# 随机裁剪图片
def random_crop(image):
    w = image.size[0]
    h = image.size[1]
    size = 224
    new_left = randint(0, w - size)
    new_upper = randint(0, h - size)
    return image.crop((new_left, new_upper, size + new_left, size + new_upper))


# 水平翻转图片
def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


# 图片均值归一化
def nomalizing(image, mean_value, add_num):
    image = np.array(image)
    # image = image.astype(float)
    for i in range(3):
        add_num = add_num.astype(int)
        image[:, :, i] = (image[:, :, i] - mean_value[i] + add_num[i])
    return image


if __name__ == '__main__':
    '''
    with open('./MTFood-1000/val_list','r') as f:
      train = f.read().splitlines()
    
    for i in range(len(train)):
      item = train[i]
      #current_label = item[:item.rfind('_')]
      #path = os.path.join('./train', str)
      path = './MTFood-1000/val'
      #current_label = item[:item.rfind('_')]
      train_set = [os.path.join(path, file) for file in os.listdir(path)]
    '''
    path = './MTFood-1000/test'
    test_set = [os.path.join(path, file) for file in os.listdir(path)]
    for file in test_set:
        # print(file)
        img = Image.open(file)
        img = rescale(img)
        # file, 1), cv2.COLOR_BGR2RGB))
        img = random_crop(img)
        # img = nomalizing(img,mean_value, add_num)
        img.save(file)
